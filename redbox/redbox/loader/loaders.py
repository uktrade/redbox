import logging
import time
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING, List, Tuple

import environ
import fitz
import requests
from requests.exceptions import RequestException

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError
from redbox_app.setting_enums import Environment

from redbox.redbox.chains.components import get_chat_llm
from redbox.redbox.chains.parser import ClaudeParser
from redbox.redbox.models.chain import GeneratedMetadata
from redbox.redbox.models.file import ChunkResolution, UploadedFileMetadata
from redbox.redbox.models.settings import Settings
from redbox.redbox.transform import bedrock_tokeniser
import pandas as pd
import math
import time as _time

env = environ.Env()
ENVIRONMENT = Environment[env.str("ENVIRONMENT").upper()]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokeniser = bedrock_tokeniser

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object


def is_large_pdf(file_name: str, filebytes: BytesIO, page_threshold: int = 150) -> Tuple[bool, int]:
    if not file_name.lower().endswith(".pdf"):
        return False, 0
    try:
        doc = fitz.open(stream=filebytes.getvalue(), filetype="pdf")
        return len(doc) > page_threshold, len(doc)
    except Exception as e:
        logger.warning("error opening PDF - %s", e)
        # assume its not large if you can't open it
        return False, 0


def split_pdf(filebytes: BytesIO, pages_per_chunk: int = 75) -> List[BytesIO]:
    doc = fitz.open(stream=filebytes.getvalue(), filetype="pdf")
    chunks: List[BytesIO] = []
    total_pages = len(doc)
    if total_pages == 0:
        return chunks

    for start in range(0, total_pages, pages_per_chunk):
        sub_doc = fitz.open()
        end = min(start + pages_per_chunk, total_pages)
        sub_doc.insert_pdf(doc, from_page=start, to_page=end)
        if len(sub_doc) == 0:
            continue  # Skip empty chunks
        chunk_bytes = BytesIO(sub_doc.tobytes())
        chunks.append(chunk_bytes)
    return chunks


def _pdf_is_image_heavy(file_bytes: BytesIO, sample_pages: int = 5, image_threshold: int = 1) -> bool:
    try:
        doc = fitz.open(stream=file_bytes.getvalue(), filetype="pdf")
        pages_to_check = min(len(doc), sample_pages)
        images_found = 0
        for i in range(pages_to_check):
            page = doc[i]
            images = page.get_images(full=True)
            if images:
                images_found += 1
        # if more than half of sampled pages have images, file == image-heavy
        return images_found >= math.ceil(pages_to_check / 2)
    except Exception as e:
        logger.debug("can't work out quantity of images for the file - %s", e)
        return False


def read_csv_text(file_bytes: BytesIO) -> list[dict[str, str | dict]]:
    """Reads in a csv file, validates it using pandas and then returns the csv as string with a null metadata dictionary"""
    try:
        file_bytes.seek(0)
        # Read bytes into pandas df. This acts as a pre-check that the csv is well formed
        df = pd.read_csv(file_bytes)
        if df.empty:
            logger.error("Empty File Uploaded")
            raise ValidationError("Empty File Uploaded")
        return [
            {
                "text": str(df.to_csv(index=False)),  # Convert bytes to string
                "metadata": {},  # returning empty metadata dictionary
            }
        ]
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        logger.error(f"Error while trying to upload csv file {e}")
        return None


def read_excel_file(file_bytes: BytesIO) -> list[dict[str, str | dict]]:
    """Reads in an excel file, validates each sheet using pandas and then returns a list of each valid sheet as string with a null metadata dictionary"""
    try:
        sheets = pd.read_excel(file_bytes, sheet_name=None)
        elements = []

        for name, df in sheets.items():
            try:
                if df.empty:
                    logger.info(f"Skipping Sheet {name}")
                    continue
                # Include the table name in the text that is stored. This will be extracted by the retriever
                csv_text = f"<table_name>{name.lower().replace(' ', '_')}</table_name>" + str(df.to_csv(index=False))
                elements.append({"text": csv_text, "metadata": {}})
            except Exception as e:
                logger.info(f"Skipping Sheet {name} due to error: {e}")
                continue
        return elements if len(elements) else None
    except Exception as e:
        logger.error(f"Excel Read Error: {e}")
        return None


def load_tabular_file(file_name: str, file_bytes: BytesIO) -> list[dict[str, str]]:
    """Selects the right read method for each file type. Returns an empty list if n"""
    if file_name.endswith(".csv"):
        elements = read_csv_text(file_bytes=file_bytes)
    else:
        elements = read_excel_file(file_bytes=file_bytes)

    return elements if elements else []


class UnstructuredChunkLoader:
    """
    Load, partition and chunk a document using local unstructured library.

    Notes:
    - Large PDFs - split into smaller PDF files and call the API per chunk
    - Image heavy PDFs - we avoid the fast strategy to prevent the "fast not available for image files" error
    - Exponential backoff for transient errors
    """

    def __init__(
        self,
        chunk_resolution: ChunkResolution,
        env: Settings,
        min_chunk_size: int,
        max_chunk_size: int,
        metadata: GeneratedMetadata | None = None,
        overlap_chars: int = 0,
        overlap_all_chunks: bool = True,
        request_timeout: int = 480,
        max_retries: int = 3,
        pages_per_pdf_chunk: int = 75,
    ):
        self.chunk_resolution = chunk_resolution
        self.env = env
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._overlap_chars = overlap_chars
        self._overlap_all_chunks = overlap_all_chunks
        self.metadata = metadata or GeneratedMetadata(name="", description="", keywords=[])
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.pages_per_pdf_chunk = pages_per_pdf_chunk

    def _get_chunks(self, file_name: str, file_bytes: BytesIO) -> List[dict]:
        """Helper method to perform chunking via the unstructured API with robust fallback behaviour."""
        if ENVIRONMENT.is_local:
            url = f"http://{self.env.unstructured_host}:8000/general/v0/general"
        else:
            url = f"http://{self.env.unstructured_host}/general/v0/general"

        is_large, page_count = is_large_pdf(file_name=file_name, filebytes=file_bytes)
        is_tabular = file_name.endswith((".csv", ".xls", ".xlsx"))

        if is_large and page_count > 0:
            logger.info(
                "Large PDF with (%d pages) - splitting into chunks with %d pages", page_count, self.pages_per_pdf_chunk
            )
            elements: List[dict] = []
            pdf_chunks = split_pdf(filebytes=file_bytes, pages_per_chunk=self.pages_per_pdf_chunk)
            for idx, chunk in enumerate(pdf_chunks):
                chunk.seek(0)
                files = {"files": (file_name, chunk)}
                try:
                    chunk_elements = self._post_files_with_fallback(
                        url=url, files=files, file_name=file_name, file_bytes=chunk
                    )
                except Exception as e:
                    msg = f"Chunk {idx + 1} failed: {e}"
                    logger.exception(msg)
                    raise ValueError(msg)
                elements.extend(chunk_elements)
            logger.debug("Unstructured returned %d elements", len(elements))
            return elements

        elif is_tabular and self.chunk_resolution == ChunkResolution.tabular:
            # Carry out the special ingest process for tabular files - will be carried out in addition to
            elements = load_tabular_file(file_name=file_name, file_bytes=file_bytes)
            logger.debug("Unstructured returned %d elements", len(elements))
            return elements

        try:
            file_bytes.seek(0)
        except Exception as e:
            logger.warning("Unable to seek file %s before upload - %s", file_name, str(e))

        files = {"files": (file_name, file_bytes)}
        elements = self._post_files_with_fallback(url=url, files=files, file_name=file_name, file_bytes=file_bytes)
        if not elements:
            raise ValueError("Unstructured failed to extract text for this file")
        logger.debug("Unstructured returned %d elements", len(elements))
        return elements

    def _post_files_with_fallback(self, url: str, files: dict, file_name: str, file_bytes: BytesIO) -> List[dict]:
        try:
            file_bytes.seek(0)
        except Exception as e:
            logger.warning("Unable to seek file %s before upload - %s", file_name, e)

        # build default data payload
        base_data = {
            "chunking_strategy": "by_title",
            "max_characters": self._max_chunk_size,
            "combine_under_n_chars": self._min_chunk_size,
            "overlap": self._overlap_chars,
            "overlap_all": str(self._overlap_all_chunks).lower(),
            "infer_table_structure": "true",
        }

        # detect if file is an image-heavy pdf
        lower_name = file_name.lower()
        is_pdf_image_heavy = False
        if lower_name.endswith(".pdf"):
            try:
                is_pdf_image_heavy = _pdf_is_image_heavy(file_bytes)
            except Exception:
                is_pdf_image_heavy = False

        logger.debug("file %s pdf_image_heavy=%s", file_name, is_pdf_image_heavy)

        candidate_data_payloads = []

        # try fast strategy first
        if not is_pdf_image_heavy:
            candidate_data_payloads.append({**base_data, "strategy": "fast"})
        # then fallback 1 let unstructured pick the strategy
        candidate_data_payloads.append({**base_data})
        # then fallback 2 conservative chunking
        candidate_data_payloads.append({**base_data, "infer_table_structure": "false"})

        last_exc = None
        for attempt, data in enumerate(candidate_data_payloads, start=1):
            for retry in range(self.max_retries):
                try:
                    file_bytes.seek(0)
                    logger.info(
                        "calling Unstructured API - attempt %d.%d for %s with payload keys=%s",
                        attempt,
                        retry + 1,
                        file_name,
                        list(data.keys()),
                    )
                    resp = requests.post(url, files=files, data=data, timeout=self.request_timeout)
                    status = resp.status_code
                    text = resp.text or ""
                    try:
                        json_body = resp.json()
                    except Exception:
                        json_body = None

                    if status == 200:
                        try:
                            elements = resp.json()
                        except Exception as parse_exc:
                            logger.exception("Failed parsing Unstructured JSON - %s", parse_exc)
                            try:
                                elements = json.loads(resp.text)
                            except Exception as fallback_exc:
                                raise ValueError("Failed to parse Unstructured JSON response") from fallback_exc

                        if not isinstance(elements, list):
                            logger.warning("Unstructured responded with unexpected payload type - %s", type(elements))
                            raise ValueError("Unexpected payload from Unstructured")
                        return elements

                    detail_msg = ""
                    if isinstance(json_body, dict):
                        detail_msg = json_body.get("detail", "") or json_body.get("error", "")
                    if "fast strategy" in text.lower() or "fast strategy" in str(detail_msg).lower():
                        logger.warning(
                            "Unstructured server reported fast strategy unavailable so trying fallback payloads"
                        )
                        last_exc = ValueError(f"Unstructured error - {resp.status_code} {resp.text}")
                        break

                    if 400 <= status < 500:
                        logger.error("Unstructured returned client error %d - %s", status, resp.text)
                        last_exc = ValueError(f"Client error {status} - {resp.text}")
                        break

                    if status >= 500:
                        logger.warning(
                            "Server error %d from Unstructured, will retry - response was: %s", status, resp.text[:200]
                        )
                        last_exc = RequestException(f"Server error {status}")
                        _time.sleep((2**retry) * 0.5)
                        continue

                    last_exc = ValueError(f"Unexpected status {status} - {resp.text}")
                    break

                except RequestException as re:
                    logger.warning("RequestException communicating with Unstructured - %s", re)
                    last_exc = re
                    _time.sleep((2**retry) * 0.5)
                    continue
            else:
                logger.debug("Exhausted retries for payload moving to next approach")
                continue
            continue

        # if we're at this point everything failed
        logger.exception("All Unstructured requests failed for file %s. Last exception: %s", file_name, last_exc)
        raise last_exc or RuntimeError("Unstructured requests failed without a recorded exception")

    def lazy_load(self, file_name: str, file_bytes: BytesIO) -> Iterator[Document]:
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        elements = self._get_chunks(file_name, file_bytes)

        for i, raw_chunk in enumerate(elements):
            raw_metadata = raw_chunk.get("metadata") or {}
            page_number = raw_metadata.get("page_number") or 1
            token_count = tokeniser(raw_chunk.get("text", ""))
            uploaded_meta = UploadedFileMetadata(
                index=i,
                uri=file_name,
                page_number=page_number,
                created_datetime=datetime.now(UTC),
                token_count=token_count,
                chunk_resolution=self.chunk_resolution,
                name=self.metadata.name,
                description=self.metadata.description,
                keywords=self.metadata.keywords,
            ).model_dump()

            yield Document(page_content=raw_chunk.get("text", ""), metadata=uploaded_meta)


class MetadataLoader:
    def __init__(self, env: Settings, s3_client: S3Client, file_name: str):
        self.env = env
        self.s3_client = s3_client
        self.llm = get_chat_llm(env.metadata_extraction_llm)
        self.file_name = file_name

    def _get_file_bytes(self, s3_client: S3Client, file_name: str) -> BytesIO:
        return BytesIO(s3_client.get_object(Bucket=self.env.bucket_name, Key=file_name)["Body"].read())

    def extract_metadata(self) -> GeneratedMetadata:
        """
        Extract metadata from first 1_000 chunks using UnstructuredChunkLoader
        """
        start_time = time.time()

        chunk_loader = UnstructuredChunkLoader(
            chunk_resolution=ChunkResolution.normal,
            env=self.env,
            min_chunk_size=self.env.worker_ingest_min_chunk_size,
            max_chunk_size=self.env.worker_ingest_max_chunk_size,
            overlap_chars=0,
            metadata=None,
        )

        file_bytes = self._get_file_bytes(s3_client=self.s3_client, file_name=self.file_name)

        try:
            chunks = chunk_loader._get_chunks(file_name=self.file_name, file_bytes=file_bytes)
        except Exception as e:
            logger.info("Failed metadata extraction - %s", e)
            chunks = []

        original_metadata = chunks[0]["metadata"] if chunks else {}
        first_thousand_words = "".join(chunk.get("text", "") for chunk in chunks)[:10_000]
        try:
            metadata = self.create_file_metadata(first_thousand_words, original_metadata=original_metadata)
        except Exception as e:
            logger.info(e)
            if original_metadata and original_metadata.get("filename"):
                metadata = GeneratedMetadata(name=original_metadata.get("filename"))
            else:
                metadata = GeneratedMetadata(name=self.file_name)

        duration = time.time() - start_time
        logger.info("total metadata extraction time for file [%s] took %.2f seconds", self.file_name, duration)

        return metadata

    def create_file_metadata(self, page_content: str, original_metadata: dict | None = None) -> GeneratedMetadata:
        """Uses a sample of the document and any extracted metadata to generate further metadata."""
        if not original_metadata:
            original_metadata = {}

        def trim(obj, max_length=1000):
            """original_metadata can be very long as it includes the original text"""
            if isinstance(obj, dict):
                return {k: trim(v, max_length) for k, v in obj.items()}
            if isinstance(obj, list):
                return [trim(v, max_length) for v in obj]
            if isinstance(obj, str):
                return obj[:max_length]
            return obj

        original_metadata = trim(original_metadata)

        parser = ClaudeParser(pydantic_object=GeneratedMetadata)

        metadata_prompt = PromptTemplate(
            template="".join(self.env.metadata_prompt)
            + "\n\n{format_instructions}\n\n{page_content}\n\n{original_metadata}",
            input_variables=["page_content"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "original_metadata": original_metadata,
            },
        )
        metadata_chain = metadata_prompt | self.llm | parser

        try:
            return metadata_chain.invoke({"page_content": page_content})
        except ValidationError as e:
            # error due to LLM return incorrect response
            logger.info(e.errors())
            return GeneratedMetadata(name=original_metadata.get("filename") or self.file_name)
