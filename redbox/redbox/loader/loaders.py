import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING

import environ
import fitz
import requests

# from requests.adapters import HTTPAdapter
# from requests.packages.urllib.util.retry import Retry
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError
from redbox_app.setting_enums import Environment

from redbox.chains.components import get_chat_llm
from redbox.chains.parser import ClaudeParser
from redbox.models.chain import GeneratedMetadata
from redbox.models.file import ChunkResolution, UploadedFileMetadata
from redbox.models.settings import Settings
from redbox.transform import bedrock_tokeniser
import pandas as pd

env = environ.Env()

ENVIRONMENT = Environment[env.str("ENVIRONMENT").upper()]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

tokeniser = bedrock_tokeniser

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object


def is_large_pdf(file_name: str, filebytes: BytesIO, page_threshold=150) -> tuple[bool, int]:
    if not file_name.lower().endswith(".pdf"):
        return False, 0
    doc = fitz.open(stream=filebytes.getvalue(), filetype="pdf")
    return len(doc) > page_threshold, len(doc)


def split_pdf(filebytes: BytesIO, pages_per_chunk: int = 75) -> list[BytesIO]:
    doc = fitz.open(stream=filebytes.getvalue(), filetype="pdf")
    chunks = []
    for start in range(0, len(doc), pages_per_chunk):
        sub_doc = fitz.open()
        for page_num in range(start, min(start + pages_per_chunk, len(doc))):
            sub_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        if len(sub_doc) == 0:
            continue  # Skip empty chunks
        chunk_bytes = BytesIO(sub_doc.tobytes())
        chunks.append(chunk_bytes)
    return chunks


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
        print(e)
        logger.info(f"Excel Read Error: {e}")
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
    ):
        self.chunk_resolution = chunk_resolution
        self.env = env
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._overlap_chars = overlap_chars
        self._overlap_all_chunks = overlap_all_chunks
        self.metadata = metadata or GeneratedMetadata(name="", description="", keywords=[])

    def _get_chunks(self, file_name: str, file_bytes: BytesIO) -> list[dict]:
        """Helper method to perform chunking via the unstructured API"""
        if ENVIRONMENT.is_local:
            url = f"http://{self.env.unstructured_host}:8000/general/v0/general"
        else:
            url = f"http://{self.env.unstructured_host}/general/v0/general"

        is_large, _ = is_large_pdf(file_name=file_name, filebytes=file_bytes)
        is_tabular = file_name.endswith((".csv", ".xls", ".xlsx"))
        file_bytes.seek(0)
        if is_large:
            elements = []
            pdf_chunks = split_pdf(filebytes=file_bytes)
            for idx, chunk in enumerate(pdf_chunks):
                chunk.seek(0)
                files = {
                    "files": (file_name, chunk),
                }
                response = self.post_files(url, files)

                if response.status_code != 200:
                    msg = f"Chunk {idx + 1} failed: {response.text}"
                    print(response)
                    raise ValueError(msg)
                elements.extend(response.json())
        elif is_tabular and self.chunk_resolution == ChunkResolution.tabular:
            # Carry out the special ingest process for tabular files - will be carried out in addition to
            elements = load_tabular_file(file_name=file_name, file_bytes=file_bytes)
        else:
            files = {
                "files": (file_name, file_bytes),
            }

            response = self.post_files(url, files)

            if response.status_code != 200:
                raise ValueError(response.text)

            elements = response.json()

            if not elements:
                raise ValueError("Unstructured failed to extract text for this file")
        logger.debug(f"{len(elements)}")
        return elements

    def post_files(self, url, files):
        response = requests.post(
            url,
            files=files,
            data={
                "strategy": "fast",
                "chunking_strategy": "by_title",
                "max_characters": self._max_chunk_size,
                "combine_under_n_chars": self._min_chunk_size,
                "overlap": self._overlap_chars,
                "overlap_all": self._overlap_all_chunks,
                "infer_table_structure": "true",
            },
        )
        return response

    def lazy_load(self, file_name: str, file_bytes: BytesIO) -> Iterator[Document]:
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        elements = self._get_chunks(file_name, file_bytes)

        for i, raw_chunk in enumerate(elements):
            yield Document(
                page_content=raw_chunk["text"],
                metadata=UploadedFileMetadata(
                    index=i,
                    uri=file_name,
                    page_number=1
                    if not raw_chunk["metadata"].get("page_number")
                    else raw_chunk["metadata"].get("page_number"),
                    created_datetime=datetime.now(UTC),
                    token_count=tokeniser(raw_chunk["text"]),
                    chunk_resolution=self.chunk_resolution,
                    name=self.metadata.name,
                    description=self.metadata.description,
                    keywords=self.metadata.keywords,
                ).model_dump(),
            )


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

        chunks = chunk_loader._get_chunks(file_name=self.file_name, file_bytes=file_bytes)

        original_metadata = chunks[0]["metadata"] if chunks else {}
        first_thousand_words = "".join(chunk["text"] for chunk in chunks)[:10_000]
        try:
            metadata = self.create_file_metadata(first_thousand_words, original_metadata=original_metadata)
        except Exception as e:
            logger.info(e)
            if original_metadata.get("filename"):
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
