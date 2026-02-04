import logging
import time
from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING, List, Tuple

import environ
import fitz

from langchain_core.documents import Document
from pydantic import ValidationError
from redbox_app.setting_enums import Environment

from redbox.chains.components import get_chat_llm
from redbox.models.chain import GeneratedMetadata
from redbox.models.file import TabularSchema
from redbox.models.settings import Settings
from redbox.transform import bedrock_tokeniser
import pandas as pd
import math
import boto3
from typing import Iterator

from docx import Document as DocxDocument

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


def infer_sqlite_type(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    return "TEXT"


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
                table_name = name.lower().replace(" ", "_")
                sheet_schema = TabularSchema(
                    name=table_name, columns={col: infer_sqlite_type(df[col].dtype) for col in df.columns}
                )

                csv_text = f"<table_name>{table_name}</table_name>" + str(df.to_csv(index=False))
                elements.append({"text": csv_text, "metadata": {"document_schema": sheet_schema.model_dump()}})
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


class TextractChunkLoader:
    """
    Load, partition and chunk a document using:
    - Textract for PDFs
    - python-docx for DOCX.
    """

    def __init__(
        self,
        bucket: str,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap_chars: int = 200,
        region: str = "eu-west-2",
    ):
        self.bucket = bucket
        self.textract = boto3.client("textract", region_name=region)
        self.s3 = boto3.client("s3", region_name=region)

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_chars = overlap_chars

    def _extract_docx(self, file_bytes: BytesIO) -> List[str]:
        doc = DocxDocument(file_bytes)
        text = "\n".join(p.text for p in doc.paragraphs)
        return [text]

    def _upload_to_s3(self, file_name: str, file_bytes: BytesIO):
        file_bytes.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=f"ingest/{file_name}", Body=file_bytes)

    def _start_textract_job(self, file_name: str) -> str:
        response = self.textract.start_document_text_detection(
            DocumentLocation={
                "S3Object": {
                    "Bucket": self.bucket,
                    "Name": f"ingest/{file_name}",
                }
            }
        )
        return response["JobId"]

    def _wait_for_job(self, job_id: str):
        while True:
            response = self.textract.get_document_text_detection(JobId=job_id)
            status = response["JobStatus"]

            if status in ["SUCCEEDED", "FAILED"]:
                return status

            time.sleep(3)

    def _get_textract_results(self, job_id: str) -> List[str]:
        pages: dict[int, List[str]] = {}
        next_token = None

        while True:
            kwargs = {"JobId": job_id}
            if next_token:
                kwargs["NextToken"] = next_token

            response = self.textract.get_document_text_detection(**kwargs)

            for block in response.get("Blocks", []):
                if block["BlockType"] == "LINE":
                    page = block.get("Page", 1)
                    pages.setdefault(page, []).append(block["Text"])

            next_token = response.get("NextToken")
            if not next_token:
                break

        return ["\n".join(pages[p]) for p in sorted(pages)]

    def _extract_pdf(self, file_name: str, file_bytes: BytesIO) -> List[str]:
        self._upload_to_s3(file_name, file_bytes)

        job_id = self._start_textract_job(file_name)

        status = self._wait_for_job(job_id)

        if status != "SUCCEEDED":
            raise RuntimeError(f"Textract failed for {file_name}")

        return self._get_textract_results(job_id)

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.max_chunk_size, length)
            chunk = text[start:end]

            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)

            start = end - self.overlap_chars

        return chunks

    def _extract_pdf_from_s3(self, bucket: str, key: str) -> list[str]:
        response = self.textract.start_document_text_detection(
            DocumentLocation={
                "S3Object": {
                    "Bucket": bucket,
                    "Name": key,
                }
            }
        )

        job_id = response["JobId"]

        self._wait_for_job(job_id)

        return self._get_textract_results(job_id)

    def lazy_load(
        self,
        file_name: str,
        s3_bucket: str,
        s3_key: str,
        file_bytes: BytesIO | None = None,
    ) -> Iterator[Document]:
        if file_name.lower().endswith(".docx"):
            if file_bytes is None:
                raise ValueError("DOCX requires file_bytes")
            pages = self._extract_docx(file_bytes)
        else:
            pages = self._extract_pdf_from_s3(bucket=s3_bucket, key=s3_key)

        index = 0

        for page_num, page_text in enumerate(pages, start=1):
            chunks = self._chunk_text(page_text)

            for chunk in chunks:
                metadata = {
                    "index": index,
                    "uri": file_name,
                    "page_number": page_num,
                    "created_datetime": datetime.now(UTC),
                    "token_count": len(chunk.split()),
                }

                yield Document(page_content=chunk, metadata=metadata)

                index += 1


class MetadataLoader:
    def __init__(self, env: Settings, s3_client: S3Client, file_name: str):
        self.env = env
        self.s3_client = s3_client
        self.llm = get_chat_llm(env.metadata_extraction_llm)
        self.file_name = file_name

    def _get_file_bytes(self, file_name: str) -> BytesIO:
        return BytesIO(self.s3_client.get_object(Bucket=self.env.bucket_name, Key=file_name)["Body"].read())

    def extract_metadata(self) -> GeneratedMetadata:
        start_time = time.time()

        loader = TextractChunkLoader(
            bucket=self.env.bucket_name,
            min_chunk_size=200,
            max_chunk_size=2000,
            overlap_chars=0,
        )

        file_bytes = None
        if self.file_name.lower().endswith(".docx"):
            file_bytes = self._get_file_bytes(self.file_name)

        try:
            chunks = list(
                loader.lazy_load(
                    file_name=self.file_name,
                    s3_bucket=self.env.bucket_name,
                    s3_key=self.file_name,
                    file_bytes=file_bytes,
                )
            )
        except Exception as e:
            logger.info("Failed metadata extraction - %s", e)
            chunks = []

        text_sample = "".join(c.page_content for c in chunks)[:10_000]

        try:
            metadata = self.create_file_metadata(text_sample)
        except Exception as e:
            logger.info(e)
            metadata = GeneratedMetadata(name=self.file_name)

        logger.info(
            "total metadata extraction time for file [%s] took %.2f seconds",
            self.file_name,
            time.time() - start_time,
        )

        return metadata
