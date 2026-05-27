import environ
import logging
import os
import time
from datetime import UTC, datetime
from io import BytesIO
from typing import Iterator

import boto3
import pandas as pd
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError
from unstructured.partition.auto import partition
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx

from redbox.chains.components import get_chat_llm
from redbox.chains.parser import ClaudeParser
from redbox.models.chain import GeneratedMetadata
from redbox.models.file import (
    ChunkResolution,
    TabularSchema,
    UploadedFileMetadata,
)
from redbox.models.settings import Settings
from redbox.transform import bedrock_tokeniser
from redbox.loader.loaders import load_tabular_file

env = environ.Env()

logger = logging.getLogger(__name__)


tokeniser = bedrock_tokeniser


def infer_sqlite_type(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"

    if pd.api.types.is_float_dtype(dtype):
        return "REAL"

    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"

    return "TEXT"


def parse_tabular_schema(
    table_name: str,
    df: pd.DataFrame,
):
    csv_text = f"<table_name>{table_name}</table_name>\n" + df.to_csv(index=False)

    schema = TabularSchema(
        name=table_name,
        columns={col: infer_sqlite_type(df[col].dtype) for col in df.columns},
    )

    return csv_text, schema.model_dump()


class TextractChunkLoader:
    def __init__(
        self,
        bucket: str,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap_chars: int = 200,
        region: str = "eu-west-2",
        metadata: GeneratedMetadata | None = None,
        include_schema_metadata: bool = False,
    ):
        self.bucket = bucket

        self.textract = boto3.client(
            "textract",
            region_name=region,
        )

        self.s3 = boto3.client(
            "s3",
            region_name=region,
        )

        self.metadata = metadata or GeneratedMetadata(
            name="",
            description="",
            keywords=[],
        )

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_chars = overlap_chars
        self.include_schema_metadata = include_schema_metadata

        logger.info(
            "Initialised TextractChunkLoader (bucket=%s, region=%s, min_chunk=%s, max_chunk=%s, overlap=%s)",
            bucket,
            region,
            min_chunk_size,
            max_chunk_size,
            overlap_chars,
        )

    def _wait_for_job(self, job_id: str):
        logger.info("Waiting for Textract job %s to complete", job_id)

        while True:
            response = self.textract.get_document_text_detection(JobId=job_id)

            status = response["JobStatus"]

            logger.info(
                "Textract job %s status=%s",
                job_id,
                status,
            )

            if status in [
                "SUCCEEDED",
                "FAILED",
            ]:
                return status

            time.sleep(3)

    def _stream_textract_pages(
        self,
        job_id: str,
    ) -> Iterator[str]:
        next_token = None

        current_page = None
        current_lines = []

        while True:
            kwargs = {"JobId": job_id}

            if next_token:
                kwargs["NextToken"] = next_token

            response = self.textract.get_document_text_detection(**kwargs)

            for block in response.get("Blocks", []):
                if block["BlockType"] != "LINE":
                    continue

                page = block.get("Page", 1)

                if current_page is None:
                    current_page = page

                if page != current_page:
                    yield "\n".join(current_lines)

                    current_lines = []
                    current_page = page

                current_lines.append(block["Text"])

            next_token = response.get("NextToken")

            if not next_token:
                break

        if current_lines:
            yield "\n".join(current_lines)

    def _extract_pdf_from_s3(
        self,
        bucket: str,
        key: str,
    ) -> Iterator[str]:
        response = self.textract.start_document_text_detection(
            DocumentLocation={
                "S3Object": {
                    "Bucket": bucket,
                    "Name": key,
                }
            }
        )

        job_id = response["JobId"]

        status = self._wait_for_job(job_id)

        if status != "SUCCEEDED":
            raise RuntimeError(f"Textract failed for s3://{bucket}/{key}")

        yield from self._stream_textract_pages(job_id)

    def _extract_docx(
        self,
        file_bytes: BytesIO,
    ) -> Iterator[str]:
        file_bytes.seek(0)

        elements = partition_docx(file=file_bytes)

        current_page = []
        last_page = None

        for el in elements:
            page_number = getattr(el.metadata, "page_number", None)

            text = str(el).strip()

            if not text:
                continue

            if page_number is not None:
                if last_page is None:
                    last_page = page_number

                if page_number != last_page:
                    yield "\n".join(current_page)

                    current_page = []
                    last_page = page_number

            current_page.append(text)

        if current_page:
            yield "\n".join(current_page)

    def _extract_pptx(
        self,
        file_bytes: BytesIO,
    ) -> Iterator[str]:
        file_bytes.seek(0)

        elements = partition_pptx(file=file_bytes)

        slides = {}

        for el in elements:
            slide_number = getattr(
                el.metadata,
                "slide_number",
                1,
            )

            text = str(el).strip()

            if text:
                slides.setdefault(slide_number, []).append(text)

        for slide_num in sorted(slides):
            yield "\n".join(slides[slide_num])

    def _extract_with_unstructured(
        self,
        file_bytes: BytesIO,
    ) -> Iterator[str]:
        file_bytes.seek(0)

        elements = partition(file=file_bytes)

        current_page = []
        last_page = None

        for el in elements:
            page_number = getattr(el.metadata, "page_number", None) or getattr(el.metadata, "slide_number", None)

            text = str(el).strip()

            if not text:
                continue

            if page_number is not None:
                if last_page is None:
                    last_page = page_number

                if page_number != last_page:
                    yield "\n".join(current_page)

                    current_page = []
                    last_page = page_number

            current_page.append(text)

        if current_page:
            yield "\n".join(current_page)

    def _extract_tabular(
        self,
        file_name: str,
        file_bytes: BytesIO,
    ) -> Iterator[Document]:

        logger.info("lazy_load called for %s", file_name)

        s3_key = file_name
        display_name = os.path.basename(file_name).lower()
        logger.info("File type detected: %s", display_name)

        if file_bytes is None:
            obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            file_bytes = BytesIO(obj["Body"].read())

        if display_name.endswith((".csv", ".tsv", ".xls", ".xlsx")):
            tabular_elements = load_tabular_file(display_name, file_bytes)
            for idx, el in enumerate(tabular_elements or []):
                metadata = UploadedFileMetadata(
                    index=idx,
                    uri=s3_key,
                    page_number=1,
                    created_datetime=datetime.now(UTC),
                    token_count=tokeniser(el["text"]),
                    chunk_resolution=ChunkResolution.tabular,
                    name=self.metadata.name,
                    description=self.metadata.description,
                    keywords=self.metadata.keywords,
                ).model_dump()

                merged_metadata = metadata
                if self.include_schema_metadata:
                    merged_metadata = {**metadata, **el.get("metadata", {})}

                yield Document(page_content=el["text"], metadata=merged_metadata)
            return

        if display_name.endswith(".pdf"):
            yield from self._extract_pdf_from_s3(
                bucket=self.bucket,
                key=file_name,
            )

            return

        if file_bytes is None:
            obj = self.s3.get_object(
                Bucket=self.bucket,
                Key=file_name,
            )

            file_bytes = BytesIO(obj["Body"].read())

        if display_name.endswith(".docx"):
            yield from self._extract_docx(file_bytes)
            return

        if display_name.endswith((".ppt", ".pptx")):
            yield from self._extract_pptx(file_bytes)
            return

        yield from self._extract_with_unstructured(file_bytes)

    def _chunk_text(
        self,
        text: str,
    ) -> Iterator[str]:
        if not text:
            return

        start = 0
        length = len(text)

        while start < length:
            end = min(
                start + self.max_chunk_size,
                length,
            )

            chunk = text[start:end]

            if len(chunk) >= self.min_chunk_size or start == 0:
                yield chunk

            if end >= length:
                break

            start = max(
                0,
                end - self.overlap_chars,
            )

    def pages_to_documents(
        self,
        pages: Iterator[str],
        s3_key: str,
        chunk_resolution: ChunkResolution,
    ) -> Iterator[Document]:
        idx = 0

        for page_num, page_text in enumerate(
            pages,
            start=1,
        ):
            for chunk in self._chunk_text(page_text):
                metadata = UploadedFileMetadata(
                    index=idx,
                    uri=s3_key,
                    page_number=page_num,
                    created_datetime=datetime.now(UTC),
                    token_count=tokeniser(chunk),
                    chunk_resolution=chunk_resolution,
                    name=self.metadata.name,
                    description=self.metadata.description,
                    keywords=self.metadata.keywords,
                ).model_dump()

                yield Document(
                    page_content=chunk,
                    metadata=metadata,
                )

                idx += 1

    def lazy_load(
        self,
        file_name: str,
        file_bytes: BytesIO | None = None,
        chunk_resolution: ChunkResolution = ChunkResolution.normal,
    ) -> Iterator[Document]:
        display_name = os.path.basename(file_name).lower()

        if file_bytes is None:
            obj = self.s3.get_object(
                Bucket=self.bucket,
                Key=file_name,
            )

            file_bytes = BytesIO(obj["Body"].read())

        if display_name.endswith(
            (
                ".csv",
                ".tsv",
                ".xls",
                ".xlsx",
            )
        ):
            yield from self._extract_tabular(
                file_name=file_name,
                file_bytes=file_bytes,
            )

            return

        pages = self.extract_pages(
            file_name=file_name,
            file_bytes=file_bytes,
        )

        yield from self.pages_to_documents(
            pages=pages,
            s3_key=file_name,
            chunk_resolution=chunk_resolution,
        )


class MetadataLoader:
    def __init__(
        self,
        env: Settings,
        s3_client,
        file_name: str,
    ):
        self.env = env
        self.s3_client = s3_client
        self.file_name = file_name

        self.llm = get_chat_llm(env.metadata_extraction_llm)

    def extract_metadata(self) -> GeneratedMetadata:
        loader = TextractChunkLoader(
            bucket=self.env.bucket_name,
            min_chunk_size=200,
            max_chunk_size=2000,
            overlap_chars=0,
        )

        docs_iter = loader.lazy_load(file_name=self.file_name)

        collected = []
        current_size = 0

        for doc in docs_iter:
            remaining = 10_000 - current_size

            if remaining <= 0:
                break

            text = doc.page_content[:remaining]

            collected.append(text)

            current_size += len(text)

        first_10k_chars = "".join(collected)

        parser = ClaudeParser(pydantic_object=GeneratedMetadata)

        metadata_prompt = PromptTemplate(
            template="".join(self.env.metadata_prompt) + "\n\n{format_instructions}\n\n{page_content}",
            input_variables=["page_content"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
            },
        )

        metadata_chain = metadata_prompt | self.llm | parser

        try:
            metadata = metadata_chain.invoke(
                {
                    "page_content": first_10k_chars,
                }
            )

            if not metadata.name:
                metadata.name = self.file_name

            return metadata

        except ValidationError:
            return GeneratedMetadata(name=self.file_name)
