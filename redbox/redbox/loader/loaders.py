import logging
import os

from io import BytesIO
from typing import Iterator

import boto3

from unstructured.partition.auto import partition
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx

from redbox.loader.services.textract_service import (
    TextractService,
)
from redbox.loader.services.tabular import load_tabular_file

import time

from redbox.chains.components import get_chat_llm
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from redbox.models.chain import GeneratedMetadata
from redbox.chains.parser import ClaudeParser


logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(
        self,
        bucket: str,
        textract_service: TextractService,
        region: str = "eu-west-2",
    ):

        self.bucket = bucket
        self.textract_service = textract_service

        self.s3 = boto3.client(
            "s3",
            region_name=region,
        )

    def _extract_docx(
        self,
        file_bytes: BytesIO,
    ) -> Iterator[tuple[int, str]]:

        elements = partition_docx(file=file_bytes)

        current = []
        current_page = 1

        for el in elements:
            page = getattr(
                el.metadata,
                "page_number",
                1,
            )

            if page != current_page:
                yield current_page, "\n".join(current)

                current = []
                current_page = page

            current.append(str(el).strip())

        if current:
            yield current_page, "\n".join(current)

    def _extract_pptx(
        self,
        file_bytes: BytesIO,
    ) -> Iterator[tuple[int, str]]:

        elements = partition_pptx(file=file_bytes)

        current = []
        current_page = 1

        for el in elements:
            page = getattr(
                el.metadata,
                "page_number",
                1,
            )

            if page != current_page:
                yield current_page, "\n".join(current)

                current = []
                current_page = page

            current.append(str(el).strip())

        if current:
            yield current_page, "\n".join(current)

    def _extract_unstructured(
        self,
        file_bytes: BytesIO,
    ) -> Iterator[tuple[int, str]]:

        elements = partition(file=file_bytes)

        current = []
        current_page = 1

        for el in elements:
            page = getattr(
                el.metadata,
                "page_number",
                1,
            )

            if page != current_page:
                yield current_page, "\n".join(current)

                current = []
                current_page = page

            current.append(str(el).strip())

        if current:
            yield current_page, "\n".join(current)

    def iter_pages(self, file_name: str, file_bytes: BytesIO | None = None) -> Iterator[tuple[int, str]]:
        display_name = os.path.basename(file_name).lower()

        if file_bytes is None:
            obj = self.s3.get_object(Bucket=self.bucket, Key=file_name)
            file_bytes = BytesIO(obj["Body"].read())

        if display_name.endswith((".csv", ".tsv", ".xls", ".xlsx")):
            elements = load_tabular_file(display_name, file_bytes)
            for idx, el in enumerate(elements or []):
                yield 1, el["text"]
            return

        if display_name.endswith(".pdf"):
            output_prefix = f"textract-output/{file_name}/"
            yield from self.textract_service.iter_output_pages(
                output_bucket=self.bucket,
                output_prefix=output_prefix,
            )
            return

        if display_name.endswith(".docx"):
            yield from self._extract_docx(file_bytes)
            return

        if display_name.endswith((".ppt", ".pptx")):
            yield from self._extract_pptx(file_bytes)
            return

        yield from self._extract_unstructured(file_bytes)


class MetadataLoader:
    def __init__(self, env, s3_client, file_name: str, document_loader: DocumentLoader):
        self.env = env
        self.s3_client = s3_client
        self.file_name = file_name
        self.llm = get_chat_llm(env.metadata_extraction_llm)

        self.document_loader = document_loader

    def _get_file_bytes(self, file_name: str) -> BytesIO:
        obj = self.s3_client.get_object(Bucket=self.env.bucket_name, Key=file_name)
        return BytesIO(obj["Body"].read())

    def extract_metadata(self) -> GeneratedMetadata:
        start_time = time.time()

        display_name = os.path.basename(self.file_name).lower()

        if display_name.endswith((".csv", ".tsv", ".xls", ".xlsx")):
            file_bytes = self._get_file_bytes(self.file_name)
            elements = load_tabular_file(display_name, file_bytes)
            sample_text = elements[0]["text"] if elements else ""

        else:
            pages_text = []
            char_count = 0
            MAX_CHARS = 10_000

            for page_num, page_text in self.document_loader.iter_pages(self.file_name):
                pages_text.append(page_text)
                char_count += len(page_text)

                if char_count >= MAX_CHARS or len(pages_text) >= 8:
                    break

            sample_text = "\n".join(pages_text)[:MAX_CHARS]

        file_type = self._infer_file_type()

        original_metadata = {
            "file_type": file_type,
            "filename": self.file_name,
        }

        try:
            metadata = self.create_file_metadata(sample_text, original_metadata)
        except Exception as e:
            logger.exception("Metadata extraction failed: %s", e)
            metadata = GeneratedMetadata(name=self.file_name)

        logger.info(
            "Metadata extraction for [%s] took %.2fs",
            self.file_name,
            time.time() - start_time,
        )

        return metadata

    def _infer_file_type(self):
        name = self.file_name.lower()

        if name.endswith(".docx"):
            return "DOCX"
        if name.endswith(".csv"):
            return "CSV"
        if name.endswith((".xlsx", ".xls")):
            return "Excel"
        if name.endswith(".pdf"):
            return "PDF"
        if name.endswith((".ppt", ".pptx")):
            return "PPT"
        return "unknown"

    def create_file_metadata(self, page_content: str, original_metadata: dict):
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

        chain = metadata_prompt | self.llm | parser

        try:
            metadata = chain.invoke({"page_content": page_content})

            if not metadata.name:
                metadata.name = original_metadata.get("filename", self.file_name)

            return metadata

        except ValidationError as e:
            logger.warning("Metadata validation failed: %s", e)
            return GeneratedMetadata(name=self.file_name)
