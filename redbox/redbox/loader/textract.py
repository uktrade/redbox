import logging
import boto3
import time
import random
import fitz
import os

from io import BytesIO
from datetime import UTC, datetime
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import List, Iterator
from langchain_core.documents import Document
from unstructured.partition.docx import partition_docx
from unstructured.partition.auto import partition
from unstructured.partition.pptx import partition_pptx

from redbox.models.chain import GeneratedMetadata
from redbox.models.file import UploadedFileMetadata, ChunkResolution
from redbox.transform import bedrock_tokeniser
from redbox.loader.loaders import load_tabular_file
from redbox.loader.chunker import DocumentChunker, LayoutBlock
from redbox.loader.parsers.markdown import _MarkdownLayoutParser


logger = logging.getLogger(__name__)

tokeniser = bedrock_tokeniser

_LAYOUT_TITLE_TYPES = {"LAYOUT_TITLE", "LAYOUT_SECTION_HEADER"}

_LAYOUT_SKIP_TYPES = {"LAYOUT_HEADER", "LAYOUT_FOOTER", "LAYOUT_PAGE_NUMBER", "LAYOUT_FIGURE"}


class TextractChunkLoader:
    """
    Load, partition and chunk a document using:
    - Textract for PDFs
    - python-docx for DOCX
    - html.parser for HTML
    - regex-based parser for Markdown
    - Pandas for CSV/Excel
    """

    def __init__(
        self,
        bucket: str,
        chunk_resolution: ChunkResolution = ChunkResolution.normal,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap_chars: int = 200,
        region: str = "eu-west-2",
        metadata: GeneratedMetadata | None = None,
        include_schema_metadata: bool = False,
    ):
        self.bucket = bucket
        self.chunk_resolution = chunk_resolution
        textract_config = Config(
            retries={"mode": "adaptive", "max_attempts": 10},
            connect_timeout=20,
            read_timeout=70,
        )
        self.textract = boto3.client("textract", region_name=region, config=textract_config)
        self.s3 = boto3.client("s3", region_name=region)
        self.metadata = metadata or GeneratedMetadata(name="", description="", keywords=[])
        self.include_schema_metadata = include_schema_metadata
        self.chunker = DocumentChunker(
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap_chars=overlap_chars,
        )

        logger.info(
            "Initialised TextractChunkLoader (bucket=%s, chunk_resolution=%s, region=%s, min_chunk=%s, max_chunk=%s, overlap=%s)",
            bucket,
            chunk_resolution,
            region,
            min_chunk_size,
            max_chunk_size,
            overlap_chars,
        )

    def _is_retryable_textract_error(self, error: Exception) -> bool:
        if not isinstance(error, ClientError):
            return False
        error_code = error.response.get("Error", {}).get("Code", "")
        return error_code in {
            "ProvisionedThroughputExceededException",
            "ThrottlingException",
            "Throttling",
            "RequestLimitExceeded",
        }

    def _retry_textract_request(self, func, *args, max_attempts: int = 6, base_delay: float = 3.0, **kwargs):
        attempt = 0
        while True:
            attempt += 1
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if self._is_retryable_textract_error(e) and attempt < max_attempts:
                    sleep_time = base_delay * (2 ** (attempt - 1)) + random.random()
                    logger.warning(
                        "Textract throttled on attempt %s/%s for %s; sleeping %.1fs before retrying",
                        attempt,
                        max_attempts,
                        getattr(func, "__name__", str(func)),
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue
                logger.exception("Textract API error on %s: %s", getattr(func, "__name__", str(func)), e)
                raise

    def _get_textract_layout_results(self, job_id: str) -> list[LayoutBlock]:
        logger.info("Fetching Textract LAYOUT results for job %s", job_id)

        blocks: list[LayoutBlock] = []
        next_token = None
        api_calls = 0

        while True:
            kwargs = {"JobId": job_id}
            if next_token:
                kwargs["NextToken"] = next_token

            response = self._retry_textract_request(self.textract.get_document_analysis, **kwargs)
            api_calls += 1

            for block in response.get("Blocks", []):
                block_type = block.get("BlockType", "")
                text = block.get("Text", "").strip()
                page = block.get("Page", 1)

                if not text or block_type in _LAYOUT_SKIP_TYPES:
                    continue
                if not block_type.startswith("LAYOUT_"):
                    continue

                blocks.append(
                    LayoutBlock(
                        text=text,
                        block_type=block_type,
                        page_number=page,
                        is_title=block_type in _LAYOUT_TITLE_TYPES,
                    )
                )

            next_token = response.get("NextToken")
            if not next_token:
                break

        logger.info(
            "Retrieved %d layout blocks for job %s via %d API calls",
            len(blocks),
            job_id,
            api_calls,
        )
        return blocks

    def _extract_pdf_layout_from_s3(self, bucket: str, key: str) -> list[LayoutBlock]:
        logger.info("Starting Textract LAYOUT analysis for s3://%s/%s", bucket, key)

        try:
            response = self._retry_textract_request(
                self.textract.start_document_analysis,
                DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
                FeatureTypes=["LAYOUT"],
            )

            job_id = response["JobId"]
            logger.info("Started Textract LAYOUT job %s", job_id)
            status = self._wait_for_layout_job(job_id)

            if status != "SUCCEEDED":
                raise RuntimeError(f"Textract LAYOUT job failed for s3://{bucket}/{key}")

            return self._get_textract_layout_results(job_id)

        except Exception as e:
            logger.exception("Textract LAYOUT extraction failed for s3://%s/%s: %s", bucket, key, e)
            raise

    def _wait_for_layout_job(self, job_id: str) -> str:
        logger.info("Waiting for Textract LAYOUT job %s", job_id)

        while True:
            response = self._retry_textract_request(self.textract.get_document_analysis, JobId=job_id)
            status = response["JobStatus"]
            logger.debug("Textract LAYOUT job %s status: %s", job_id, status)

            if status in ("SUCCEEDED", "FAILED"):
                logger.info("Textract LAYOUT job %s finished: %s", job_id, status)
                return status

            time.sleep(5)

    def _extract_pdf_text_direct(self, file_bytes: BytesIO) -> List[str]:
        logger.info("Extracting PDF text directly with PyMuPDF")
        file_bytes.seek(0)
        doc = fitz.open(stream=file_bytes.getvalue(), filetype="pdf")
        pages: List[str] = []

        for page in doc:
            text = page.get_text().strip()
            if text:
                pages.append(text)

        if not pages:
            raise ValueError("PDF contains no extractable text")

        logger.info("Extracted %d page(s) directly from PDF", len(pages))
        return pages

    def _extract_markdown(
        self,
        file_bytes: BytesIO,
    ) -> list[LayoutBlock]:
        """
        Parses Markdown into LayoutBlocks.

        Headings become title blocks.
        Paragraphs become text blocks.
        """

        logger.info("Extracting Markdown")

        file_bytes.seek(0)

        raw = file_bytes.read().decode(
            "utf-8",
            errors="replace",
        )

        parser = _MarkdownLayoutParser()

        blocks = parser.parse(raw)

        if not blocks:
            raise ValueError("Markdown document contains no extractable text")

        logger.info(
            "Extracted %d layout blocks from Markdown",
            len(blocks),
        )

        return blocks

    def _extract_docx(self, file_bytes: BytesIO) -> List[str]:
        logger.info("Extracting DOCX with unstructured")
        file_bytes.seek(0)

        try:
            elements = partition_docx(file=file_bytes)

            if not elements:
                raise ValueError("unstructured returned no elements from DOCX")

            text_pages = []
            current_page = []
            last_page = None

            for el in elements:
                page_number = getattr(el.metadata, "page_number", None)

                if page_number is not None:
                    if last_page is None:
                        last_page = page_number
                    if page_number != last_page:
                        if current_page:
                            text_pages.append("\n".join(current_page))
                        current_page = []
                        last_page = page_number

                current_page.append(str(el).strip())

            if current_page:
                text_pages.append("\n".join(current_page))

            if not text_pages:
                raise ValueError("unstructured extracted no readable text from DOCX")

            logger.info("Extracted %d page(s) from DOCX using unstructured", len(text_pages))
            return text_pages

        except Exception as e:
            logger.exception("unstructured failed to process DOCX: %s", str(e))
            raise

    def _extract_pptx(self, file_bytes: BytesIO) -> List[str]:
        logger.info("Extracting PPTX with unstructured.partition.pptx")
        file_bytes.seek(0)

        try:
            elements = partition_pptx(file=file_bytes)

            logger.info("partition_pptx returned %d elements", len(elements))
            if not elements:
                raise ValueError("unstructured.partition.pptx returned no elements")

            text_pages = []
            current_page = []
            last_page = None

            for el in elements:
                page_number = getattr(el.metadata, "page_number", None)

                if page_number is not None:
                    if last_page is None:
                        last_page = page_number
                    if page_number != last_page:
                        if current_page:
                            text_pages.append("\n".join(current_page))
                        current_page = []
                        last_page = page_number

                current_page.append(str(el).strip())

            if current_page:
                text_pages.append("\n".join(current_page))

            if not text_pages:
                text_pages = ["\n".join(str(el).strip() for el in elements)]

            logger.info("Extracted %d slide(s) from PPTX", len(text_pages))
            return text_pages

        except ImportError:
            logger.error("unstructured[pptx] extra not installed")
            raise
        except Exception as e:
            logger.exception("PPTX extraction failed: %s", e)
            raise

    def _extract_with_unstructured(self, file_bytes: BytesIO, file_name: str) -> List[str]:
        file_bytes.seek(0)

        elements = partition(file=file_bytes)

        if not elements:
            raise ValueError(f"unstructured returned no elements from {file_name}")

        text_pages: List[str] = []
        current_page: List[str] = []
        last_page = None

        for el in elements:
            page_number = getattr(el.metadata, "page_number", None) or getattr(el.metadata, "slide_number", None)

            if page_number is not None:
                if last_page is None or page_number != last_page:
                    if current_page:
                        text_pages.append("\n".join(current_page))
                    current_page = []
                    last_page = page_number

            current_page.append(str(el).strip())

        if current_page:
            text_pages.append("\n".join(current_page))

        if not text_pages:
            text_pages = ["\n".join(str(el).strip() for el in elements)]

        logger.info("Extracted %d page(s) from %s using unstructured", len(text_pages), file_name)
        return text_pages

    def _extract_pdf_layout(self, file_bytes: BytesIO, s3_key: str, display_name: str) -> list[LayoutBlock]:
        try:
            return self._extract_pdf_layout_from_s3(bucket=self.bucket, key=s3_key)
        except Exception as e:
            logger.warning("Textract LAYOUT failed for %s; falling back to direct extraction - %s", display_name, e)
            raw_pages = self._extract_pdf_text_direct(file_bytes)
            return [
                LayoutBlock(text=text, block_type="LAYOUT_TEXT", page_number=i + 1, is_title=False)
                for i, text in enumerate(raw_pages)
                if text.strip()
            ]

    def _extract_layout_blocks(
        self,
        display_name: str,
        file_bytes: BytesIO,
        s3_key: str,
    ) -> list[LayoutBlock]:

        if display_name.endswith(".pdf"):
            return self._extract_pdf_layout(file_bytes, s3_key, display_name)

        if display_name.endswith((".md", ".markdown")):
            return self._extract_markdown(file_bytes)

        if display_name.endswith(".docx"):
            # _extract_docx returns List[str]; wrap each page into a LayoutBlock.
            pages = self._extract_docx(file_bytes)
            return [
                LayoutBlock(text=text, block_type="LAYOUT_TEXT", page_number=i + 1, is_title=False)
                for i, text in enumerate(pages)
            ]

        if display_name.endswith((".pptx", ".ppt")):
            pages = self._extract_pptx(file_bytes)
            return [
                LayoutBlock(text=text, block_type="LAYOUT_TEXT", page_number=i + 1, is_title=False)
                for i, text in enumerate(pages)
            ]

        pages = self._extract_with_unstructured(file_bytes, display_name)
        return [
            LayoutBlock(text=text, block_type="LAYOUT_TEXT", page_number=i + 1, is_title=False)
            for i, text in enumerate(pages)
        ]

    def lazy_load(
        self,
        file_name: str,
        file_bytes: BytesIO | None = None,
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

        layout_blocks = self._extract_layout_blocks(
            display_name=display_name,
            file_bytes=file_bytes,
            s3_key=s3_key,
        )

        for idx, chunk in enumerate(self.chunker.chunk(layout_blocks)):
            metadata = UploadedFileMetadata(
                index=idx,
                uri=s3_key,
                page_number=chunk.page_start,
                created_datetime=datetime.now(UTC),
                token_count=tokeniser(chunk.text),
                chunk_resolution=self.chunk_resolution,
                name=self.metadata.name,
                description=self.metadata.description,
                keywords=self.metadata.keywords,
            ).model_dump()

            yield Document(
                page_content=chunk.text,
                metadata=metadata,
            )
