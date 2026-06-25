from io import BytesIO
import os
from uuid import uuid4
import logging
from typing import List
import fitz
import boto3
import signal
from contextlib import contextmanager

from redbox_app.redbox_core.models import File
from redbox.models.file import ChunkResolution
from redbox.models.settings import get_settings
from unstructured.documents.elements import Element

from redbox.loader.loaders import load_tabular_file
from redbox.loader.extraction.textract import TextractService
from redbox.loader.extraction.unstructured import UnstructuredService

logger = logging.getLogger(__name__)

env = get_settings()


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutException()

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)


class DocumentExtractionService:
    """
    Orchestrates document extraction from S3-backed files using multiple
    extraction engines with fallback and timeout-aware execution.

    This service provides a unified interface for extracting structured or
    semi-structured content from a variety of file types including PDFs,
    DOCX, PPTX, and tabular formats. It dynamically selects the appropriate
    extraction strategy based on file type and configuration, and applies
    progressive fallback mechanisms to maximise extraction success.

    Extraction strategies include:
        - Unstructured-based parsing (auto/fast modes)
        - AWS Textract (document analysis)
        - PyMuPDF direct text extraction (PDF fallback)
        - Tabular file loader for structured datasets

    A timeout mechanism is used to prevent long-running extractions from
    blocking the pipeline, and each strategy is attempted in sequence until
    a successful extraction is achieved.

    Attributes:
        bucket (str):
            S3 bucket from which documents are retrieved.
        region (str):
            AWS region for S3 and Textract services.
        s3 (boto3.client):
            S3 client used to fetch raw file bytes.
        textract (TextractService):
            Wrapper service for AWS Textract extraction workflows.
        unstructured (UnstructuredService):
            Wrapper around `unstructured` parsing library.
        log_stub (str):
            Unique identifier prefix for structured logging.
        extract_calls (int):
            Counter tracking number of extraction calls made.

    Methods:
        _extract_pdf_text_direct(file_bytes, log_stub):
            Extracts text directly from PDFs using PyMuPDF when structured
            extraction is unnecessary or has failed.

        _run_with_fallbacks(file_bytes, s3_key, log_stub):
            Executes a sequence of extraction strategies with configurable
            timeouts and fallback behaviour:
                1. Unstructured (auto)
                2. Unstructured (fast)
                3. Textract (document analysis)
                4. PyMuPDF direct extraction (final fallback)

        extract(file_name, chunk_resolution):
            Main entry point for document extraction. Fetches file from S3,
            detects file type, selects appropriate extraction strategy, and
            returns extracted content along with the strategy used.
    """

    def __init__(
        self,
        bucket: str,
        region: str = "eu-west-2",
    ):
        self.bucket = bucket
        self.region = region

        self.s3 = boto3.client("s3", region_name=region)
        self.textract = TextractService(bucket=bucket, region=region)
        self.unstructured = UnstructuredService()
        self.log_stub = f"[DocumentExtractionService run_id='{str(uuid4())[:8]}']"
        self.extract_calls = 0

        logger.warning(
            "%s Initialised (bucket=%s, region=%s)",
            self.log_stub,
            bucket,
            region,
        )

    def _extract_pdf_text_direct(self, file_bytes: BytesIO, log_stub: str) -> List[str]:
        logger.warning("%s Extracting PDF text directly with PyMuPDF", log_stub)
        file_bytes.seek(0)
        doc = fitz.open(stream=file_bytes.getvalue(), filetype="pdf")
        pages: List[str] = []

        for page in doc:
            text = page.get_text().strip()
            if text:
                pages.append(text)

        if not pages:
            raise ValueError("PDF contains no extractable text")

        logger.warning("%s Extracted %d page(s) directly from PDF", log_stub, len(pages))
        return pages

    def _run_with_fallbacks(
        self,
        file_bytes,
        s3_key: str,
        log_stub: str,
    ) -> tuple[File.IngestExtractionStrategy, list[Element] | list[str]]:
        timeout_map = {
            "auto": env.document_pdf_extraction_default_timeout,
            "fast": env.document_pdf_extraction_fallback_one_timeout,
            "textract": env.document_pdf_extraction_fallback_two_timeout,
        }
        strategies = ["auto", "fast", "textract"]

        for i, strategy in enumerate(strategies, start=1):
            try:
                timeout = timeout_map.get(strategy, 20)

                logger.warning(
                    "%s Trying extraction strategy %s/%s '%s' with timeout=%ss",
                    log_stub,
                    i,
                    len(strategies),
                    strategy,
                    timeout,
                )

                if timeout == 0:
                    logger.warning("%s Skipping strategy=%s because timeout=0", log_stub, strategy)
                    continue

                with time_limit(timeout):
                    if strategy == "auto":
                        logger.warning("%s Trying Unstructured strategy=auto", log_stub)
                        result = self.unstructured._extract(file_bytes, s3_key, strategy="auto")
                        logger.warning("%s Successfully extracted with Unstructured strategy=auto", log_stub)
                        return File.IngestExtractionStrategy.unstructured_auto, result

                    if strategy == "fast":
                        logger.warning("%s Trying Unstructured strategy=fast", log_stub)
                        result = self.unstructured._extract(file_bytes, s3_key, strategy="fast")
                        logger.warning("%s Successfully extracted with Unstructured strategy=fast", log_stub)
                        return File.IngestExtractionStrategy.unstructured_fast, result

                    if strategy == "textract":
                        logger.warning("%s Trying Textract document_analysis", log_stub)
                        result = self.textract.document_analysis(key=s3_key)
                        logger.warning("%s Successfully extracted with Textract document_analysis", log_stub)
                        return File.IngestExtractionStrategy.textract_document_analysis, result

            except Exception:
                logger.exception(
                    "%s Strategy %s failed. Falling back...",
                    log_stub,
                    strategy,
                )

        try:
            logger.warning("%s All strategies failed. Using direct PDF text extraction fallback.", log_stub)
            result = self._extract_pdf_text_direct(file_bytes, log_stub)
            logger.warning("%s Successfully extracted with fallback direct PDF text extraction", log_stub)
            return File.IngestExtractionStrategy.pymupdf, result
        except Exception as e:
            logger.error("%s Final fallback (_extract_pdf_text_direct) also failed: %s", log_stub, str(e))
            raise RuntimeError("All extraction strategies including final fallback failed") from e

    def extract(
        self, file_name: str, chunk_resolution: ChunkResolution
    ) -> tuple[File.IngestExtractionStrategy, list[Element] | list[str] | list[dict[str, str]]]:
        self.extract_calls += 1
        extract_log_stub = f"{self.log_stub} (call {self.extract_calls}) {chunk_resolution} - "

        logger.warning(
            "%s .extract() called for %s",
            extract_log_stub,
            file_name,
        )

        s3_key = file_name
        display_name = os.path.basename(file_name).lower()

        logger.warning("%s File type detected: %s", extract_log_stub, display_name)

        obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        file_bytes = BytesIO(obj["Body"].read())

        if display_name.endswith((".csv", ".tsv", ".xls", ".xlsx")):
            logger.warning("%s Tabular detected: %s", extract_log_stub, display_name)
            result = load_tabular_file(display_name, file_bytes)
            logger.warning("%s Successfully extracted tabular file: %s", extract_log_stub, display_name)
            return File.IngestExtractionStrategy.tabular, result

        if display_name.lower().endswith((".pptx", ".ppt")):
            logger.warning("%s PowerPoint detected: %s", extract_log_stub, display_name)
            result = self.unstructured._extract_pptx(file_bytes)
            logger.warning("%s Successfully extracted PowerPoint: %s", extract_log_stub, display_name)
            return File.IngestExtractionStrategy.unstructured, result

        if display_name.lower().endswith(".docx"):
            logger.warning("%s DOCX detected: %s", extract_log_stub, display_name)
            result = self.unstructured._extract_docx(file_bytes)
            logger.warning("%s Successfully extracted DOCX: %s", extract_log_stub, display_name)
            return File.IngestExtractionStrategy.unstructured, result

        if display_name.endswith(".pdf"):
            logger.warning("%s PDF detected: %s", extract_log_stub, display_name)

            if chunk_resolution == ChunkResolution.largest:
                logger.warning("%s Using direct PDF text extraction.", extract_log_stub)
                result = self._extract_pdf_text_direct(file_bytes, extract_log_stub)
                logger.warning("%s Successfully extracted PDF via direct text extraction", extract_log_stub)
                return File.IngestExtractionStrategy.pymupdf, result

            logger.warning("%s Starting PDF extraction with fallbacks...", extract_log_stub)
            strategy, result = self._run_with_fallbacks(
                file_bytes=file_bytes,
                s3_key=s3_key,
                log_stub=extract_log_stub,
            )
            logger.warning("%s Successfully extracted PDF via fallback pipeline", extract_log_stub)
            return strategy, result

        logger.warning("%s No file type matched - defaulting to generic extraction...", extract_log_stub)
        logger.warning("%s Processing with unstructured: %s", extract_log_stub, display_name)

        result = self.unstructured._extract(file_bytes, file_name)
        logger.warning(
            "%s Successfully extracted via generic unstructured extraction: %s", extract_log_stub, display_name
        )
        return File.IngestExtractionStrategy.unstructured_auto, result
