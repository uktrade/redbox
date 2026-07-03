from io import BytesIO
import os
from uuid import uuid4
import logging
from typing import Callable
from functools import partial
from dataclasses import dataclass
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz
import boto3

from django.core.cache import cache

from redbox_app.redbox_core.enums import IngestExtractionStrategy
from redbox.models.file import ChunkResolution
from redbox.models.settings import get_settings
from unstructured.documents.elements import Element

from redbox.loader.loaders import load_tabular_file
from redbox.loader.extraction.textract import TextractService, TextractTimeout
from redbox.loader.extraction.unstructured import UnstructuredService

logger = logging.getLogger(__name__)

env = get_settings()

# How long an idempotency lock is held for, in seconds
INGEST_LOCK_TIMEOUT_SECONDS = env.document_ingest_lock_timeout_seconds

# Size threshold (bytes) above which a PDF is routed directly to Textract
LARGE_PDF_BYTES_THRESHOLD = env.document_large_pdf_bytes_threshold


@dataclass
class ExtractionStrategyConfig:
    name: IngestExtractionStrategy
    ingestion_fn: Callable[[], object]


STRATEGY_TIMEOUT_MAP = {
    env.document_pdf_extraction_default_strategy: env.document_pdf_extraction_default_timeout,
    env.document_pdf_extraction_fallback_one_strategy: env.document_pdf_extraction_fallback_one_timeout,
    env.document_pdf_extraction_fallback_two_strategy: env.document_pdf_extraction_fallback_two_timeout,
}
STRATEGIES = [
    env.document_pdf_extraction_default_strategy,
    env.document_pdf_extraction_fallback_one_strategy,
    env.document_pdf_extraction_fallback_two_strategy,
]


@dataclass
class ExtractedPdf:
    bytes: BytesIO
    page_count: int
    file_size: int


class IngestionAlreadyInProgress(Exception):
    """Raised when an idempotency lock for this file is already held."""


class DocumentExtractionService:
    """
    Orchestrates document extraction from S3-backed files using multiple
    extraction engines with fallback and timeout-aware execution.

    Strategies:
        - Unstructured-based parsing (auto/fast modes)
        - AWS Textract (document analysis)
        - PyMuPDF direct text extraction (PDF fallback)
        - Tabular file loader for structured datasets

    Large PDFs are routed straight to Textract, skipping `unstructured`
    entirely, since `unstructured_auto` is typically the slowest strategy
    and the least suited to large documents.

    Each call to `extract()` is guarded by a short-lived idempotency lock
    keyed on the S3 key, so that a Django Q retry firing while a previous
    attempt is still legitimately running (or an OOM/redeploy causes
    redelivery) does not result in duplicate concurrent extractions.
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

    @staticmethod
    def _lock_key(s3_key: str) -> str:
        return f"ingest-lock:{s3_key}"

    def _convert_office_to_pdf(
        self,
        file_bytes: BytesIO,
        suffix: str,
        log_stub: str,
    ) -> ExtractedPdf:
        """
        Convert an Office document to PDF using LibreOffice.

        Returns:
            BytesIO containing the generated PDF.
        """
        soffice = shutil.which("soffice")
        if soffice is None:
            mac_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
            if Path(mac_path).exists():
                soffice = mac_path
            else:
                raise RuntimeError("LibreOffice (soffice) is not installed.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_file = tmpdir / f"input{suffix}"
            output_pdf = tmpdir / "input.pdf"

            input_file.write_bytes(file_bytes.getvalue())

            logger.warning("%s Converting %s to PDF", log_stub, suffix)

            subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(tmpdir),
                    str(input_file),
                ],
                check=True,
                timeout=120,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if not output_pdf.exists():
                raise RuntimeError("LibreOffice did not produce a PDF.")

            pdf = BytesIO(output_pdf.read_bytes())

            with fitz.open(stream=pdf.getvalue(), filetype="pdf") as doc:
                page_count = doc.page_count

            logger.warning(
                "%s Office document converted to PDF (%d pages)",
                log_stub,
                page_count,
            )

            return ExtractedPdf(bytes=pdf, page_count=page_count, file_size=len(pdf.getbuffer()))

    def docx_to_pdf(
        self,
        file_bytes: BytesIO,
        log_stub: str,
    ) -> ExtractedPdf:
        """Convert a DOCX document to PDF."""
        return self._convert_office_to_pdf(
            file_bytes=file_bytes,
            suffix=".docx",
            log_stub=log_stub,
        )

    def pptx_to_pdf(
        self,
        file_bytes: BytesIO,
        log_stub: str,
    ) -> ExtractedPdf:
        """Convert a PowerPoint presentation to PDF."""
        return self._convert_office_to_pdf(
            file_bytes=file_bytes,
            suffix=".pptx",
            log_stub=log_stub,
        )

    def _extract_pdf_text_direct(
        self,
        file_bytes: BytesIO,
        log_stub: str,
    ) -> list[str]:
        logger.warning("%s Extracting PDF text directly with PyMuPDF", log_stub)

        file_bytes.seek(0)

        with fitz.open(stream=file_bytes.getvalue(), filetype="pdf") as doc:
            pages = [text for page in doc if (text := page.get_text().strip())]

        if not pages:
            raise ValueError("PDF contains no extractable text")

        logger.warning("%s Extracted %d page(s) directly from PDF", log_stub, len(pages))

        return pages

    def _run_with_fallbacks(
        self,
        file_bytes: BytesIO,
        s3_key: str,
        log_stub: str,
        use_s3_textract: bool = True,
    ) -> tuple[IngestExtractionStrategy, list[Element] | list[str]]:
        strategy_map: dict[str, ExtractionStrategyConfig] = {
            "unstructured_auto": ExtractionStrategyConfig(
                name=IngestExtractionStrategy.unstructured_auto,
                ingestion_fn=partial(self.unstructured._extract, file_bytes, s3_key, strategy="auto"),
            ),
            "unstructured_fast": ExtractionStrategyConfig(
                name=IngestExtractionStrategy.unstructured_fast,
                ingestion_fn=partial(self.unstructured._extract, file_bytes, s3_key, strategy="fast"),
            ),
            "textract_document_analysis": ExtractionStrategyConfig(
                name=IngestExtractionStrategy.textract_document_analysis,
                ingestion_fn=partial(
                    self.textract.document_analysis,
                    key=s3_key,
                    file_bytes=None if use_s3_textract else file_bytes,
                ),
            ),
        }

        for i, strategy in enumerate(STRATEGIES, start=1):
            config = strategy_map.get(strategy)
            if config is None:
                logger.error("%s Skipping unsupported strategy=%s, config could not be found", log_stub, strategy)
                continue

            timeout = STRATEGY_TIMEOUT_MAP.get(strategy, 20)

            if timeout == 0:
                logger.warning("%s Skipping strategy=%s because timeout=0", log_stub, strategy)
                continue

            logger.warning(
                "%s Trying extraction strategy %s/%s '%s' with timeout=%ss",
                log_stub,
                i,
                len(STRATEGIES),
                strategy,
                timeout,
            )

            try:
                result = config.ingestion_fn(timeout=timeout)
                logger.warning("%s Strategy '%s' succeeded", log_stub, strategy)
                return config.name, result

            except TextractTimeout:
                logger.warning(
                    "%s Direct Textract timed out after %ss. Falling back to PyMuPDF.",
                    log_stub,
                    timeout,
                )

            except Exception as e:
                logger.exception("%s Strategy '%s' failed: %s", log_stub, strategy, str(e))

        try:
            logger.warning("%s All strategies failed. Using direct PDF text extraction fallback.", log_stub)
            result = self._extract_pdf_text_direct(file_bytes, log_stub)
            logger.warning("%s Successfully extracted with fallback direct PDF text extraction", log_stub)
            return IngestExtractionStrategy.pymupdf, result
        except Exception as e:
            logger.error("%s Final fallback (_extract_pdf_text_direct) also failed: %s", log_stub, str(e))
            raise RuntimeError("All extraction strategies including final fallback failed") from e

    def _extract_pdf(
        self,
        *,
        s3_key: str,
        pdf: ExtractedPdf,
        chunk_resolution: ChunkResolution,
        log_stub: str,
        use_s3_textract: bool = True,
    ) -> tuple[IngestExtractionStrategy, list[Element] | list[str]]:
        """
        Extract text/layout from a PDF.
        This method is shared by: native PDF uploads, DOC/DOCX converted to PDF, PPT/PPTX converted to PDF

        Args:
            s3_key:
                Original file name (used for logging and S3 Textract where possible).

            file_bytes:
                PDF bytes.

            file_size:
                Size of the PDF in bytes.

            chunk_resolution:
                Requested chunk resolution.

            log_stub:
                Logging prefix.

            use_s3_textract:
                If True, Textract will reference the S3 object directly.
                If False, Textract should operate on the supplied PDF bytes
                (used for converted Office documents).
        """

        logger.warning("%s PDF detected (%d bytes)", log_stub, pdf.file_size)

        # Largest chunks always use direct text extraction.
        if chunk_resolution == ChunkResolution.largest:
            logger.warning(
                "%s Using direct PyMuPDF extraction for largest chunk resolution",
                log_stub,
            )
            return (
                IngestExtractionStrategy.pymupdf,
                self._extract_pdf_text_direct(pdf.bytes, log_stub),
            )

        # Large PDFs bypass unstructured entirely.
        if pdf.page_count > 200 or pdf.file_size > LARGE_PDF_BYTES_THRESHOLD:
            logger.warning(
                "%s Large PDF (%d > 200 pages or %d > %d bytes); using Textract large-document pipeline",
                log_stub,
                pdf.page_count,
                pdf.file_size,
                LARGE_PDF_BYTES_THRESHOLD,
            )

            timeout = env.document_large_pdf_timeout

            try:
                result = self.textract.document_analysis_large(
                    key=s3_key,
                    file_bytes=None if use_s3_textract else pdf.bytes,
                    timeout=timeout,
                )

                logger.warning("%s Large Textract extraction succeeded", log_stub)

                return (
                    IngestExtractionStrategy.textract_document_analysis_large,
                    result,
                )

            except TextractTimeout:
                logger.exception(
                    "%s Large Textract timed out after %ss. Falling back to PyMuPDF.",
                    log_stub,
                    timeout,
                )

                return (
                    IngestExtractionStrategy.pymupdf,
                    self._extract_pdf_text_direct(pdf.bytes, log_stub),
                )

        logger.warning("%s Starting PDF fallback extraction pipeline", log_stub)

        return self._run_with_fallbacks(
            file_bytes=pdf.bytes,
            s3_key=s3_key,
            log_stub=log_stub,
            use_s3_textract=use_s3_textract,
        )

    def _extract_locked(
        self, file_name: str, chunk_resolution: ChunkResolution
    ) -> tuple[IngestExtractionStrategy, list[Element] | list[str] | list[dict[str, str | dict]]]:
        self.extract_calls += 1
        extract_log_stub = f"{self.log_stub} (call {self.extract_calls}) {chunk_resolution} - "

        logger.warning("%s .extract() called for %s", extract_log_stub, file_name)

        s3_key = file_name
        display_name = os.path.basename(file_name).lower()

        # HEAD only - avoids pulling the whole object into memory for files that won't need local bytes
        head = self.s3.head_object(Bucket=self.bucket, Key=s3_key)
        file_size = head["ContentLength"]

        logger.warning("%s File type detected: %s (%d bytes)", extract_log_stub, display_name, file_size)

        def get_bytes() -> BytesIO:
            obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            return BytesIO(obj["Body"].read())

        if display_name.endswith((".csv", ".tsv", ".xls", ".xlsx")):
            logger.warning("%s Tabular detected: %s", extract_log_stub, display_name)
            result = load_tabular_file(display_name, get_bytes())
            return IngestExtractionStrategy.tabular, result

        if display_name.endswith(".pdf"):
            pdf = get_bytes()
            with fitz.open(stream=pdf.getvalue(), filetype="pdf") as doc:
                extracted_pdf = ExtractedPdf(
                    bytes=pdf,
                    page_count=doc.page_count,
                    file_size=len(pdf.getbuffer()),
                )

            return self._extract_pdf(
                s3_key=s3_key,
                pdf=extracted_pdf,
                chunk_resolution=chunk_resolution,
                log_stub=extract_log_stub,
                use_s3_textract=True,
            )

        if display_name.endswith((".doc", ".docx")):
            pdf = self.docx_to_pdf(get_bytes(), extract_log_stub)
            return self._extract_pdf(
                s3_key=s3_key,
                pdf=pdf,
                chunk_resolution=chunk_resolution,
                log_stub=extract_log_stub,
                use_s3_textract=False,
            )

        if display_name.endswith((".ppt", ".pptx")):
            pdf = self.pptx_to_pdf(get_bytes(), extract_log_stub)
            return self._extract_pdf(
                s3_key=s3_key,
                pdf=pdf,
                chunk_resolution=chunk_resolution,
                log_stub=extract_log_stub,
                use_s3_textract=False,
            )

        logger.warning("%s No file type matched - defaulting to generic extraction...", extract_log_stub)
        result = self.unstructured._extract(get_bytes(), file_name)
        return IngestExtractionStrategy.unstructured_auto, result

    def extract(
        self, file_name: str, chunk_resolution: ChunkResolution
    ) -> tuple[IngestExtractionStrategy, list[Element] | list[str] | list[dict[str, str]]]:
        lock_key = self._lock_key(f"{file_name}:{chunk_resolution}")

        if not cache.add(lock_key, "1", timeout=INGEST_LOCK_TIMEOUT_SECONDS):
            logger.warning(
                "%s Ingestion already in progress for %s (%s) - skipping duplicate run",
                self.log_stub,
                file_name,
                chunk_resolution,
            )
            raise IngestionAlreadyInProgress(file_name)

        try:
            return self._extract_locked(file_name, chunk_resolution)
        finally:
            cache.delete(lock_key)
