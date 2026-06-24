from io import BytesIO
import os
import logging
from typing import List
import fitz
import boto3
import signal
from contextlib import contextmanager

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

        logger.warning(
            "Initialised DocumentExtractionService (bucket=%s, region=%s)",
            bucket,
            region,
        )

    def _extract_pdf_text_direct(self, file_bytes: BytesIO) -> List[str]:
        logger.warning("Extracting PDF text directly with PyMuPDF")
        file_bytes.seek(0)
        doc = fitz.open(stream=file_bytes.getvalue(), filetype="pdf")
        pages: List[str] = []

        for page in doc:
            text = page.get_text().strip()
            if text:
                pages.append(text)

        if not pages:
            raise ValueError("PDF contains no extractable text")

        logger.warning("Extracted %d page(s) directly from PDF", len(pages))
        return pages

    def _run_with_fallbacks(
        self,
        file_bytes,
        s3_key: str,
    ):
        timeout_map = {
            "auto": env.document_pdf_extraction_default_timeout,
            "fast": env.document_pdf_extraction_fallback_one_timeout,
            "textract": env.document_pdf_extraction_fallback_two_timeout,
        }

        for strategy in ["auto", "fast", "textract"]:
            try:
                logger.warning("Trying extraction strategy=%s", strategy)

                timeout = timeout_map.get(strategy, 20)
                if timeout == 0:
                    logger.warning("Skipping strategy=%s because timeout=0", strategy)
                    continue

                with time_limit(timeout):
                    if strategy == "auto":
                        return self.unstructured._extract(file_bytes, s3_key, strategy="auto")

                    if strategy == "fast":
                        return self.unstructured._extract(file_bytes, s3_key, strategy="fast")

                    if strategy == "textract":
                        return self.textract.document_analysis(key=s3_key)

            except Exception:
                logger.exception(
                    "Strategy %s failed. Falling back...",
                    strategy,
                )

        try:
            logger.warning("All strategies failed. Using direct PDF text extraction fallback.")
            return self._extract_pdf_text_direct(file_bytes)
        except Exception as e:
            logger.error("Final fallback (_extract_pdf_text_direct) also failed: %s", str(e))
            raise RuntimeError("All extraction strategies including final fallback failed") from e

        # raise RuntimeError("All extraction strategies failed")

    def extract(
        self, file_name: str, use_direct_extraction: bool = False
    ) -> list[Element] | list[str] | list[dict[str, str]]:
        logger.warning("DocumentExtractionService.extract() called for %s", file_name)

        s3_key = file_name
        display_name = os.path.basename(file_name).lower()

        logger.warning("File type detected: %s", display_name)

        obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        file_bytes = BytesIO(obj["Body"].read())

        if display_name.endswith((".csv", ".tsv", ".xls", ".xlsx")):
            logger.warning("Tabular detected: %s", display_name)
            return load_tabular_file(display_name, file_bytes)

        if display_name.lower().endswith((".pptx", ".ppt")):
            logger.warning("PowerPoint detected: %s", display_name)
            return self.unstructured._extract_pptx(file_bytes)

        if display_name.lower().endswith(".docx"):
            logger.warning("DOCX detected: %s", display_name)
            return self.unstructured._extract_docx(file_bytes)

        if display_name.endswith(".pdf"):
            logger.warning("PDF detected: %s", display_name)

            if use_direct_extraction:
                logger.warning("Using direct PDF text extraction.")
                return self._extract_pdf_text_direct(file_bytes)

            return self._run_with_fallbacks(
                file_bytes=file_bytes,
                s3_key=s3_key,
            )

        # if display_name.endswith(".pdf"):
        #     logger.warning("This is a PDF file: %s", display_name)
        #     large_pdf, page_count = is_large_pdf(display_name, file_bytes)
        #     if large_pdf:
        #         if _pdf_is_image_heavy(file_bytes):
        #             logger.warning(
        #                 "Large image-heavy PDF detected (%d pages); using Textract with adaptive backoff",
        #                 page_count,
        #             )

        #             return self.textract.document_analysis(key=s3_key)
        #         else:
        #             logger.warning(
        #                 "Large PDF detected (%d pages); extracting text directly instead of Textract",
        #                 page_count,
        #             )
        #             return self._extract_pdf_text_direct(file_bytes)
        #     else:
        #         try:
        #             return self.textract.document_analysis(key=s3_key)
        #         except Exception:
        #             logger.warning(
        #                 "Textract failed for %s; falling back to direct PDF text extraction",
        #                 display_name,
        #             )
        #             return self._extract_pdf_text_direct(file_bytes)

        logger.warning("No file type matched - defaulting to generic extraction...")
        logger.warning("Processing with unstructured: %s", display_name)
        return self.unstructured._extract(file_bytes, file_name)
