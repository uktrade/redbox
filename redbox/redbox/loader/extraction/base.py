from io import BytesIO
import os
import logging
from typing import List
import fitz
import boto3

from redbox.loader.extraction.checks import is_large_pdf, _pdf_is_image_heavy
from redbox.loader.extraction.textract import TextractService
from redbox.loader.extraction.unstructured import UnstructuredService
from redbox.loader.extraction.tabular import load_tabular_file

logger = logging.getLogger(__name__)


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

    def extract(self, file_name: str) -> list[str] | list[dict[str, str]]:
        logger.warning("DocumentExtractionService.extract() called for %s", file_name)

        s3_key = file_name

        display_name = os.path.basename(file_name).lower()
        logger.warning("File type detected: %s", display_name)

        obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        file_bytes = BytesIO(obj["Body"].read())

        if display_name.endswith((".csv", ".tsv", ".xls", ".xlsx")):
            return load_tabular_file(display_name, file_bytes)

        if display_name.lower().endswith(".pptx"):
            logger.warning("This is a PPTX file: %s", display_name)
            pages = self.unstructured._extract_pptx(file_bytes)

        if display_name.lower().endswith(".ppt"):
            logger.warning("This is a legacy PowerPoint file: %s", display_name)
            pages = self.unstructured._extract_pptx(file_bytes)

        if display_name.lower().endswith(".docx"):
            logger.warning("This is a document file: %s", display_name)
            pages = self.unstructured._extract_docx(file_bytes)

        if display_name.endswith(".pdf"):
            logger.warning("This is a PDF file: %s", display_name)
            large_pdf, page_count = is_large_pdf(display_name, file_bytes)
            if large_pdf:
                if _pdf_is_image_heavy(file_bytes):
                    logger.warning(
                        "Large image-heavy PDF detected (%d pages); using Textract with adaptive backoff",
                        page_count,
                    )

                    pages = self.textract.document_analysis(key=s3_key)
                else:
                    logger.warning(
                        "Large PDF detected (%d pages); extracting text directly instead of Textract",
                        page_count,
                    )
                    pages = self._extract_pdf_text_direct(file_bytes)
            else:
                try:
                    pages = self.textract.document_analysis(key=s3_key)
                except Exception:
                    logger.warning(
                        "Textract failed for %s; falling back to direct PDF text extraction",
                        display_name,
                    )
                    pages = self._extract_pdf_text_direct(file_bytes)

        else:
            logger.warning("Processing with unstructured: %s", display_name)
            pages = self.unstructured._extract(file_bytes, file_name)

        return pages
