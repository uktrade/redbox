import logging
from pathlib import Path
from typing import Any
from io import BytesIO

from unstructured.partition.docx import partition_docx

from redbox.loader.extraction.base import DocumentExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class DocxExtractor(DocumentExtractor):
    @property
    def supported_extensions(self) -> set[str]:
        return {".docx"}

    def _extract(self, path: Path, file_bytes: BytesIO) -> ExtractionResult:
        logger.warning("Extracting DOCX with unstructured")
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

            logger.warning("Extracted %d page(s) from DOCX using unstructured", len(text_pages))

            return text_pages

        except Exception as e:
            logger.exception("unstructured failed to process DOCX: %s", str(e))
            raise

    def extract_metadata(self, file_bytes: BytesIO) -> dict[str, Any]: ...
    def extract_images(self, file_bytes: BytesIO) -> list[bytes]: ...
    def extract_tables(self, file_bytes: BytesIO) -> list[dict]: ...
