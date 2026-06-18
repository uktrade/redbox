from io import BytesIO
import logging
from typing import List


from unstructured.partition.docx import partition_docx
from unstructured.partition.auto import partition
from unstructured.partition.pptx import partition_pptx


logger = logging.getLogger(__name__)


class UnstructuredService:
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

                text = str(el).strip()
                if text:
                    current_page.append(text)

            if current_page:
                text_pages.append("\n".join(current_page))

            if not any(p.strip() for p in text_pages):
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
            if elements:
                logger.info("First element: %s", repr(elements[0]))
                logger.info("Has slide_number? %s", hasattr(elements[0].metadata, "slide_number"))
                slide_nums = {getattr(el.metadata, "slide_number", None) for el in elements}
                logger.info("Unique slide numbers found: %s", sorted(slide_nums - {None}))
            else:
                logger.warning("partition_pptx returned ZERO elements!")

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

    def _extract(self, file_bytes: BytesIO, file_name: str) -> List[str]:
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
