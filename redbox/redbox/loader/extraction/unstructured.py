from io import BytesIO
import logging
from typing import List

from unstructured.documents.elements import Element
from unstructured.partition.docx import partition_docx
from unstructured.partition.auto import partition
from unstructured.partition.pptx import partition_pptx


logger = logging.getLogger(__name__)


class UnstructuredService:
    """
    Service layer wrapper around the `unstructured` library for extracting
    structured elements from binary document formats.

    Supported formats:
        - DOCX via `partition_docx`
        - PPTX via `partition_pptx`
        - Generic formats via `partition` (auto strategy or custom strategies)

    Methods:
        _extract_docx(file_bytes):
            Extracts elements from a DOCX file using `unstructured.partition.docx`.

        _extract_pptx(file_bytes):
            Extracts elements from a PPTX file using `unstructured.partition.pptx`,
            with additional logging for slide-level metadata debugging.

        _extract(file_bytes, file_name, strategy):
            Generic extraction method using `unstructured.partition`, supporting
            configurable parsing strategies (default: "auto").
    """

    def _extract_docx(self, file_bytes: BytesIO) -> List[Element]:
        logger.info("Extracting DOCX with unstructured")

        file_bytes.seek(0)

        try:
            elements = partition_docx(file=file_bytes)

            if not elements:
                raise ValueError("unstructured returned no elements from DOCX")

            return elements

        except Exception as e:
            logger.exception("unstructured failed to process DOCX: %s", str(e))
            raise

    def _extract_pptx(self, file_bytes: BytesIO) -> List[Element]:
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

            return elements

        except ImportError:
            logger.error("unstructured[pptx] extra not installed")
            raise
        except Exception as e:
            logger.exception("PPTX extraction failed: %s", e)
            raise

    def _extract(self, file_bytes: BytesIO, file_name: str, strategy: str = "auto") -> List[Element]:
        file_bytes.seek(0)

        elements = partition(file=file_bytes, strategy=strategy)

        if not elements:
            raise ValueError(f"unstructured returned no elements from {file_name}")

        return elements
