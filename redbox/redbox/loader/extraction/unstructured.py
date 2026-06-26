from io import BytesIO
import logging
from typing import List

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from unstructured.documents.elements import Element
from unstructured.partition.docx import partition_docx
from unstructured.partition.auto import partition
from unstructured.partition.pptx import partition_pptx


logger = logging.getLogger(__name__)


class UnstructuredTimeout(TimeoutError):
    pass


class UnstructuredService:
    """
    Service layer wrapper around the `unstructured` library for extracting
    structured elements from binary document formats.
    """

    def _run_with_timeout(self, fn, timeout: int | None):
        if timeout is None:
            return fn()

        executor = ThreadPoolExecutor(max_workers=1)

        future = executor.submit(fn)

        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as e:
            raise UnstructuredTimeout(f"Unstructured extraction exceeded {timeout}s") from e
        finally:
            executor.shutdown(wait=False)

    def _extract_docx(self, file_bytes: BytesIO, timeout: int | None = None) -> List[Element]:
        logger.info("Extracting DOCX with unstructured")

        file_bytes.seek(0)

        def extract():
            try:
                elements = partition_docx(file=file_bytes)

                if not elements:
                    raise ValueError("unstructured returned no elements from DOCX")

                return elements

            except Exception as e:
                logger.exception("unstructured failed to process DOCX: %s", str(e))
                raise

        return self._run_with_timeout(extract, timeout)

    def _extract_pptx(self, file_bytes: BytesIO, timeout: int | None = None) -> List[Element]:
        logger.info("Extracting PPTX with unstructured.partition.pptx")
        file_bytes.seek(0)

        def extract():
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

        return self._run_with_timeout(extract, timeout)

    def _extract(
        self, file_bytes: BytesIO, file_name: str, strategy: str = "auto", timeout: int | None = None
    ) -> List[Element]:
        file_bytes.seek(0)

        def extract():
            elements = partition(
                file=file_bytes,
                strategy=strategy,
            )

            if not elements:
                raise ValueError(f"unstructured returned no elements from {file_name}")

            return elements

        return self._run_with_timeout(extract, timeout)
