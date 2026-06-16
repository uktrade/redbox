from pathlib import Path
from typing import Any
from io import BytesIO

from redbox.loader.textract.extractor import TextractExtractor
from redbox.loader.extraction.base import DocumentExtractor, ExtractionResult


class PdfExtractor(DocumentExtractor):
    def __init__(self, bucket: str, region: str = "eu-west-2"):
        self._textract = TextractExtractor(bucket, region)  # delegate, not inherit

    @property
    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    def _extract(self, path: Path, file_bytes: BytesIO) -> ExtractionResult:
        blocks = self._textract._extract_pdf_layout(file_bytes, s3_key=str(path), display_name=path.name)
        return ExtractionResult(
            source_path=str(path),
            file_type="pdf",
            layout_blocks=blocks,
            metadata=self.extract_metadata(file_bytes),
        )

    def extract_metadata(self, file_bytes: BytesIO) -> dict[str, Any]: ...
    def extract_images(self, file_bytes: BytesIO) -> list[bytes]: ...
    def extract_tables(self, file_bytes: BytesIO) -> list[dict]: ...
