from pathlib import Path

from io import BytesIO
from redbox.loader.extraction.base import DocumentExtractor, ExtractionResult


class ExtractorRegistry:
    def __init__(self, extractors: list[DocumentExtractor]):
        self._map: dict[str, DocumentExtractor] = {}
        for extractor in extractors:
            for ext in extractor.supported_extensions:
                self._map[ext] = extractor

    def get(self, file_name: str) -> DocumentExtractor:
        ext = Path(file_name).suffix.lower()
        if ext not in self._map:
            raise ValueError(f"No extractor registered for '{ext}'")
        return self._map[ext]

    def extract(self, file_name: str, file_bytes: BytesIO) -> ExtractionResult:
        return self.get(file_name).extract(file_name, file_bytes)


# # Wire it up once at startup
# registry = ExtractorRegistry([
#     PdfExtractor(bucket="my-bucket"),
#     DocxExtractor(),
#     # PptxExtractor(),
#     # MarkdownExtractor(),
#     # TabularExtractor(),
#     # GenericExtractor(),   # fallback registered for remaining extensions
# ])
