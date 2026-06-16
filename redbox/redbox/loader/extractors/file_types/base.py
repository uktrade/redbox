from enum import StrEnum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path


class FileType(StrEnum):
    PDF = ".pdf"
    DOCX = ".docx"
    PPTX = ".pptx"
    PPT = ".ppt"
    XLS = ".xls"
    XLSX = ".xlsx"
    CSV = ".csv"
    TSV = ".tsv"
    OTHER = "other"

    def is_tabular(self) -> bool:
        return self in [FileType.XLSX, FileType.XLS, FileType.CSV, FileType.TSV]


@dataclass
class ExtractionResult:
    source_path: str
    file_type: FileType
    result: list[str]  # | list[LayoutBlock]


class DocumentExtractor(ABC):
    """Abstract base for all document extractors."""

    def extract(self, file_name: str, file_bytes: BytesIO) -> ExtractionResult:
        """Public entry point. Validates, then delegates to _extract."""
        path = Path(file_name)
        self._validate(path)
        return self._extract(path, file_bytes)

    def _validate(self, path: Path) -> None:
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"{self.__class__.__name__} does not support '{path.suffix}'. Expected: {self.supported_extensions}"
            )

    # ── Abstract interface ──────────────────────────────────────────────── #

    @property
    @abstractmethod
    def supported_extensions(self) -> set[FileType]:
        """e.g. {'.pdf'} or {'.docx'}"""

    @abstractmethod
    def _extract(self, path: Path, file_bytes: BytesIO) -> ExtractionResult:
        """Perform extraction and return a normalised ExtractionResult."""

    # @abstractmethod
    # def extract_metadata(self, file_bytes: BytesIO) -> dict[str, Any]:
    #     """Return document-level metadata (author, title, created date…)."""

    # @abstractmethod
    # def extract_images(self, file_bytes: BytesIO) -> list[bytes]:
    #     """Return raw image bytes found in the document."""

    # @abstractmethod
    # def extract_tables(self, file_bytes: BytesIO) -> list[dict]:
    #     """Return structured table data found in the document."""
