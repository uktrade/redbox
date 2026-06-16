from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from redbox.loader.chunker import LayoutBlock


@dataclass
class ExtractionResult:
    source_path: str
    file_type: str
    layout_blocks: list[LayoutBlock]
    tables: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.layout_blocks)

    @property
    def page_count(self) -> int:
        pages = {b.page_number for b in self.layout_blocks}
        return len(pages)


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
    def supported_extensions(self) -> set[str]:
        """e.g. {'.pdf'} or {'.docx'}"""

    @abstractmethod
    def _extract(self, path: Path, file_bytes: BytesIO) -> ExtractionResult:
        """Perform extraction and return a normalised ExtractionResult."""

    @abstractmethod
    def extract_metadata(self, file_bytes: BytesIO) -> dict[str, Any]:
        """Return document-level metadata (author, title, created date…)."""

    @abstractmethod
    def extract_images(self, file_bytes: BytesIO) -> list[bytes]:
        """Return raw image bytes found in the document."""

    @abstractmethod
    def extract_tables(self, file_bytes: BytesIO) -> list[dict]:
        """Return structured table data found in the document."""
