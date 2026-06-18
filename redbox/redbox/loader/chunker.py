import logging
import bisect
from typing import List, Iterator
from datetime import UTC, datetime

from langchain_core.documents import Document

from redbox.models.chain import GeneratedMetadata
from redbox.models.file import UploadedFileMetadata, ChunkResolution
from redbox.transform import bedrock_tokeniser


logger = logging.getLogger(__name__)

tokeniser = bedrock_tokeniser


class DocumentChunker:
    def __init__(
        self,
        chunk_resolution: ChunkResolution,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap_chars: int = 200,
        include_schema_metadata: bool = False,
    ):
        self.chunk_resolution = chunk_resolution
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_chars = overlap_chars
        self.include_schema_metadata = include_schema_metadata

        logger.info(
            "Initialised DocumentChunker (chunk_resolution=%s, min_chunk_size=%s, max_chunk_size=%s, overlap_chars=%s, include_schema_metadata=%s)",
            chunk_resolution,
            min_chunk_size,
            max_chunk_size,
            overlap_chars,
            include_schema_metadata,
        )

    def _page_for_offset(self, offset: int, page_offsets: list[int]) -> int:
        """Return the 1-indexed page number containing the given offset."""
        return bisect.bisect_right(page_offsets, offset)

    def _parse_pages(self, pages: list[str]) -> tuple[str, list[int]]:
        """Combine pages into one string, tracking each page's start offset."""
        full_text = ""
        page_offsets: list[int] = []
        separator = "\n\n"

        for i, page in enumerate(pages):
            page_offsets.append(len(full_text))
            full_text += page
            if i < len(pages) - 1:
                full_text += separator

        return full_text, page_offsets

    def _chunk_text(self, text: str) -> List[tuple[str, int]]:
        if not text:
            return []

        advance = self.max_chunk_size - self.overlap_chars
        if advance <= 0:
            raise ValueError(
                f"overlap_chars ({self.overlap_chars}) must be less than max_chunk_size ({self.max_chunk_size})"
            )

        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.max_chunk_size, length)
            chunk = text[start:end]

            if len(chunk) >= self.min_chunk_size or not chunks:
                chunks.append((chunk, start))

            start += advance

        return chunks

    def tabular_chunks(
        self, s3_key: str, tabular_elements: list[dict[str, str]], generated_metadata: GeneratedMetadata
    ) -> Iterator[Document]:
        for idx, el in enumerate(tabular_elements or []):
            metadata = UploadedFileMetadata(
                index=idx,
                uri=s3_key,
                page_number=1,
                created_datetime=datetime.now(UTC),
                token_count=tokeniser(el["text"]),
                chunk_resolution=ChunkResolution.tabular,
                name=generated_metadata.name,
                description=generated_metadata.description,
                keywords=generated_metadata.keywords,
            ).model_dump()

            merged_metadata = metadata
            if self.include_schema_metadata:
                merged_metadata = {**metadata, **el.get("metadata", {})}

            yield Document(page_content=el["text"], metadata=merged_metadata)

    def chunks(self, s3_key: str, pages: list[str], generated_metadata: GeneratedMetadata) -> Iterator[Document]:
        full_text, page_offsets = self._parse_pages(pages)
        chunks = self._chunk_text(full_text)

        if not chunks:
            logger.warning("No chunks produced for s3_key %s", s3_key)
            return

        for idx, (chunk, start_offset) in enumerate(chunks):
            page_num = self._page_for_offset(start_offset, page_offsets)
            metadata = UploadedFileMetadata(
                index=idx,
                uri=s3_key,
                page_number=page_num,
                created_datetime=datetime.now(UTC),
                token_count=tokeniser(chunk),
                chunk_resolution=self.chunk_resolution,
                name=generated_metadata.name,
                description=generated_metadata.description,
                keywords=generated_metadata.keywords,
            ).model_dump()
            yield Document(page_content=chunk, metadata=metadata)
