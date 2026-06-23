import logging
from typing import List, Iterator
from datetime import UTC, datetime

from langchain_core.documents import Document

from redbox.models.chain import GeneratedMetadata
from redbox.models.file import UploadedFileMetadata, ChunkResolution
from redbox.transform import bedrock_tokeniser


logger = logging.getLogger(__name__)

tokeniser = bedrock_tokeniser


class PageByPageDocumentChunker:
    def __init__(
        self,
        chunk_resolution: ChunkResolution,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap_chars: int = 200,
        include_schema_metadata: bool = False,
    ):
        if min_chunk_size <= 0:
            raise ValueError(f"{__name__} - min_chunk_size ({min_chunk_size}) must be >= 0")

        if max_chunk_size < min_chunk_size:
            raise ValueError(
                f"{__name__} - max_chunk_size ({max_chunk_size}) must be >= min_chunk_size ({min_chunk_size})"
            )

        if overlap_chars < 0:
            raise ValueError(f"{__name__} - overlap_chars ({overlap_chars}) must be >= 0")

        self.chunk_resolution = chunk_resolution
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_chars = overlap_chars
        self.include_schema_metadata = include_schema_metadata

        logger.info(
            "Initialised PageByPageDocumentChunker (chunk_resolution=%s, min_chunk_size=%s, max_chunk_size=%s, overlap_chars=%s, include_schema_metadata=%s)",
            chunk_resolution,
            min_chunk_size,
            max_chunk_size,
            overlap_chars,
            include_schema_metadata,
        )

    def _chunk_text(self, text: str) -> List[str]:
        if not text.strip():
            return []

        advance = self.max_chunk_size - self.overlap_chars
        if advance <= 0:
            raise ValueError(
                f"overlap_chars ({self.overlap_chars}) must be less than max_chunk_size ({self.max_chunk_size})"
            )

        chunks: list[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.max_chunk_size, length)
            chunk = text[start:end]

            if len(chunk) >= self.min_chunk_size or not chunks:
                chunks.append(chunk)

            start += advance

        return chunks

    def tabular_chunks(
        self, s3_key: str, tabular_elements: list[dict[str, str]], generated_metadata: GeneratedMetadata
    ) -> Iterator[Document]:
        created_datetime = datetime.now(UTC)

        for idx, el in enumerate(tabular_elements or []):
            metadata = UploadedFileMetadata(
                index=idx,
                uri=s3_key,
                page_number=1,
                created_datetime=created_datetime,
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

    def chunks(
        self,
        s3_key: str,
        pages: list[str],
        generated_metadata: GeneratedMetadata,
    ) -> Iterator[Document]:
        chunk_idx = 0

        created_datetime = datetime.now(UTC)

        for page_num, page_text in enumerate(pages, start=1):
            page_chunks = self._chunk_text(page_text)

            if not page_chunks:
                logger.debug(
                    "No chunks produced for s3_key=%s page=%s",
                    s3_key,
                    page_num,
                )
                continue

            for chunk in page_chunks:
                metadata = UploadedFileMetadata(
                    index=chunk_idx,
                    uri=s3_key,
                    page_number=page_num,
                    created_datetime=created_datetime,
                    token_count=tokeniser(chunk),
                    chunk_resolution=self.chunk_resolution,
                    name=generated_metadata.name,
                    description=generated_metadata.description,
                    keywords=generated_metadata.keywords,
                ).model_dump()

                yield Document(
                    page_content=chunk,
                    metadata=metadata,
                )

                chunk_idx += 1
