import logging

from datetime import UTC, datetime
from typing import Iterator
from langchain_core.documents import Document

from redbox.models.chain import GeneratedMetadata
from redbox.models.file import UploadedFileMetadata, ChunkResolution
from redbox.transform import bedrock_tokeniser
from redbox.loader.chunker import DocumentChunker, LayoutBlock

logger = logging.getLogger(__name__)

tokeniser = bedrock_tokeniser


class TextractChunkLoader:
    """Chunks pre-extracted layout blocks. No extraction happens here."""

    def __init__(
        self,
        chunk_resolution: ChunkResolution,
        min_chunk_size: int,
        max_chunk_size: int,
        overlap_chars: int,
        metadata: GeneratedMetadata | None = None,
        include_schema_metadata: bool = False,
    ):
        self.chunk_resolution = chunk_resolution
        self.metadata = metadata or GeneratedMetadata(name="", description="", keywords=[])
        self.include_schema_metadata = include_schema_metadata
        self.chunker = DocumentChunker(
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap_chars=overlap_chars,
            tokeniser=tokeniser,
        )

    def lazy_load_from_blocks(
        self,
        file_name: str,
        layout_blocks: list[LayoutBlock] | None,
        tabular_elements: list[dict] | None,
    ) -> Iterator[Document]:
        s3_key = file_name

        if tabular_elements is not None:
            for idx, el in enumerate(tabular_elements):
                metadata = UploadedFileMetadata(
                    index=idx,
                    uri=s3_key,
                    page_number=1,
                    created_datetime=datetime.now(UTC),
                    token_count=tokeniser(el["text"]),
                    chunk_resolution=ChunkResolution.tabular,
                    name=self.metadata.name,
                    description=self.metadata.description,
                    keywords=self.metadata.keywords,
                ).model_dump()

                merged_metadata = metadata
                if self.include_schema_metadata:
                    merged_metadata = {**metadata, **el.get("metadata", {})}

                yield Document(page_content=el["text"], metadata=merged_metadata)
            return

        for idx, chunk in enumerate(self.chunker.chunk(layout_blocks)):
            metadata = UploadedFileMetadata(
                index=idx,
                uri=s3_key,
                page_number=chunk.page_start,
                created_datetime=datetime.now(UTC),
                token_count=tokeniser(chunk.text),
                chunk_resolution=self.chunk_resolution,
                name=self.metadata.name,
                description=self.metadata.description,
                keywords=self.metadata.keywords,
            ).model_dump()

            yield Document(page_content=chunk.text, metadata=metadata)
