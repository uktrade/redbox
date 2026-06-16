import logging
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
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap_chars: int = 200,
        include_schema_metadata: bool = False,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_chars = overlap_chars
        self.include_schema_metadata = include_schema_metadata

        logger.info(
            "Initialised DocumentChunker (min_chunk_size=%s, max_chunk_size=%s, overlap_chars=%s, include_schema_metadata=%s)",
            min_chunk_size,
            max_chunk_size,
            overlap_chars,
            include_schema_metadata,
        )

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []

        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.max_chunk_size, length)
            chunk = text[start:end]

            if len(chunk) >= self.min_chunk_size or not chunks:
                chunks.append(chunk)

            start = end - self.overlap_chars

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
        idx = 0
        for page_num, page_text in enumerate(pages, start=1):
            chunks = self._chunk_text(page_text)
            if not chunks:
                logger.warning("No chunks produced for page %s", page_num)
                continue
            for chunk in chunks:
                metadata = UploadedFileMetadata(
                    index=idx,
                    uri=s3_key,
                    page_number=page_num,
                    created_datetime=datetime.now(UTC),
                    token_count=tokeniser(chunk),
                    chunk_resolution=ChunkResolution.normal,
                    name=generated_metadata.name,
                    description=generated_metadata.description,
                    keywords=generated_metadata.keywords,
                ).model_dump()
                yield Document(page_content=chunk, metadata=metadata)
                idx += 1
