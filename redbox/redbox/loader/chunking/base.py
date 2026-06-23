import logging
from typing import Iterator

from langchain_core.documents import Document
from unstructured.documents.elements import Element

from redbox.loader.chunking.page_by_page import PageByPageDocumentChunker
from redbox.loader.chunking.unstructured import UnstructuredDocumentChunker
from redbox.models.chain import GeneratedMetadata
from redbox.models.file import ChunkResolution
from redbox.transform import bedrock_tokeniser


logger = logging.getLogger(__name__)

tokeniser = bedrock_tokeniser


class DocumentChunkingService:
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
            "Initialised DocumentChunker (chunk_resolution=%s, min_chunk_size=%s, max_chunk_size=%s, overlap_chars=%s, include_schema_metadata=%s)",
            chunk_resolution,
            min_chunk_size,
            max_chunk_size,
            overlap_chars,
            include_schema_metadata,
        )

        self.chunker_page_by_page = PageByPageDocumentChunker(
            chunk_resolution=chunk_resolution,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap_chars=overlap_chars,
            include_schema_metadata=include_schema_metadata,
        )
        self.chunker_unstructured = UnstructuredDocumentChunker(
            chunk_resolution=chunk_resolution,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap_chars=overlap_chars,
            include_schema_metadata=include_schema_metadata,
        )

    def tabular_chunks(
        self, s3_key: str, tabular_elements: list[dict[str, str]], generated_metadata: GeneratedMetadata
    ) -> Iterator[Document]:
        return self.chunker_page_by_page.tabular_chunks(
            s3_key=s3_key,
            tabular_elements=tabular_elements,
            generated_metadata=generated_metadata,
        )

    def chunks(
        self,
        s3_key: str,
        elements: list[str]
        | list[Element],  # list[str] if using textract or pymupdf for extraction, list[Element] if using unstructured
        generated_metadata: GeneratedMetadata,
    ) -> Iterator[Document]:
        if not elements:
            logger.error("No extracted content passed to chunker...")
            raise RuntimeError("Unable to ingest chunks for null document contents.")

        if all(isinstance(p, str) for p in elements):
            return self.chunker_page_by_page.chunks(
                s3_key=s3_key,
                pages=elements,
                generated_metadata=generated_metadata,
            )

        if all(isinstance(p, Element) for p in elements):
            return self.chunker_unstructured.chunks(
                s3_key=s3_key,
                elements=elements,
                generated_metadata=generated_metadata,
            )

        raise TypeError("pages must be either list[str] or list[Element], not mixed")
