import logging
from typing import Iterator

from langchain_core.documents import Document
from unstructured.documents.elements import Element

from redbox.loader.chunking.chunkers.page_by_page import PageByPageDocumentChunker
from redbox.loader.chunking.chunkers.joined_pages import JoinedPagesDocumentChunker
from redbox.loader.chunking.chunkers.unstructured import UnstructuredDocumentChunker
from redbox.loader.chunking.chunkers.tabular import TabularDocumentChunker
from redbox.models.chain import GeneratedMetadata
from redbox.models.file import ChunkResolution

logger = logging.getLogger(__name__)


class DocumentChunkingService:
    def __init__(
        self,
        chunk_resolution: ChunkResolution,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap_chars: int = 200,
    ):
        self.chunker_page_by_page = PageByPageDocumentChunker(
            chunk_resolution=chunk_resolution,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap_chars=overlap_chars,
        )
        self.chunker_joined_pages = JoinedPagesDocumentChunker(
            chunk_resolution=chunk_resolution,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap_chars=overlap_chars,
        )
        self.chunker_unstructured = UnstructuredDocumentChunker(
            chunk_resolution=chunk_resolution,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap_chars=overlap_chars,
        )
        self.chunker_tabular = TabularDocumentChunker(
            chunk_resolution=chunk_resolution,
        )

        logger.info(
            "Initialised DocumentChunkerService (chunk_resolution=%s, min_chunk_size=%s, max_chunk_size=%s, overlap_chars=%s)",
            chunk_resolution,
            min_chunk_size,
            max_chunk_size,
            overlap_chars,
        )

    def tabular_chunks(
        self,
        s3_key: str,
        tabular_elements: list[dict[str, str]],
        generated_metadata: GeneratedMetadata,
        include_schema_metadata: bool,
    ) -> Iterator[Document]:
        return self.chunker_tabular.tabular_chunks(
            s3_key=s3_key,
            tabular_elements=tabular_elements,
            generated_metadata=generated_metadata,
            include_schema_metadata=include_schema_metadata,
        )

    def chunks(
        self,
        s3_key: str,
        elements: list[str] | list[Element],
        generated_metadata: GeneratedMetadata,
        chunks_overlap_pages: bool,
    ) -> Iterator[Document]:
        if not elements:
            logger.error("No extracted content passed to chunker...")
            raise ValueError("Unable to ingest chunks for null document contents.")

        # list[str] if using textract or pymupdf for extraction
        if all(isinstance(p, str) for p in elements):
            if chunks_overlap_pages:
                return self.chunker_joined_pages.chunks(
                    s3_key=s3_key,
                    pages=elements,
                    generated_metadata=generated_metadata,
                )

            return self.chunker_page_by_page.chunks(
                s3_key=s3_key,
                pages=elements,
                generated_metadata=generated_metadata,
            )

        # list[Element] if using unstructured
        if all(isinstance(p, Element) for p in elements):
            return self.chunker_unstructured.chunks(
                s3_key=s3_key,
                elements=elements,
                generated_metadata=generated_metadata,
            )

        raise TypeError("pages must be either list[str] or list[Element], not mixed")
