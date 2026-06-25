import logging
from typing import Iterator

from langchain_core.documents import Document
from unstructured.documents.elements import Element

from redbox_app.redbox_core.enums import IngestChunkingStrategy

from redbox.loader.chunking.chunkers.page_by_page import PageByPageDocumentChunker
from redbox.loader.chunking.chunkers.joined_pages import JoinedPagesDocumentChunker
from redbox.loader.chunking.chunkers.unstructured import UnstructuredDocumentChunker
from redbox.loader.chunking.chunkers.tabular import TabularDocumentChunker
from redbox.models.chain import GeneratedMetadata
from redbox.models.file import ChunkResolution

logger = logging.getLogger(__name__)


class DocumentChunkingService:
    """
    High-level service responsible for orchestrating document chunking strategies.

    This service acts as a unified interface over multiple chunking implementations,
    selecting and delegating to the appropriate chunker based on input data type
    and configuration options.

    It supports multiple ingestion strategies, including:
    - Page-by-page chunking
    - Joined/overlapping page chunking
    - Unstructured element-based chunking
    - Tabular data chunking

    The service ensures consistent configuration across chunkers and provides a
    single entry point for converting raw extracted content into LangChain `Document`
    chunks for downstream processing.

    Attributes:
        chunker_page_by_page (PageByPageDocumentChunker):
            Chunker that processes documents one page at a time.
        chunker_joined_pages (JoinedPagesDocumentChunker):
            Chunker that creates overlapping chunks across adjacent pages.
        chunker_unstructured (UnstructuredDocumentChunker):
            Chunker for unstructured elements (e.g. from Unstructured.io).
        chunker_tabular (TabularDocumentChunker):
            Chunker for structured/tabular data sources.
        log_stub (str):
            Logging prefix used for consistent log formatting.

    Methods:
        tabular_chunks(s3_key, tabular_elements, generated_metadata, include_schema_metadata):
            Generates chunks for tabular data sources and returns them with the
            associated ingestion strategy.

        chunks(s3_key, elements, generated_metadata, chunks_overlap_pages):
            Main entry point for document chunking. Routes input to the correct
            chunker based on element type (string pages vs unstructured Elements)
            and configuration flags.
    """

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

        self.log_stub = "[DocumentChunkingService]"

        logger.warning(
            "%s Initialised DocumentChunkingService (chunk_resolution=%s, min_chunk_size=%s, max_chunk_size=%s, overlap_chars=%s)",
            self.log_stub,
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
    ) -> tuple[IngestChunkingStrategy, Iterator[Document]]:
        logger.warning("%s Loading tabular chunks for %s...", self.log_stub, s3_key)
        result = self.chunker_tabular.tabular_chunks(
            s3_key=s3_key,
            tabular_elements=tabular_elements,
            generated_metadata=generated_metadata,
            include_schema_metadata=include_schema_metadata,
        )
        logger.warning("%s Successfully loaded tabular chunks for %s", self.log_stub, s3_key)
        return IngestChunkingStrategy.tabular, result

    def chunks(
        self,
        s3_key: str,
        elements: list[str] | list[Element],
        generated_metadata: GeneratedMetadata,
        chunks_overlap_pages: bool,
    ) -> tuple[IngestChunkingStrategy, Iterator[Document]]:
        logger.warning("%s Loading chunks for %s...", self.log_stub, s3_key)
        if not elements:
            logger.error("No extracted content passed to chunker...")
            raise ValueError("Unable to ingest chunks for null document contents.")

        # list[str] if using textract or pymupdf for extraction
        if all(isinstance(p, str) for p in elements):
            if chunks_overlap_pages:
                result = self.chunker_joined_pages.chunks(
                    s3_key=s3_key,
                    pages=elements,
                    generated_metadata=generated_metadata,
                )
                logger.warning("%s Successfully loaded overlapping-page chunks for %s...", self.log_stub, s3_key)
                return IngestChunkingStrategy.overlapping_pages, result

            result = self.chunker_page_by_page.chunks(
                s3_key=s3_key,
                pages=elements,
                generated_metadata=generated_metadata,
            )
            logger.warning("%s Successfully loaded page-by-page chunks for %s...", self.log_stub, s3_key)
            return IngestChunkingStrategy.page_by_page, result

        # list[Element] if using unstructured
        if all(isinstance(p, Element) for p in elements):
            result = self.chunker_unstructured.chunks(
                s3_key=s3_key,
                elements=elements,
                generated_metadata=generated_metadata,
            )
            logger.warning("%s Successfully loaded unstructured chunks for %s...", self.log_stub, s3_key)
            return IngestChunkingStrategy.unstructured_chunk_by_title, result

        raise TypeError("pages must be either list[str] or list[Element], not mixed")
