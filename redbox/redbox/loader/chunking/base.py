from abc import ABC, abstractmethod
from typing import Iterator
import logging

from langchain_core.documents import Document

from redbox.models.chain import GeneratedMetadata
from redbox.models.file import UploadedFileMetadata, ChunkResolution
from redbox.loader.chunking.tokeniser import tokeniser

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """
    Abstract base class defining a generic interface for chunking operations.

    Subclasses are expected to implement specific chunking strategies while relying
    on the shared utilities provided here.

    Attributes:
        chunk_resolution (ChunkResolution):
            Configuration defining how input data should be segmented into chunks.

    Methods:
        _build_metadata(...):
            Constructs a standardized metadata dictionary for an uploaded chunk,
            including source information, token counts, and generated metadata.
    """

    def __init__(
        self,
        chunk_resolution: ChunkResolution,
    ):
        self.chunk_resolution = chunk_resolution

        logger.info("Initialised %s", self.__class__.__name__)

    def _build_metadata(
        self,
        *,
        index: int,
        s3_key: str,
        page_number: int,
        created_datetime,
        text: str,
        generated_metadata,
    ) -> dict:
        return UploadedFileMetadata(
            index=index,
            uri=s3_key,
            page_number=page_number,
            created_datetime=created_datetime,
            token_count=tokeniser(text),
            chunk_resolution=self.chunk_resolution,
            name=generated_metadata.name,
            description=generated_metadata.description,
            keywords=generated_metadata.keywords,
        ).model_dump()


class BaseDocumentChunker(BaseChunker):
    """
    Abstract base class for document-based chunking implementations.

    Extends BaseChunker with additional configuration and contracts required
    for chunking unstructured or semi-structured documents into smaller
    retrievable units.

    This class enforces constraints on chunk sizing and overlap, ensuring
    consistency across all document chunking strategies. Subclasses are
    responsible for implementing the actual chunking logic via the `chunks`
    method.

    Attributes:
        chunk_resolution (ChunkResolution):
            Defines the granularity or strategy used for chunking.
        min_chunk_size (int):
            Minimum allowed size of a chunk (must be > 0).
        max_chunk_size (int):
            Maximum allowed size of a chunk (must be >= min_chunk_size).
        overlap_chars (int):
            Number of overlapping characters between consecutive chunks
            (must be >= 0).

    Methods:
        _chunk(*args, **kwargs):
            Optional helper hook for chunking strategies that naturally
            operate on intermediate structures (e.g., pages, sections).
            Not required for all implementations.

        chunks(s3_key, data, generated_metadata):
            Abstract method that converts input document data into an
            iterator of Document chunks. Must be implemented by subclasses.
    """

    def __init__(
        self,
        chunk_resolution: ChunkResolution,
        min_chunk_size: int,
        max_chunk_size: int,
        overlap_chars: int,
    ):
        super().__init__(chunk_resolution)

        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be > 0")
        if max_chunk_size < min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be >= 0")

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_chars = overlap_chars

        logger.info("Initialised %s", self.__class__.__name__)

    # optional helper hook
    def _chunk(self, *args, **kwargs):
        """
        Optional helper for chunking strategies that fit (pages, etc).
        Not required for all implementations.
        """
        raise NotImplementedError

    # required
    @abstractmethod
    def chunks(
        self,
        s3_key: str,
        data,
        generated_metadata: GeneratedMetadata,
    ) -> Iterator[Document]:
        """
        Convert input data into Document chunks.
        """
        raise NotImplementedError
