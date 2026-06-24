from abc import ABC, abstractmethod
from typing import Iterator
from datetime import UTC, datetime
import logging

from langchain_core.documents import Document

from redbox.models.chain import GeneratedMetadata
from redbox.models.file import UploadedFileMetadata, ChunkResolution
from redbox.transform import bedrock_tokeniser

logger = logging.getLogger(__name__)

tokeniser = bedrock_tokeniser


class BaseChunker(ABC):
    def __init__(
        self,
        chunk_resolution: ChunkResolution,
    ):
        self.chunk_resolution = chunk_resolution

        logger.info("Initialised %s", self.__class__.__name__)

    def _now(self):
        return datetime.now(UTC)

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
