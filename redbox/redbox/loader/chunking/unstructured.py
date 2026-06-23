import logging
from typing import Iterator

from datetime import UTC, datetime

from langchain_core.documents import Document

from redbox.models.settings import get_settings
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element

from redbox.models.chain import GeneratedMetadata
from redbox.models.file import UploadedFileMetadata, ChunkResolution
from redbox.transform import bedrock_tokeniser


logger = logging.getLogger(__name__)

tokeniser = bedrock_tokeniser

env = get_settings()


class UnstructuredDocumentChunker:
    def __init__(
        self,
        chunk_resolution: ChunkResolution,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap_chars: int = 200,
        include_schema_metadata: bool = False,
    ):
        if min_chunk_size <= 0:
            raise ValueError(f"{__name__} - min_chunk_size ({min_chunk_size}) must be > 0")

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
            "Initialised UnstructuredDocumentChunker "
            "(chunk_resolution=%s, min_chunk_size=%s, "
            "max_chunk_size=%s, overlap_chars=%s, "
            "include_schema_metadata=%s)",
            chunk_resolution,
            min_chunk_size,
            max_chunk_size,
            overlap_chars,
            include_schema_metadata,
        )

    def _chunk_elements(self, elements: list[Element]) -> list[tuple[str, list[int]]]:
        if not elements:
            return []

        chunks = chunk_by_title(
            elements=elements,
            max_characters=self.max_chunk_size,
            new_after_n_chars=self.max_chunk_size,
            overlap=self.overlap_chars,
            multipage_sections=True,
            overlap_all=env.unstructured_chunking_overlap_all,
            include_orig_elements=True,
        )

        return [
            (chunk.text, [e.metadata.page_number for e in chunk.metadata.orig_elements])
            for chunk in chunks
            if getattr(chunk, "text", "").strip()
        ]

    def chunks(
        self,
        s3_key: str,
        elements: list[Element],
        generated_metadata: GeneratedMetadata,
    ) -> Iterator[Document]:
        created_datetime = datetime.now(UTC)

        for el in elements:
            page = getattr(el.metadata, "page_number", None)
            if page in (2, 3, 4):  # wherever your TOC actually sits
                logger.warning(page, el.category, repr(str(el))[:80])

        chunks = self._chunk_elements(elements)

        for idx, chunk in enumerate(chunks):
            chunk_text, pages = chunk

            metadata = UploadedFileMetadata(
                index=idx,
                uri=s3_key,
                page_number=pages[0],
                created_datetime=created_datetime,
                token_count=tokeniser(chunk_text),
                chunk_resolution=self.chunk_resolution,
                name=generated_metadata.name,
                description=generated_metadata.description,
                keywords=generated_metadata.keywords,
            ).model_dump()

            yield Document(
                page_content=chunk_text,
                metadata=metadata,
            )
