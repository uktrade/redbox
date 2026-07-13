import logging
from datetime import UTC, datetime

from langchain_core.documents import Document

from redbox.models.chain import GeneratedMetadata
from redbox.models.settings import get_settings
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element

from redbox.loader.chunking.base import BaseDocumentChunker


logger = logging.getLogger(__name__)

env = get_settings()


class UnstructuredDocumentChunker(BaseDocumentChunker):
    """
    Chunker that leverages unstructured Element objects from extraction and uses
    unstructured chunk_by_title for layout-aware chunking.
    """

    def _chunk(self, elements: list[Element]):
        if not elements:
            return

        chunks = chunk_by_title(
            elements=elements,
            max_characters=self.max_chunk_size,
            new_after_n_chars=self.max_chunk_size,
            overlap=self.overlap_chars,
            multipage_sections=True,
            overlap_all=env.document_chunking_unstructured_overlap_all,
            include_orig_elements=True,
        )

        for chunk in chunks:
            text = getattr(chunk, "text", "").strip()
            if not text:
                continue

            pages = [e.metadata.page_number for e in chunk.metadata.orig_elements]
            yield text, pages[0]

    def chunks(self, s3_key: str, elements: list[Element], generated_metadata: GeneratedMetadata):
        created_datetime = datetime.now(UTC)

        for idx, (text, page_num) in enumerate(self._chunk(elements)):
            metadata = self._build_metadata(
                index=idx,
                s3_key=s3_key,
                page_number=page_num,
                created_datetime=created_datetime,
                text=text,
                generated_metadata=generated_metadata,
            )

            yield Document(page_content=text, metadata=metadata)
