import logging
from datetime import UTC, datetime

from langchain_core.documents import Document

from redbox.models.chain import GeneratedMetadata
from redbox.loader.chunking.base import BaseDocumentChunker

logger = logging.getLogger(__name__)


class PageByPageDocumentChunker(BaseDocumentChunker):
    """
    Chunker that performs sliding-window chunking over each
    page in the document (chunks do not overlap over pages).
    """

    def _chunk(self, pages: list[str]):
        for page_num, page_text in enumerate(pages, start=1):
            yield from self._chunk_text(page_text, page_num)

    def _chunk_text(self, text: str, page_num: int):
        if not text.strip():
            return

        advance = self.max_chunk_size - self.overlap_chars
        if advance <= 0:
            raise ValueError("overlap must be < max_chunk_size")

        start = 0
        length = len(text)

        while start < length:
            chunk = text[start : start + self.max_chunk_size]
            yield chunk, page_num

            start += advance

    def chunks(self, s3_key: str, pages: list[str], generated_metadata: GeneratedMetadata):
        created_datetime = datetime.now(UTC)

        for idx, (text, page_num) in enumerate(self._chunk(pages)):
            metadata = self._build_metadata(
                index=idx,
                s3_key=s3_key,
                page_number=page_num,
                created_datetime=created_datetime,
                text=text,
                generated_metadata=generated_metadata,
            )

            yield Document(page_content=text, metadata=metadata)
