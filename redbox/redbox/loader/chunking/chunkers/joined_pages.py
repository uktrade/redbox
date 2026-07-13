import logging
from datetime import UTC, datetime

from langchain_core.documents import Document

from redbox.models.chain import GeneratedMetadata
from redbox.loader.chunking.base import BaseDocumentChunker

logger = logging.getLogger(__name__)


class JoinedPagesDocumentChunker(BaseDocumentChunker):
    """
    Chunker that joins all pages first, then performs sliding-window chunking
    over the full document instead of page-by-page chunking.
    """

    def _join_pages(self, pages: list[str]):
        """
        Returns:
            full_text: str
            page_spans: list of (start_offset, end_offset, page_num)
        """
        full_text_parts = []
        page_spans = []

        offset = 0

        for page_num, page_text in enumerate(pages, start=1):
            page_text = page_text or ""
            start = offset

            full_text_parts.append(page_text)
            offset += len(page_text)

            end = offset
            page_spans.append((start, end, page_num))

            # add a separator between pages (accounts for newline in offsets)
            full_text_parts.append("\n")
            offset += 1

        full_text = "".join(full_text_parts)
        return full_text, page_spans

    def _get_page_for_chunk(self, start: int, end: int, page_spans: list[tuple[int, int, int]]):
        """
        Find page range covered by a chunk.
        """
        pages = [page_num for span_start, span_end, page_num in page_spans if span_end > start and span_start < end]

        return pages or [1]

    def _chunk(self, pages: list[str]):
        full_text, page_spans = self._join_pages(pages)

        if not full_text.strip():
            return

        advance = self.max_chunk_size - self.overlap_chars
        if advance <= 0:
            raise ValueError("overlap must be < max_chunk_size")

        start = 0
        length = len(full_text)

        while start < length:
            end = start + self.max_chunk_size
            chunk = full_text[start:end]

            yield chunk, start, end, page_spans

            start += advance

    def chunks(self, s3_key: str, pages: list[str], generated_metadata: GeneratedMetadata):
        created_datetime = datetime.now(UTC)

        for idx, (text, start, end, page_spans) in enumerate(self._chunk(pages)):
            page_nums = self._get_page_for_chunk(start, end, page_spans)
            page_number = min(page_nums)  # keep metadata compatible (single int)

            metadata = self._build_metadata(
                index=idx,
                s3_key=s3_key,
                page_number=page_number,
                created_datetime=created_datetime,
                text=text,
                generated_metadata=generated_metadata,
            )

            yield Document(page_content=text, metadata=metadata)
