import logging
from typing import Iterator
from datetime import UTC, datetime
# from copy import deepcopy

from langchain_core.documents import Document

from redbox.models.chain import GeneratedMetadata
from redbox.loader.chunking.base import BaseChunker
from redbox.transform import bedrock_tokeniser
from redbox.models.settings import get_settings

logger = logging.getLogger(__name__)

tokeniser = bedrock_tokeniser

env = get_settings()


class TabularDocumentChunker(BaseChunker):
    """
    Chunker that loads pre-processed tabular elements
    """

    MAX_CHARS = env.embedding_max_chars
    MAX_TOKENS = (
        env.embedding_max_tokens - 2_500
    )  # subtracted 2500 tokens from max for breathing room on tokeniser inconsistencies

    def _split_table(self, text: str) -> list[str]:
        """Split a tabular document into chunks whilst preserving the header."""

        if len(text) <= self.MAX_CHARS and tokeniser(text) <= self.MAX_TOKENS:
            return [text]

        lines = text.splitlines(keepends=True)
        if not lines:
            return []

        header = lines[0]
        chunks: list[str] = []

        current = header

        for row in lines[1:]:
            candidate = current + row

            # Candidate still fits within embedding limits.
            if len(candidate) <= self.MAX_CHARS and tokeniser(candidate) <= self.MAX_TOKENS:
                current = candidate
                continue

            # Flush the current chunk if it contains data rows.
            if current != header:
                chunks.append(current)

            # Start a new chunk with the header repeated.
            current = header + row

            # A single row is too large to embed.
            if len(current) > self.MAX_CHARS or tokeniser(current) > self.MAX_TOKENS:
                raise ValueError(
                    "A single table row exceeds the maximum supported embedding size "
                    f"({len(current)} chars, {tokeniser(current)} tokens)."
                )

        # Append the final chunk if it contains data rows.
        if current != header:
            chunks.append(current)

        return chunks

    def tabular_chunks(
        self,
        s3_key: str,
        tabular_elements: list[dict[str, str]],
        generated_metadata: GeneratedMetadata,
        include_schema_metadata: bool,
    ) -> Iterator[Document]:
        created_datetime = datetime.now(UTC)

        for el in tabular_elements or []:
            metadata = self._build_metadata(
                index=0,
                s3_key=s3_key,
                page_number=1,
                created_datetime=created_datetime,
                text=el["text"],
                generated_metadata=generated_metadata,
            )

            if include_schema_metadata:
                metadata["document_schema"] = el.get("metadata", {}).get("document_schema", {})

            chunks = self._split_table(el["text"])
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()

                chunk_metadata["index"] = chunk_idx
                chunk_metadata["token_count"] = tokeniser(chunk)

                yield Document(
                    page_content=chunk,
                    metadata=chunk_metadata,
                )
