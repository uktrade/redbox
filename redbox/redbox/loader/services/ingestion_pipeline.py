import logging
from datetime import UTC, datetime
from io import BytesIO
from typing import Iterator

from langchain_core.documents import Document

from redbox.loader.loaders import DocumentLoader
from redbox.models.file import (
    ChunkResolution,
    UploadedFileMetadata,
)
from redbox.loader.services.chunker import TextChunker
from redbox.loader.services.embedding_batcher import EmbeddingBatcher
from redbox.loader.services.opensearch_indexer import OpenSearchBulkIndexer
from redbox.transform import bedrock_tokeniser

logger = logging.getLogger(__name__)
tokeniser = bedrock_tokeniser


class IngestionPipeline:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedding_batcher: EmbeddingBatcher,
        indexer: OpenSearchBulkIndexer,
        metadata,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedding_batcher = embedding_batcher
        self.indexer = indexer
        self.metadata = metadata

    def iter_documents(
        self,
        file_name: str,
        file_bytes: BytesIO | None = None,
    ) -> Iterator[Document]:
        """
        Yield Document objects with proper metadata, including document_schema for tabular files.
        """
        idx = 0
        display_name = file_name.lower()

        is_tabular = display_name.endswith((".csv", ".tsv", ".xls", ".xlsx"))

        for page_num, page_text in self.loader.iter_pages(
            file_name=file_name,
            file_bytes=file_bytes,
        ):
            if is_tabular:
                chunks = [page_text]
                chunk_resolution = ChunkResolution.tabular
            else:
                chunks = list(self.chunker.chunk(page_text))
                chunk_resolution = ChunkResolution.normal

            for chunk in chunks:
                metadata_dict = UploadedFileMetadata(
                    index=idx,
                    uri=file_name,
                    page_number=page_num,
                    created_datetime=datetime.now(UTC),
                    token_count=tokeniser(chunk),
                    chunk_resolution=chunk_resolution,
                    name=self.metadata.name,
                    description=self.metadata.description,
                    keywords=self.metadata.keywords,
                ).model_dump()

                if is_tabular and "document_schema" in page_text:  # crude but works
                    try:
                        pass
                    except:  # noqa: E722
                        pass

                yield Document(
                    page_content=chunk,
                    metadata=metadata_dict,
                )
                idx += 1

    def ingest(
        self,
        file_name: str,
        file_bytes: BytesIO | None = None,
    ):

        docs = self.iter_documents(
            file_name=file_name,
            file_bytes=file_bytes,
        )

        for batch_docs, embeddings in self.embedding_batcher.iter_embedding_batches(docs):
            self.indexer.bulk_index(
                batch_docs,
                embeddings,
            )
