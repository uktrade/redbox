import logging
from functools import partial
from typing import Iterator

from langchain.vectorstores import VectorStore
from langchain_core.documents.base import Document
from langchain_core.runnables import Runnable, RunnableLambda, chain, RunnableParallel
from unstructured.documents.elements import Element

from redbox_app.redbox_core.models import File

from redbox.loader.chunking.service import DocumentChunkingService
from redbox.models.chain import GeneratedMetadata


logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


@chain
def log_chunks(chunks: list[Document]):
    log.info("Processing %s chunks", len(chunks))
    return chunks


def chunk_loader(
    chunker: DocumentChunkingService,
    elements: list[str] | list[Element],
    metadata: GeneratedMetadata,
    chunks_overlap_pages: bool,
) -> Runnable:
    @chain
    def wrapped(file_name: str) -> tuple[File.IngestChunkingStrategy, Iterator[Document]]:
        try:
            log.info("wrapped START: %s", file_name)

            strategy, raw_docs = chunker.chunks(
                s3_key=file_name,
                elements=elements,
                generated_metadata=metadata,
                chunks_overlap_pages=chunks_overlap_pages,
            )

            docs = list(raw_docs)
            if not docs:
                raise ValueError(f"No content extracted from {file_name}")

            log.info("Extracted %d documents with strategy %s", len(docs), strategy)
            return strategy, docs

        except Exception:
            log.exception("wrapped() crashed for %s", file_name)
            raise

    return wrapped


def chunk_loader_tabular(
    chunker: DocumentChunkingService,
    tabular_elements: list[dict[str, str]],
    metadata: GeneratedMetadata,
    include_schema_metadata: bool,
) -> Runnable:
    @chain
    def wrapped(file_name: str) -> tuple[File.IngestChunkingStrategy, Iterator[Document]]:
        try:
            log.info("wrapped START: %s", file_name)

            strategy, raw_docs = chunker.tabular_chunks(
                s3_key=file_name,
                tabular_elements=tabular_elements,
                generated_metadata=metadata,
                include_schema_metadata=include_schema_metadata,
            )

            docs = list(raw_docs)
            if not docs:
                raise ValueError(f"No content extracted from {file_name}")

            log.info("Extracted %d documents", len(docs))
            return strategy, docs

        except Exception:
            log.exception("wrapped() crashed for %s", file_name)
            raise

    return wrapped


def ingest_chunks(
    chunker: DocumentChunkingService,
    vectorstore: VectorStore,
    elements: list[str] | list[Element],
    metadata: GeneratedMetadata,
    chunks_overlap_pages: bool = False,
) -> Runnable:
    loader = chunk_loader(
        chunker=chunker,
        elements=elements,
        metadata=metadata,
        chunks_overlap_pages=chunks_overlap_pages,
    )

    ingest_branch = (
        RunnableLambda(lambda docs: list(docs))
        | log_chunks
        | RunnableLambda(
            partial(
                vectorstore.add_documents,
                create_index_if_not_exists=False,
            )
        )
    )

    return loader | RunnableParallel(
        documents=RunnableLambda(lambda x: x[1]) | ingest_branch,
        strategy=RunnableLambda(lambda x: x[0]),
    )


def ingest_tabular_chunks(
    chunker: DocumentChunkingService,
    vectorstore: VectorStore,
    tabular_elements: list[dict[str, str]],
    metadata: GeneratedMetadata,
    include_schema_metadata: bool = False,
) -> Runnable:
    loader = chunk_loader_tabular(
        chunker=chunker,
        tabular_elements=tabular_elements,
        metadata=metadata,
        include_schema_metadata=include_schema_metadata,
    )

    ingest_branch = (
        RunnableLambda(lambda docs: list(docs))
        | log_chunks
        | RunnableLambda(
            partial(
                vectorstore.add_documents,
                create_index_if_not_exists=False,
            )
        )
    )

    return loader | RunnableParallel(
        documents=RunnableLambda(lambda x: x[1]) | ingest_branch,
        strategy=RunnableLambda(lambda x: x[0]),
    )
