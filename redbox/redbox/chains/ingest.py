import logging
from functools import partial
from typing import Iterator

from langchain_core.documents.base import Document
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, chain
from langchain_core.vectorstores import VectorStore
from redbox_app.redbox_core.enums import IngestChunkingStrategy
from unstructured.documents.elements import Element

from redbox.loader.chunking.service import DocumentChunkingService
from redbox.models.chain import GeneratedMetadata
from redbox.models.file import ChunkResolution

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


@chain
def log_chunks(chunks: list[Document]):
    log.info("Processing %s chunks", len(chunks))

    for i, doc in enumerate(chunks):
        log.info(
            "chunk %d length=%d",
            i,
            len(doc.page_content),
        )

    return chunks


def _delete_existing_chunks(
    vectorstore: VectorStore,
    uri: str,
    chunk_resolution: ChunkResolution,
) -> None:
    """Delete previously ingested chunks for this file and chunk resolution.

    Prevents duplicate chunks from accumulating when a file is reingested.
    Matches on both metadata.uri.keyword and
    metadata.chunk_resolution.keyword.
    """
    es_client = vectorstore.client
    index_name = vectorstore.index_name

    try:
        response = es_client.delete_by_query(
            index=index_name,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"metadata.uri.keyword": uri}},
                            {"term": {"metadata.chunk_resolution.keyword": chunk_resolution.value}},
                        ]
                    }
                }
            },
            conflicts="proceed",
            refresh=False,
        )
        log.warning(
            "Deleted %s existing chunks for uri=%s, chunk_resolution=%s from index=%s",
            response.get("deleted", 0),
            uri,
            chunk_resolution,
            index_name,
        )
    except Exception:
        log.exception(
            "Failed to delete existing chunks for uri=%s, chunk_resolution=%s from index=%s",
            uri,
            chunk_resolution,
            index_name,
        )
        raise


def chunk_loader(
    chunker: DocumentChunkingService,
    vectorstore: VectorStore,
    elements: list[str] | list[Element] | list[dict[str, str]],
    metadata: GeneratedMetadata,
    chunks_overlap_pages: bool,
) -> Runnable:
    @chain
    def wrapped(file_name: str) -> tuple[IngestChunkingStrategy, Iterator[Document]]:
        try:
            log.info("wrapped START: %s", file_name)

            _delete_existing_chunks(vectorstore, file_name, chunker.chunk_resolution)

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
    vectorstore: VectorStore,
    tabular_elements: list[dict[str, str]],
    metadata: GeneratedMetadata,
    include_schema_metadata: bool,
) -> Runnable:
    @chain
    def wrapped(file_name: str) -> tuple[IngestChunkingStrategy, Iterator[Document]]:
        try:
            log.info("wrapped START: %s", file_name)

            _delete_existing_chunks(vectorstore, file_name, chunker.chunk_resolution)

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
    elements: list[str] | list[Element] | list[dict[str, str]],
    metadata: GeneratedMetadata,
    chunks_overlap_pages: bool = False,
) -> Runnable:
    if elements and isinstance(elements[0], dict):
        return ingest_tabular_chunks(
            chunker=chunker,
            vectorstore=vectorstore,
            tabular_elements=elements,
            metadata=metadata,
        )

    loader = chunk_loader(
        chunker=chunker,
        vectorstore=vectorstore,
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
        vectorstore=vectorstore,
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
