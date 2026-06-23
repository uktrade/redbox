import logging
from functools import partial
from typing import Iterator

from langchain.vectorstores import VectorStore
from langchain_core.documents.base import Document
from langchain_core.runnables import Runnable, RunnableLambda, chain
from unstructured.documents.elements import Element

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
) -> Runnable:
    @chain
    def wrapped(file_name: str) -> Iterator[Document]:
        try:
            log.info("wrapped START: %s", file_name)

            docs = list(
                chunker.chunks(
                    s3_key=file_name,
                    elements=elements,
                    generated_metadata=metadata,
                )
            )

            if not docs:
                raise ValueError(f"No content extracted from {file_name}")

            log.info("Extracted %d documents", len(docs))
            return docs

        except Exception:
            log.exception("wrapped() crashed for %s", file_name)
            raise

    return wrapped


def chunk_loader_tabular(
    chunker: DocumentChunkingService,
    tabular_elements: list[dict[str, str]],
    metadata: GeneratedMetadata,
) -> Runnable:
    @chain
    def wrapped(file_name: str) -> Iterator[Document]:
        try:
            log.info("wrapped START: %s", file_name)

            docs = list(
                chunker.tabular_chunks(
                    s3_key=file_name,
                    tabular_elements=tabular_elements,
                    generated_metadata=metadata,
                )
            )

            if not docs:
                raise ValueError(f"No content extracted from {file_name}")

            log.info("Extracted %d documents", len(docs))
            return docs

        except Exception:
            log.exception("wrapped() crashed for %s", file_name)
            raise

    return wrapped


def ingest_chunks(
    chunker: DocumentChunkingService,
    vectorstore: VectorStore,
    elements: list[str] | list[Element],
    metadata: GeneratedMetadata,
) -> Runnable:
    return (
        chunk_loader(
            chunker=chunker,
            elements=elements,
            metadata=metadata,
        )
        | RunnableLambda(list)
        | log_chunks
        | RunnableLambda(
            partial(
                vectorstore.add_documents,
                create_index_if_not_exists=False,
            )
        )
    )


def ingest_tabular_chunks(
    chunker: DocumentChunkingService,
    vectorstore: VectorStore,
    tabular_elements: list[dict[str, str]],
    metadata: GeneratedMetadata,
) -> Runnable:
    return (
        chunk_loader_tabular(
            chunker=chunker,
            tabular_elements=tabular_elements,
            metadata=metadata,
        )
        | RunnableLambda(list)
        | log_chunks
        | RunnableLambda(
            partial(
                vectorstore.add_documents,
                create_index_if_not_exists=False,
            )
        )
    )
