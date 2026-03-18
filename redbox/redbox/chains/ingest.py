import logging
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Iterator

from langchain.vectorstores import VectorStore
from langchain_core.documents.base import Document
from langchain_core.runnables import Runnable, RunnableLambda, chain

from redbox.loader.loaders import TextractChunkLoader
from redbox.models.settings import Settings

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


@chain
def log_chunks(chunks: list[Document]):
    log.info("Processing %s chunks", len(chunks))
    return chunks


def document_loader(
    document_loader: TextractChunkLoader,
    s3_client: S3Client,
    env: Settings,
) -> Runnable:
    @chain
    def wrapped(file_name: str) -> Iterator[Document]:
        try:
            log.info("wrapped START: %s", file_name)

            if file_name.lower().endswith(".docx"):
                log.info("Loading DOCX from S3")
                obj = s3_client.get_object(Bucket=env.bucket_name, Key=file_name)
                file_bytes = BytesIO(obj["Body"].read())
            else:
                file_bytes = None

            docs = list(
                document_loader.lazy_load(
                    file_name=file_name,
                    file_bytes=file_bytes,
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


def ingest_from_loader(
    loader: TextractChunkLoader,
    s3_client: S3Client,
    vectorstore: VectorStore,
    env: Settings,
) -> Runnable:
    return (
        document_loader(loader, s3_client, env)
        | RunnableLambda(list)
        | log_chunks
        | RunnableLambda(
            partial(
                vectorstore.add_documents,
                create_index_if_not_exists=False,
            )
        )
    )
