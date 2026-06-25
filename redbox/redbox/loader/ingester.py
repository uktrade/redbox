import logging
import time
import os
import traceback
from typing import TYPE_CHECKING
from pydantic import BaseModel

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.runnables import RunnableParallel

from redbox_app.redbox_core.models import File
from redbox.chains.components import get_embeddings
from redbox.models.file import ChunkResolution
from redbox.models.settings import get_settings

from redbox.loader.extraction.service import DocumentExtractionService
from redbox.loader.extraction.metadata import MetadataExtraction
from redbox.chains.ingest import ingest_chunks, ingest_tabular_chunks, DocumentChunkingService


if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

env = get_settings()
alias = env.elastic_chunk_alias


def get_elasticsearch_store(es, es_index_name: str):
    log.info("Creating OpenSearchVectorSearch for index %s against %s", es_index_name, env.elastic.collection_endpoint)
    return OpenSearchVectorSearch(
        index_name=es_index_name,
        opensearch_url=env.elastic.collection_endpoint,
        embedding_function=get_embeddings(env),
        query_field="text",
        vector_query_field=env.embedding_document_field_name,
        bulk_size=3000,
    )


def get_elasticsearch_store_without_embeddings(es, es_index_name: str):
    log.info(
        "Creating OpenSearchVectorSearch (no embeddings) for index %s against %s",
        es_index_name,
        env.elastic.collection_endpoint,
    )
    return OpenSearchVectorSearch(
        index_name=es_index_name,
        opensearch_url=env.elastic.collection_endpoint,
        embedding_function=FakeEmbeddings(size=env.embedding_backend_vector_size),
        bulk_size=3000,
    )


def create_alias(alias: str):
    es = env.elasticsearch_client()

    chunk_index_name = alias[:-8]  # removes -current

    es.indices.create(index=chunk_index_name, body=env.index_mapping, ignore=400)
    es.indices.put_alias(index=chunk_index_name, name=alias)


class FileIngestionResponse(BaseModel):
    normal_extraction_strategy: File.IngestExtractionStrategy
    normal_chunking_strategy: File.IngestChunkingStrategy
    largest_extraction_strategy: File.IngestExtractionStrategy
    largest_chunking_strategy: File.IngestChunkingStrategy


def _ingest_file(
    file_name: str, es_index_name: str = alias, enable_metadata_extraction=env.enable_metadata_extraction
) -> FileIngestionResponse:
    logging.warning("Ingesting file: %s", file_name)
    start_time = time.time()

    es = env.elasticsearch_client()
    log.info("Using Elasticsearch client: %s", es)

    if es_index_name == alias:
        if not es.indices.exists_alias(name=alias):
            log.info("Alias %s does not exist; creating", alias)
            create_alias(alias)
    else:
        if es_index_name == env.elastic_schematised_chunk_index:
            log.info("Creating schematised index %s", env.elastic_schematised_chunk_index)
            es.indices.create(index=env.elastic_schematised_chunk_index, body=env.index_mapping_schematised, ignore=400)
        else:
            log.info("Creating index %s", es_index_name)
            es.indices.create(index=es_index_name, body=env.index_mapping, ignore=400)

    extraction_service = DocumentExtractionService(
        bucket=env.bucket_name,
    )

    # -- extraction --
    # normal chunk
    normal_extraction_strategy, normal_elements = extraction_service.extract(
        file_name=file_name, chunk_resolution=ChunkResolution.normal
    )

    # largest chunk
    largest_elements = normal_elements
    if os.path.basename(file_name).lower().endswith(".pdf"):  # if PDF - extract largest chunks with direct extraction
        largest_extraction_strategy, largest_elements = extraction_service.extract(
            file_name=file_name, chunk_resolution=ChunkResolution.largest
        )

    # metadata
    metadata = MetadataExtraction(env=env).extract(file_name=file_name, elements=normal_elements)

    # -- chunking and opensearch ingestion --
    # normal chunk
    chunk_ingest_chain = ingest_chunks(
        chunker=DocumentChunkingService(
            chunk_resolution=ChunkResolution.normal,
            min_chunk_size=env.worker_ingest_min_chunk_size,
            max_chunk_size=env.worker_ingest_max_chunk_size,
            overlap_chars=0,
        ),
        vectorstore=get_elasticsearch_store(es, es_index_name),
        elements=normal_elements,
        metadata=metadata,
    )

    # largest chunk
    large_chunk_ingest_chain = ingest_chunks(
        chunker=DocumentChunkingService(
            chunk_resolution=ChunkResolution.largest,
            min_chunk_size=env.worker_ingest_largest_chunk_size,
            max_chunk_size=env.worker_ingest_largest_chunk_size,
            overlap_chars=env.worker_ingest_largest_chunk_overlap,
        ),
        vectorstore=get_elasticsearch_store_without_embeddings(es, es_index_name),
        elements=largest_elements,
        metadata=metadata,
        chunks_overlap_pages=True,
    )

    # tabular chunk
    tabular_chunk_ingest_chain = ingest_tabular_chunks(
        chunker=DocumentChunkingService(
            chunk_resolution=ChunkResolution.tabular,
            min_chunk_size=env.worker_ingest_largest_chunk_size,
            max_chunk_size=env.worker_ingest_largest_chunk_size,
            overlap_chars=env.worker_ingest_largest_chunk_overlap,
        ),
        vectorstore=get_elasticsearch_store_without_embeddings(es, es_index_name),
        tabular_elements=normal_elements,
        metadata=metadata,
    )

    # tabular chunk (with metadata.document_schema in schematised index)
    tabular_schema_chunk_ingest_chain = ingest_tabular_chunks(
        chunker=DocumentChunkingService(
            chunk_resolution=ChunkResolution.tabular,
            min_chunk_size=env.worker_ingest_largest_chunk_size,
            max_chunk_size=env.worker_ingest_largest_chunk_size,
            overlap_chars=env.worker_ingest_largest_chunk_overlap,
        ),
        vectorstore=get_elasticsearch_store_without_embeddings(es, env.elastic_schematised_chunk_index),
        tabular_elements=normal_elements,
        metadata=metadata,
        include_schema_metadata=True,
    )

    if file_name.endswith((".csv", ".xls", ".xlsx", ".tsv")):
        new_ids = RunnableParallel(
            {
                "normal": chunk_ingest_chain,
                "largest": large_chunk_ingest_chain,
                "tabular": tabular_chunk_ingest_chain,
                "schematised_tabular": tabular_schema_chunk_ingest_chain,
            }
        ).invoke(file_name)  # Run an additional tabular process if a tabular file is ingested
    else:
        new_ids = RunnableParallel({"normal": chunk_ingest_chain, "largest": large_chunk_ingest_chain}).invoke(
            file_name
        )
    logging.info(
        "File: %s %s chunks ingested",
        file_name,
        {k: len(v.get("documents", [])) for k, v in new_ids.items()},
    )
    duration = time.time() - start_time
    logging.info("total ingestion for file [%s] took %.2f seconds", file_name, duration)

    return FileIngestionResponse(
        normal_extraction_strategy=normal_extraction_strategy,
        normal_chunking_strategy=new_ids.get("normal", {}).get("strategy"),
        largest_extraction_strategy=largest_extraction_strategy,
        largest_chunking_strategy=new_ids.get("largest", {}).get("strategy"),
    )


def ingest_file(file_name: str, es_index_name: str = alias) -> tuple[FileIngestionResponse | None, str | None]:
    try:
        response = _ingest_file(file_name, es_index_name)
        return response, None
    except Exception:
        logging.exception("Error while processing file [%s]", file_name)
        return None, traceback.format_exc()
