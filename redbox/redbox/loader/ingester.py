import logging
import time
import traceback
from typing import TYPE_CHECKING

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.runnables import RunnableParallel

from redbox.chains.components import get_embeddings
from redbox.chains.ingest import ingest_from_loader
from redbox.loader.loaders import TextractChunkLoader, MetadataLoader
from redbox.models.settings import get_settings

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
        bulk_size=1000,
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
        bulk_size=1000,
    )


def create_alias(alias: str):
    es = env.elasticsearch_client()

    chunk_index_name = alias[:-8]  # removes -current

    es.indices.create(index=chunk_index_name, body=env.index_mapping, ignore=400)
    es.indices.put_alias(index=chunk_index_name, name=alias)


def _ingest_file(file_name: str, es_index_name: str = alias, enable_metadata_extraction=env.enable_metadata_extraction):
    logging.info("Ingesting file: %s", file_name)
    start_time = time.time()

    metadata = MetadataLoader(env, env.s3_client(), file_name).extract_metadata()

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

    chunk_ingest_chain = ingest_from_loader(
        loader=TextractChunkLoader(
            bucket=env.bucket_name,
            min_chunk_size=env.worker_ingest_min_chunk_size,
            max_chunk_size=env.worker_ingest_max_chunk_size,
            overlap_chars=0,
            metadata=metadata,
        ),
        s3_client=env.s3_client(),
        vectorstore=get_elasticsearch_store(es, es_index_name),
        env=env,
    )

    large_chunk_ingest_chain = ingest_from_loader(
        loader=TextractChunkLoader(
            bucket=env.bucket_name,
            min_chunk_size=env.worker_ingest_min_chunk_size,
            max_chunk_size=env.worker_ingest_max_chunk_size,
            overlap_chars=0,
            metadata=metadata,
        ),
        s3_client=env.s3_client(),
        vectorstore=get_elasticsearch_store_without_embeddings(es, es_index_name),
        env=env,
    )

    tabular_chunk_ingest_chain = ingest_from_loader(
        loader=TextractChunkLoader(
            bucket=env.bucket_name,
            min_chunk_size=env.worker_ingest_min_chunk_size,
            max_chunk_size=env.worker_ingest_max_chunk_size,
            overlap_chars=0,
            metadata=metadata,
        ),
        s3_client=env.s3_client(),
        vectorstore=get_elasticsearch_store_without_embeddings(es, es_index_name),
        env=env,
    )

    tabular_schema_chunk_ingest_chain = ingest_from_loader(
        loader=TextractChunkLoader(
            bucket=env.bucket_name,
            min_chunk_size=env.worker_ingest_min_chunk_size,
            max_chunk_size=env.worker_ingest_max_chunk_size,
            overlap_chars=0,
            metadata=metadata,
        ),
        s3_client=env.s3_client(),
        vectorstore=get_elasticsearch_store_without_embeddings(es, env.elastic_schematised_chunk_index),
        env=env,
    )

    if file_name.endswith((".csv", ".xls", "xlsx")):
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
        {k: len(v) for k, v in new_ids.items()},
    )
    duration = time.time() - start_time
    logging.info("total ingestion for file [%s] took %.2f seconds", file_name, duration)


def ingest_file(file_name: str, es_index_name: str = alias) -> str | None:
    try:
        _ingest_file(file_name, es_index_name)
    except Exception:
        logging.exception("Error while processing file [%s]", file_name)
        return traceback.format_exc()
