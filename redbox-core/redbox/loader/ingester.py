import time
import logging
from typing import TYPE_CHECKING

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.runnables import RunnableParallel

from redbox.chains.components import get_embeddings
from redbox.chains.ingest import ingest_from_loader
from redbox.loader.loaders import MetadataLoader, UnstructuredChunkLoader
from redbox.models.chain import GeneratedMetadata
from redbox.models.file import ChunkResolution
from redbox.models.settings import get_settings

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

env = get_settings()
alias = env.elastic_chunk_alias


def get_elasticsearch_store(es, es_index_name: str):
    return OpenSearchVectorSearch(
        index_name=es_index_name,
        opensearch_url=env.elastic.collection_endpoint,
        embedding_function=get_embeddings(env),
        query_field="text",
        vector_query_field=env.embedding_document_field_name,
    )


def get_elasticsearch_store_without_embeddings(es, es_index_name: str):
    return OpenSearchVectorSearch(
        index_name=es_index_name,
        opensearch_url=env.elastic.collection_endpoint,
        embedding_function=FakeEmbeddings(size=env.embedding_backend_vector_size),
    )


def create_alias(alias: str):
    es = env.elasticsearch_client()
    chunk_index_name = alias[:-8]
    es.indices.create(index=chunk_index_name, body=env.index_mapping, ignore=400)
    es.indices.put_alias(index=chunk_index_name, name=alias)


def _ingest_file(file_name: str, es_index_name: str = alias, enable_metadata_extraction=env.enable_metadata_extraction):
    start_time = time.time()
    logging.info("Ingesting file: %s", file_name)

    es = env.elasticsearch_client()

    # Check and create alias/index
    alias_start = time.time()
    if es_index_name == alias:
        if not es.indices.exists_alias(name=alias):
            logging.info("Creating alias %s", alias)
            create_alias(alias)
    else:
        es.indices.create(index=es_index_name, body=env.index_mapping, ignore=400)
    alias_end = time.time()
    logging.info(f"Alias/index creation took {alias_end - alias_start:.2f} seconds")

    # Download file once from S3 using MetadataLoader
    s3_start = time.time()
    metadata_loader = MetadataLoader(env=env, s3_client=env.s3_client(), file_name=file_name)
    file_bytes_io = metadata_loader._get_file_bytes(s3_client=env.s3_client(), file_name=file_name)
    s3_end = time.time()
    logging.info(f"S3 file download (via MetadataLoader) took {s3_end - s3_start:.2f} seconds")

    # Extract metadata using shared chunks
    metadata_start = time.time()
    if enable_metadata_extraction:
        temp_loader = UnstructuredChunkLoader(
            chunk_resolution=ChunkResolution.normal,
            env=env,
            min_chunk_size=env.worker_ingest_min_chunk_size,
            max_chunk_size=env.worker_ingest_max_chunk_size,
            overlap_chars=0,
            metadata=GeneratedMetadata(name=file_name),
        )
        try:
            chunks = temp_loader.lazy_load(file_name, file_bytes_io, return_chunks=True)
            logger.info(
                f"_ingest_file: Chunks type={type(chunks)}, length={len(chunks) if isinstance(chunks, list) else 'unknown'}"
            )
        except Exception as e:
            logger.warning(f"Failed to extract chunks for metadata for file {file_name}: {str(e)}")
            chunks = []
        metadata = metadata_loader.extract_metadata(chunks=chunks)
    else:
        metadata = GeneratedMetadata(name=file_name, description="", keywords=[])
    metadata_end = time.time()
    logging.info(f"Metadata extraction took {metadata_end - metadata_start:.2f} seconds")

    # Normal chunk
    chunk_start = time.time()
    normal_loader = UnstructuredChunkLoader(
        chunk_resolution=ChunkResolution.normal,
        env=env,
        min_chunk_size=env.worker_ingest_min_chunk_size,
        max_chunk_size=env.worker_ingest_max_chunk_size,
        overlap_chars=0,
        metadata=metadata,
    )
    chunk_ingest_chain = ingest_from_loader(
        loader=normal_loader,
        s3_client=env.s3_client(),
        vectorstore=get_elasticsearch_store(es, es_index_name),
        env=env,
    )
    chunk_end = time.time()
    logging.info(f"Normal chunk chain setup took {chunk_end - chunk_start:.2f} seconds")

    # Large chunk
    large_chunk_start = time.time()
    large_loader = UnstructuredChunkLoader(
        chunk_resolution=ChunkResolution.largest,
        env=env,
        min_chunk_size=env.worker_ingest_largest_chunk_size,
        max_chunk_size=env.worker_ingest_largest_chunk_size,
        overlap_chars=env.worker_ingest_largest_chunk_overlap,
        metadata=metadata,
    )
    large_chunk_ingest_chain = ingest_from_loader(
        loader=large_loader,
        s3_client=env.s3_client(),
        vectorstore=get_elasticsearch_store_without_embeddings(es, es_index_name),
        env=env,
    )
    large_chunk_end = time.time()
    logging.info(f"Large chunk chain setup took {large_chunk_end - large_chunk_start:.2f} seconds")

    # Parallel
    parallel_start = time.time()
    new_ids = {"normal": [], "largest": []}
    try:
        normal_chunks = list(normal_loader.lazy_load(file_name, file_bytes_io))
        large_chunks = list(large_loader.lazy_load(file_name, file_bytes_io))
        logger.info(f"_ingest_file: Normal chunks length={len(normal_chunks)}, Large chunks length={len(large_chunks)}")
        if not normal_chunks and not large_chunks:
            logger.warning(f"No chunks to ingest for file: {file_name}, skipping OpenSearch ingestion")
        else:
            new_ids = RunnableParallel({"normal": chunk_ingest_chain, "largest": large_chunk_ingest_chain}).invoke(
                file_name
            )
    except Exception as e:
        logger.error(f"Failed to ingest chunks for file {file_name}: {str(e)}")
        raise
    parallel_end = time.time()
    logging.info(f"Parallel chunking and ingestion took {parallel_end - parallel_start:.2f} seconds")
    logging.info(
        "File: %s %s chunks ingested",
        file_name,
        {k: len(v) for k, v in new_ids.items()},
    )

    total_time = time.time() - start_time
    logging.info(f"Total ingest_file time: {total_time:.2f} seconds")


def ingest_file(file_name: str, es_index_name: str = alias) -> str | None:
    try:
        _ingest_file(file_name, es_index_name)
    except Exception as e:
        logger.exception("Error while processing file [%s]", file_name)
        return f"{type(e)}: {e.args[0]}"
