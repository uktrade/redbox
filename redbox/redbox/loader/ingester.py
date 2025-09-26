import docx
import logging
import time
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
log = logging.getLogger()

env = get_settings()
alias = env.elastic_chunk_alias


def get_elasticsearch_store(es, es_index_name: str):
    return OpenSearchVectorSearch(
        index_name=es_index_name,
        opensearch_url=env.elastic.collection_endpoint,
        embedding_function=get_embeddings(env),
        query_field="text",
        vector_query_field=env.embedding_document_field_name,
        bulk_size=1000,
    )


def get_elasticsearch_store_without_embeddings(es, es_index_name: str):
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
    
    
def has_tables_in_docx(file_name: str, s3_client: S3Client) -> bool:
    try:
        # Download the file to a temporary location
        local_path = "/tmp/" + file_name.split("/")[-1]
        s3_client.download_file(env.bucket_name, file_name, local_path)
        
        # Open and check for tables
        doc = docx.Document(local_path)
        return len(doc.tables) > 0
    except Exception as e:
        logging.warning(f"Failed to check tables in DOCX file {file_name}: {e}")
        return False
    finally:
        # Clean up temporary file
        import os
        try:
            os.remove(local_path)
        except:
            pass


def _ingest_file(file_name: str, es_index_name: str = alias, enable_metadata_extraction=env.enable_metadata_extraction):
    logging.info("Ingesting file: %s", file_name)
    start_time = time.time()

    es = env.elasticsearch_client()

    if es_index_name == alias:
        if not es.indices.exists_alias(name=alias):
            print("The alias does not exist")
            create_alias(alias)
    else:
        es.indices.create(index=es_index_name, body=env.index_mapping, ignore=400)

    # Extract metadata
    if enable_metadata_extraction:
        metadata_loader = MetadataLoader(env=env, s3_client=env.s3_client(), file_name=file_name)
        metadata = metadata_loader.extract_metadata()
    else:
        # return empty metadata
        metadata = GeneratedMetadata(name=file_name, description="", keywords=[])

    chunk_ingest_chain = ingest_from_loader(
        loader=UnstructuredChunkLoader(
            chunk_resolution=ChunkResolution.normal,
            env=env,
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
        loader=UnstructuredChunkLoader(
            chunk_resolution=ChunkResolution.largest,
            env=env,
            min_chunk_size=env.worker_ingest_largest_chunk_size,
            max_chunk_size=env.worker_ingest_largest_chunk_size,
            overlap_chars=env.worker_ingest_largest_chunk_overlap,
            metadata=metadata,
        ),
        s3_client=env.s3_client(),
        vectorstore=get_elasticsearch_store_without_embeddings(es, es_index_name),
        env=env,
    )

    tabular_chunk_ingest_chain = ingest_from_loader(
        loader=UnstructuredChunkLoader(
            chunk_resolution=ChunkResolution.tabular,
            env=env,
            min_chunk_size=env.worker_ingest_largest_chunk_size,
            max_chunk_size=env.worker_ingest_largest_chunk_size,
            overlap_chars=env.worker_ingest_largest_chunk_overlap,
            metadata=metadata,
        ),
        s3_client=env.s3_client(),
        vectorstore=get_elasticsearch_store_without_embeddings(es, es_index_name),
        env=env,
    )

    if file_name.endswith((".csv", ".xls", "xlsx")):
        new_ids = RunnableParallel(
            {"normal": chunk_ingest_chain, "largest": large_chunk_ingest_chain, "tabular": tabular_chunk_ingest_chain}
        ).invoke(file_name)  # Run an additional tabular process if a tabular file is ingested
    elif file_name.endswith(".docx") and has_tables_in_docx(file_name, env.s3_client()):
        new_ids = RunnableParallel(
            {"normal": chunk_ingest_chain, "largest": large_chunk_ingest_chain, "tabular": tabular_chunk_ingest_chain}
        ).invoke(file_name)  # Run an additional tabular process if a docx file with tables is ingested
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
    except Exception as e:
        logging.exception("Error while processing file [%s]", file_name)
        return f"{type(e)}: {e.args[0]}"
