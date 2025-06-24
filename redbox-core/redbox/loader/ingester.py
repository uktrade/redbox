import logging
import time
import os
from typing import Dict, List, Optional
from sqlalchemy import create_engine, inspect
import pandas as pd
from sqlalchemy.engine import Engine
from redbox_app.redbox_core.models import File
from django.conf import settings
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


def ingest_files(
    file_path: str, sheet_names: Optional[List[str] | str] = None
) -> pd.DataFrame | Dict[str, pd.DataFrame]:
    try:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            if not sheet_names:
                return pd.read_excel(file_path)
            elif sheet_names == "all":
                return pd.read_excel(file_path, sheet_name=None)
            else:
                sheets = pd.ExcelFile(file_path).sheet_names
                df_map = {}
                for sht in sheet_names:
                    if sht in sheets:
                        df_map[sht] = pd.read_excel(file_path, sheet_name=sht)
                return df_map
        else:
            raise TypeError("Only CSV and Excel files are accepted")
    except Exception as e:
        raise e


def create_db_tables(dfs: Dict[str, pd.DataFrame], engine: Engine, replace: bool = False):
    inspector = inspect(engine)
    existing = inspector.get_table_names()
    for table_name in dfs:
        table_name_clean = table_name.strip().replace(" ", "_").lower()
        if replace or table_name_clean not in existing:
            dfs[table_name].to_sql(table_name_clean, engine, index=False, if_exists="replace" if replace else "fail")


def _ingest_file(file_name: str, es_index_name: str = alias, enable_metadata_extraction=env.enable_metadata_extraction):
    logging.info("Ingesting file: %s", file_name)
    start_time = time.time()

    es = env.elasticsearch_client()

    is_tabular = file_name.endswith((".csv", ".xls", ".xlsx"))

    try:
        file_obj = File.objects.get(unique_name=file_name)
    except File.DoesNotExist:
        logging.error("File object not found for %s", file_name)
        raise

    if is_tabular:
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)
        db_location = os.path.join(settings.MEDIA_ROOT, f"{file_name}.db")

        try:
            dfs = ingest_files(file_path, sheet_names="all")
        except Exception:
            logging.exception("Error ingesting tabular file %s", file_name)
            raise

        engine = create_engine(f"sqlite:///{db_location}")

        if isinstance(dfs, pd.DataFrame):
            table_name = file_obj.file_name.split(".")[0].strip().replace(" ", "_").lower()
            create_db_tables({table_name: dfs}, engine, replace=True)
            table_names = [table_name]
        else:
            create_db_tables(dfs, engine, replace=True)
            table_names = list(dfs.keys())

        file_obj.db_location = db_location
        file_obj.table_names = table_names
        file_obj.status = File.Status.complete
        file_obj.save()

        # Extract metadata
        if enable_metadata_extraction:
            metadata_loader = MetadataLoader(env=env, s3_client=env.s3_client(), file_name=file_name)
            metadata = metadata_loader.extract_metadata()
        else:
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

        new_ids = RunnableParallel({"normal": chunk_ingest_chain, "largest": large_chunk_ingest_chain}).invoke(
            file_name
        )
        logging.info(
            "File: %s %s chunks ingested (metadata only for tabular)",
            file_name,
            {k: len(v) for k, v in new_ids.items()},
        )
    else:
        if es_index_name == alias:
            if not es.indices.exists_alias(name=alias):
                print("The alias does not exist")
                create_alias(alias)
        else:
            es.indices.create(index=es_index_name, body=env.index_mapping, ignore=400)

        if enable_metadata_extraction:
            metadata_loader = MetadataLoader(env=env, s3_client=env.s3_client(), file_name=file_name)
            metadata = metadata_loader.extract_metadata()
        else:
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

        new_ids = RunnableParallel({"normal": chunk_ingest_chain, "largest": large_chunk_ingest_chain}).invoke(
            file_name
        )
        logging.info(
            "File: %s %s chunks ingested",
            file_name,
            {k: len(v) for k, v in new_ids.items()},
        )

        # Update File status for non-tabular files
        file_obj.status = File.Status.complete
        file_obj.save()

    duration = time.time() - start_time
    logging.info("Total ingestion for file [%s] took %.2f seconds", file_name, duration)


def ingest_file(file_name: str, es_index_name: str = alias) -> str | None:
    try:
        _ingest_file(file_name, es_index_name)
    except Exception as e:
        logging.exception("Error while processing file [%s]", file_name)
        return f"{type(e)}: {e.args[0]}"
