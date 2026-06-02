import logging
import traceback

from redbox.chains.components import (
    get_embeddings,
)

from redbox.loader.loaders import (
    DocumentLoader,
)

from redbox.models.settings import (
    get_settings,
)

from redbox.loader.services.chunker import (
    TextChunker,
)

from redbox.loader.services.embedding_batcher import EmbeddingBatcher

from redbox.loader.services.ingestion_pipeline import (
    IngestionPipeline,
)

from redbox.loader.services.opensearch_indexer import (
    OpenSearchBulkIndexer,
)

from redbox.loader.services.textract_service import (
    TextractService,
)

from redbox.loader.loaders import MetadataLoader


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

env = get_settings()


def ingest_file(file_name: str):
    try:
        textract_service = TextractService()

        loader = DocumentLoader(
            bucket=env.bucket_name,
            textract_service=textract_service,
        )

        metadata = MetadataLoader(
            env=env,
            s3_client=env.s3_client(),
            file_name=file_name,
            document_loader=loader,
        ).extract_metadata()

        chunker = TextChunker(
            min_chunk_size=env.worker_ingest_min_chunk_size,
            max_chunk_size=env.worker_ingest_max_chunk_size,
            overlap_chars=0,
        )

        embedding_batcher = EmbeddingBatcher(
            embedding_model=get_embeddings(env),
            batch_size=64,
        )

        normal_indexer = OpenSearchBulkIndexer(
            client=env.elasticsearch_client(),
            index_name=env.elastic_chunk_alias,
            vector_field_name=env.embedding_document_field_name,
        )

        schematised_indexer = OpenSearchBulkIndexer(
            client=env.elasticsearch_client(),
            index_name=env.elastic_schematised_chunk_index,
            vector_field_name=env.embedding_document_field_name,
        )

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_batcher=embedding_batcher,
            indexer=normal_indexer,
            schematised_indexer=schematised_indexer,
            metadata=metadata,
        )

        pipeline.ingest(file_name)

        logger.info("Successfully ingested %s", file_name)

    except Exception:
        logger.exception("Error ingesting file %s", file_name)
        return traceback.format_exc()
