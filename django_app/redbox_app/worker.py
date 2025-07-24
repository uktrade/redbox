import logging
from uuid import UUID

from django.core.files.base import ContentFile
from minio.error import S3Error

from redbox.loader.ingester import ingest_file
from redbox.models.settings import get_settings

env = get_settings()

minio_client = env.s3_client()


def ingest(file_id: UUID, es_index: str | None = None) -> None:
    # These models need to be loaded at runtime otherwise they can be loaded before they exist
    from redbox_app.redbox_core.models import File, build_s3_key

    if not es_index:
        es_index = env.elastic_chunk_alias

    file = File.objects.get(id=file_id)
    logging.info("Ingesting file: %s", file)

    try:
        logging.info("Fetching file from MinIO: %s", file.minio_path)
        response = minio_client.get_object("processed-files", file.minio_path)
        content = response.read()

        s3_key = build_s3_key(file)
        file.original_file.save(s3_key, ContentFile(content), save=False)
        file.save()

        if error := ingest_file(file.unique_name, es_index):
            file.status = File.Status.errored
            file.ingest_error = error
        else:
            file.status = File.Status.complete

    except S3Error:
        logging.exception("MinIO error for file %s", file.minio_path)
        file.status = File.Status.errored
        file.ingest_error = "Failed to retrieve file from storage"
    except Exception:
        logging.exception("Unexpected error ingesting file %s", file.minio_path)
        file.status = File.Status.errored
        file.ingest_error = "An unexpected error occurred during ingestion"
        file.save()
    finally:
        if "response" in locals():
            response.close()
            response.release_conn()

    file.save()
