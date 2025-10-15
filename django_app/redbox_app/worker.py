import logging
from uuid import UUID

from redbox.loader.ingester import ingest_file
from redbox.models.settings import get_settings

env = get_settings()
log = logging.getLogger(__name__)


def ingest(file_id: UUID, es_index: str | None = None, malware_flagged: bool = False) -> None:
    """
    Process a file:
    - If malware_flagged=True, mark as deleted and remove from S3/ES
    - Otherwise, run ingestion workflow
    """
    from redbox_app.redbox_core.models import File

    if not es_index:
        es_index = env.elastic_chunk_alias

    try:
        file = File.objects.get(id=file_id)
    except File.DoesNotExist:
        log.warning("File with ID %s not found", file_id)
        return

    if malware_flagged:
        log.info("File flagged as malware: %s. Removing file.", file)
        file.safe_delete(reason="File flagged as malware by GuardDuty")
        return

    log.info("Ingesting file: %s", file)
    if error := ingest_file(file.unique_name, es_index):
        file.status = File.Status.errored
        file.ingest_error = error
    else:
        file.status = File.Status.complete
        file.ingest_error = None

    file.save(update_fields=["status", "ingest_error"])
