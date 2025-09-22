import logging
from uuid import UUID

from django.contrib.auth import get_user_model

from redbox.loader.ingester import ingest_file
from redbox.models.settings import get_settings
from redbox_app.redbox_core.models import File
from redbox_app.redbox_core.services import documents as documents_service

env = get_settings()
User = get_user_model()
logger = logging.getLogger(__name__)


def ingest(file_id: UUID, es_index: str | None = None) -> None:
    # These models need to be loaded at runtime otherwise they can be loaded before they exist

    if not es_index:
        es_index = env.elastic_chunk_alias

    file = File.objects.get(id=file_id)
    logger.info("Ingesting file: %s", file)

    if error := ingest_file(file.unique_name, es_index):
        file.status = File.Status.errored
        file.ingest_error = error
    else:
        file.status = File.Status.complete

    file.save()


def process_uploaded_files(
    file_ids: list[UUID], user_id: UUID, es_index: str | None = None
) -> tuple[list[str], list[dict]]:
    """
    Validate, convert, and ingest files using django Q
    Returns a tuple of errors and the results where results contain file statuses.
    """
    # These models need to be loaded at runtime otherwise they can be loaded before they exist

    if not es_index:
        es_index = env.elastic_chunk_alias

    errors: list[str] = []
    results: list[dict] = []
    user = User.objects.get(id=user_id)

    for file_id in file_ids:
        try:
            file = File.objects.get(id=file_id)
            uploaded_file = file.original_file

            # Validate the file
            validation_errors = documents_service.validate_uploaded_file(uploaded_file)
            if validation_errors:
                errors.extend(validation_errors)
                file.status = File.Status.errored
                file.ingest_error = "; ".join(validation_errors)
                file.save()
                results.append({"file_id": str(file.id), "file_name": file.file_name, "status": file.status})
                continue

            # Handle doc -> docx conversion
            if documents_service.is_doc_file(uploaded_file):
                converted_file = documents_service.convert_doc_to_docx(uploaded_file)
                file.original_file = converted_file
                file.original_file_name = converted_file.name
                file.save()

            # Handle UTF-8 compatibility
            if not documents_service.is_utf8_compatible(file.original_file):
                converted_file = documents_service.convert_to_utf8(file.original_file)
                file.original_file = converted_file
                file.original_file_name = converted_file.name
                file.save()

            # Ingest the file
            ingest_errors = ingest(file.id, user.id)
            file.refresh_from_db()  # Refresh to get updated status
            result = {"file_id": str(file.id), "file_name": file.file_name, "status": file.status}
            if ingest_errors:
                result["ingest_errors"] = ingest_errors
            results.append(result)

        except Exception as e:
            logger.exception("Error processing file %s", file_id, exc_info=e)
            errors.append(f"Error processing {file.file_name}: {e!s}")
            file.status = File.Status.errored
            file.ingest_error = str(e)
            file.save()
            results.append({"file_id": str(file.id), "file_name": file.file_name, "status": file.status})

    return errors, results
