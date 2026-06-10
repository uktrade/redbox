import logging

from django.contrib import admin
from django.contrib.auth import get_user_model
from django.http import HttpResponseRedirect
from django.urls import reverse
from django_q.tasks import async_task

from redbox.admin.models import FileChunkResolutionResult
from redbox.models.settings import get_settings
from redbox_app.redbox_core.models import File
from redbox_app.worker import deduplicate_chunks, ingest

logger = logging.getLogger(__name__)
User = get_user_model()

env = get_settings()


def reupload(_self, _request, queryset):
    for file in queryset:
        logger.info("Re-uploading file to core-api: %s", file)
        async_task(ingest, file.id)
        logger.info("Successfully reuploaded file %s.", file)


@admin.action(description="Backfill original file names")
def backfill_original_file_names(self, request, queryset):
    files = []
    updated = 0

    for file in queryset:
        if not file.original_file_name and file.original_file:
            file.original_file_name = file.original_file.name.split("/")[-1]
            files.append(file)
            updated += 1

    File.objects.bulk_update(files, ["original_file_name"])

    self.message_user(
        request,
        f"Updated {updated} files.",
    )


# Opensearch Files


@admin.action(description="OpenSearch: check which chunk resolutions exist")
def check_chunk_resolutions(_, request, queryset):
    results = []

    for file in queryset:
        try:
            if file.status == "complete":
                result = FileChunkResolutionResult.from_complete_file(file, env.elastic_alias)
            else:
                result = FileChunkResolutionResult.from_incomplete_file(file)
        except Exception as exc:  # noqa: BLE001
            result = FileChunkResolutionResult.from_error(file, exc)

        results.append(result.to_dict())

    request.session["chunk_resolution_results"] = results
    return HttpResponseRedirect(reverse("admin:files_chunk_resolution_report"))


def enqueue_reingest(self, request, file: File) -> None:
    logger.info("Queueing file for reingestion: %s", file)

    if file.status == "complete":
        result = FileChunkResolutionResult.from_complete_file(
            file=file,
            index_name=env.elastic_alias,
        )
        if result.file_ingestion_ok and not result.overall_ok:
            self.message_user(request, f"Starting reingest '{file.unique_name}'...")

            if not result.chunk_resolution_ok:
                self.message_user(request, f"Reingesting chunks '{file.unique_name}'...")
                async_task(ingest, file.unique_name, env.elastic_alias)
                logger.info("Successfully queued file ingest %s.", file)

            if not result.chunk_duplicates_ok:
                self.message_user(request, f"Deduplicating chunks '{file.unique_name}'...")
                async_task(deduplicate_chunks, file.unique_name, env.elastic_alias)
                logger.info("Successfully queued chunk deduplication %s.", file)

        else:
            logger.info("Failed to queue file %s - already healthy.", file)

    else:
        logger.info("Failed to queue file %s - invalid status.", file)


@admin.action(description="Re-ingest selected files")
def reingest(self, request, queryset):
    for file in queryset:
        enqueue_reingest(self=self, request=request, file=file)
