import logging

from django.contrib import admin
from django.contrib.auth import get_user_model
from django_q.tasks import async_task

from redbox_app.redbox_core.models import File
from redbox_app.worker import ingest

logger = logging.getLogger(__name__)
User = get_user_model()


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
