import logging

from django.contrib import admin
from django.contrib.auth import get_user_model

from redbox_app.redbox_core.models import File, FileTool
from redbox_app.redbox_core.services.documents import reingest_file

logger = logging.getLogger(__name__)
User = get_user_model()


@admin.action(description="Re-ingest files")
def reingest(self, request, queryset):
    if not (file_count := len(queryset)):
        return logger.error("No files selected for re-ingestion")

    if self.model == FileTool:
        for file_tool in queryset:
            reingest_file(file_tool.file)

    else:
        for file in queryset:
            reingest_file(file)

    msg = f"Re-ingesting {file_count} files"

    logger.info(msg)
    return self.message_user(request, msg)


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

    return self.message_user(
        request,
        f"Updated {updated} files.",
    )
