from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from django.conf import settings
from django.db import models

from redbox.enums import IngestChunkingStrategy, IngestExtractionStrategy
from redbox.models.errors import InactiveFileError


def build_s3_key(instance, filename: str) -> str:
    """the s3-key is the primary key for a file,
    this needs to be unique so that if a user uploads a file with the same name as
    1. an existing file that they own, then it is overwritten
    2. an existing file that another user owns then a new file is created

    note: s3 key is not prefixed with the user's email address if not local as filename is unique
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")[:-3]
    path = Path(filename)
    new_filename = f"{path.stem}_{timestamp}{path.suffix}"
    return f"{instance.user.email}/{new_filename}"


class UUIDPrimaryKeyBase(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    class Meta:
        abstract = True


class FileBase(UUIDPrimaryKeyBase):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    class Status(models.TextChoices):
        complete = "complete"
        deleted = "deleted"
        errored = "errored"
        processing = "processing"

    INACTIVE_STATUSES = [Status.deleted, Status.errored]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_file_name = models.TextField(max_length=2048, blank=True, null=True)
    original_file = models.FileField(
        storage=settings.STORAGES["default"]["BACKEND"],
        upload_to=build_s3_key,
    )
    status = models.CharField(choices=Status.choices, null=False, blank=False)
    ingested_at = models.DateTimeField(blank=True, null=True)
    ingest_error = models.TextField(
        max_length=2048,
        blank=True,
        null=True,
        help_text="error, if any, encountered during ingest",
    )
    ingest_extraction_strategy_normal = models.CharField(
        max_length=64,
        choices=IngestExtractionStrategy.choices,
        default=IngestExtractionStrategy.unspecified,
        null=True,
    )
    ingest_chunking_strategy_normal = models.CharField(
        max_length=64,
        choices=IngestChunkingStrategy.choices,
        default=IngestChunkingStrategy.unspecified,
        null=True,
    )
    ingest_extraction_strategy_largest = models.CharField(
        max_length=64,
        choices=IngestExtractionStrategy.choices,
        default=IngestExtractionStrategy.unspecified,
        null=True,
    )
    ingest_chunking_strategy_largest = models.CharField(
        max_length=64,
        choices=IngestChunkingStrategy.choices,
        default=IngestChunkingStrategy.unspecified,
        null=True,
    )

    @property
    def unique_name(self) -> str:
        """primary key for accessing file in s3"""
        if self.status in self.INACTIVE_STATUSES:
            raise InactiveFileError(self)
        return self.original_file.name if self.original_file else ""

    class Meta:
        abstract = True
        app_label = "redbox_core"
