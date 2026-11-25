from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal, Optional
from uuid import UUID, uuid4

from django.conf import settings
from django.db import models
from pydantic import AliasChoices, BaseModel, Field, field_validator

from redbox.enums import IngestChunkingStrategy, IngestExtractionStrategy
from redbox.models.errors import InactiveFileError


class ChunkResolution(StrEnum):
    smallest = "smallest"
    small = "small"
    normal = "normal"
    large = "large"
    largest = "largest"
    tabular = "tabular"


class ChunkCreatorType(StrEnum):
    wikipedia = "Wikipedia"
    user_uploaded_document = "UserUploadedDocument"
    gov_uk = "GOV.UK"
    web_search = "WebSearch"
    datahub = "DataHub"


class BaseSchema(BaseModel):
    type: str
    name: str


class TabularSchema(BaseSchema):
    type: Literal["tabular"] = "tabular"
    columns: dict[str, str]

    _identifier_pattern: ClassVar[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _allowed_duckdb_types: ClassVar[set[str]] = {
        "INT",
        "INTEGER",
        "NUMERIC",
        "BIGINT",
        "SMALLINT",
        "TINYINT",
        "UBIGINT",
        "UINTEGER",
        "USMALLINT",
        "UTINYINT",
        "HUGEINT",
        "FLOAT",
        "DOUBLE",
        "REAL",
        "DECIMAL",
        "BOOLEAN",
        "BOOL",
        "DATE",
        "TIMESTAMP",
        "TIMESTAMPTZ",
        "TIME",
        "VARCHAR",
        "TEXT",
    }

    @field_validator("name")
    @classmethod
    def validate_table_name(cls, value: str) -> str:
        if not cls._identifier_pattern.fullmatch(value):
            raise ValueError("table name must be a valid SQL identifier")
        return value

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("tabular schema must include at least one column")

        for column_name, column_type in value.items():
            if not cls._identifier_pattern.fullmatch(column_name):
                raise ValueError(f"column '{column_name}' must be a valid SQL identifier")

            normalized_type = column_type.strip().upper()
            if normalized_type not in cls._allowed_duckdb_types:
                raise ValueError(f"column type '{column_type}' is not allowed")

        return value


class ChunkMetadata(BaseModel):
    """
    Worker model for document metadata for new style chunks.
    This is the minimal metadata that all ingest chains provide and should not be used to map retrieved documents (as fields will be lost)
    """

    uuid: UUID = Field(default_factory=uuid4)
    index: int = 0  # The order of this chunk in the original resource
    created_datetime: datetime = datetime.now(UTC)
    chunk_resolution: ChunkResolution = ChunkResolution.normal
    document_schema: Optional[TabularSchema] = None
    creator_type: ChunkCreatorType
    uri: str = Field(validation_alias=AliasChoices("uri", "file_name"))  # URL or file name
    token_count: int


class UploadedFileMetadata(ChunkMetadata):
    """
    Model for uploaded document chunk metadata.
    """

    page_number: int | None = None
    name: str | None = None
    description: str | None = None
    keywords: list[str] | None = None
    creator_type: ChunkCreatorType = ChunkCreatorType.user_uploaded_document


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
