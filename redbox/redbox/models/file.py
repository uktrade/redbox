from __future__ import annotations

import datetime
import re
from enum import StrEnum
from typing import ClassVar, Literal, Optional
from uuid import UUID, uuid4

from pydantic import AliasChoices, BaseModel, Field, field_validator


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
    created_datetime: datetime.datetime = datetime.datetime.now(datetime.UTC)
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
