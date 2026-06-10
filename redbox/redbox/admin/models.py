from dataclasses import dataclass
from redbox.admin.ingest import (
    _get_resolutions_for_file,
    _get_duplicate_chunks,
    ChunkResolutionDetail,
    ChunkDuplicateDetail,
)


@dataclass
class FileChunkResolutionResult:
    created_at_ts: int
    created_at: str
    file_id: str
    file_name: str
    user: str

    file_ingestion_status: str
    file_ingestion_ok: bool = False
    chunk_resolution_ok: bool = False
    chunk_duplicates_ok: bool = False
    overall_ok: bool = False

    stored_name: str | None = None
    resolutions: list[ChunkResolutionDetail] | None = None
    duplicates: list[ChunkDuplicateDetail] | None = None
    error: str | None = None
    reingestion_queued: bool | None = None

    @classmethod
    def from_complete_file(cls, file, index_name: str) -> "FileChunkResolutionResult":
        resolutions = _get_resolutions_for_file(
            file_uri=file.unique_name,
            index_name=index_name,
        )
        duplicates = _get_duplicate_chunks(
            file_uri=file.unique_name,
            index_name=index_name,
        )

        counts = [r.count for r in resolutions]

        chunk_resolution_ok = len(counts) > 0 and len(set(counts)) == 1
        chunk_duplicate_ok = len(duplicates) == 0
        chunk_ok = chunk_resolution_ok and chunk_duplicate_ok

        file_ingestion_ok = file.status == "complete"

        overall_ok = chunk_ok and file_ingestion_ok

        return cls(
            created_at_ts=int(file.created_at.timestamp()),
            created_at=file.created_at.strftime("%d %b %Y %H:%M"),
            file_id=str(file.pk),
            file_name=file.file_name,
            user=file.user.email,
            file_ingestion_status=file.status,
            file_ingestion_ok=file_ingestion_ok,
            chunk_resolution_ok=chunk_resolution_ok,
            chunk_duplicates_ok=chunk_duplicate_ok,
            overall_ok=overall_ok,
            stored_name=file.unique_name,
            resolutions=resolutions,
            duplicates=duplicates,
        )

    @classmethod
    def from_incomplete_file(cls, file) -> "FileChunkResolutionResult":
        return cls(
            created_at_ts=int(file.created_at.timestamp()),
            created_at=file.created_at.strftime("%d %b %Y %H:%M"),
            file_id=str(file.pk),
            file_name=file.file_name,
            user=file.user.email,
            file_ingestion_status=file.status,
        )

    @classmethod
    def from_error(cls, file, exc: Exception) -> "FileChunkResolutionResult":
        return cls(
            created_at_ts=int(file.created_at.timestamp()),
            created_at=file.created_at.strftime("%d %b %Y %H:%M"),
            file_id=str(file.pk),
            file_name=file.file_name,
            user=file.user.email,
            file_ingestion_status=file.status,
            stored_name=file.unique_name,
            error=str(exc),
        )

    def to_dict(self) -> dict:
        return {
            **self.__dict__,
            "resolutions": [r.__dict__ for r in self.resolutions] if self.resolutions else None,
            "duplicates": [d.__dict__ for d in self.duplicates] if self.duplicates else None,
        }
