from dataclasses import dataclass


@dataclass
class ChunkResolutionDetail:
    name: str
    count: int
    is_low: bool
    missing: int


@dataclass
class FileChunkResolutionResult:
    created_at_ts: int
    created_at: str
    file_id: str
    file_name: str
    user: str

    file_ingestion_status: str
    file_ingestion_ok: bool
    chunk_resolution_ok: bool
    overall_ok: bool

    stored_name: str | None = None
    resolutions: list[ChunkResolutionDetail] | None = None
    duplicates: dict | None = None
    error: str | None = None

    @classmethod
    def from_complete_file(cls, file, resolutions: dict[str, int], duplicates: dict) -> "FileChunkResolutionResult":
        counts = list(resolutions.values())
        chunk_resolution_ok = len(counts) > 0 and len(set(counts)) == 1
        file_ingestion_ok = file.status == "complete"
        max_count = max(counts) if counts else 0

        return cls(
            created_at_ts=int(file.created_at.timestamp()),
            created_at=file.created_at.strftime("%d %b %Y %H:%M"),
            file_id=str(file.pk),
            file_name=file.file_name,
            user=file.user.email,
            file_ingestion_status=file.status,
            file_ingestion_ok=file_ingestion_ok,
            chunk_resolution_ok=chunk_resolution_ok,
            overall_ok=chunk_resolution_ok and file_ingestion_ok,
            stored_name=file.unique_name,
            resolutions=[
                ChunkResolutionDetail(
                    name=resolution,
                    count=count,
                    is_low=count < max_count,
                    missing=max_count - count,
                )
                for resolution, count in sorted(resolutions.items())
            ],
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
            file_ingestion_ok=False,
            chunk_resolution_ok=False,
            overall_ok=False,
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
            file_ingestion_ok=False,
            chunk_resolution_ok=False,
            overall_ok=False,
            stored_name=file.unique_name,
            error=str(exc),
        )

    def to_dict(self) -> dict:
        return {
            **self.__dict__,
            "resolutions": [r.__dict__ for r in self.resolutions] if self.resolutions else None,
        }
