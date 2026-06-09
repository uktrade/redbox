import logging

from django.contrib import admin
from django.contrib.auth import get_user_model
from django.http import HttpResponseRedirect
from django.urls import reverse
from django_q.tasks import async_task

from redbox.models.settings import get_settings
from redbox_app.redbox_core.models import File
from redbox_app.worker import ingest

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


def _es_client():
    return env.elasticsearch_client()


def _get_resolutions_for_file(file_uri: str, index_name: str) -> dict[str, int]:
    """Return chunk_resolution -> count for *file_uri* across the main alias index."""
    es = _es_client()
    resp = es.search(
        index=index_name,
        body={
            "size": 0,
            "query": {"term": {"metadata.uri.keyword": file_uri}},
            "aggs": {"resolutions": {"terms": {"field": "metadata.chunk_resolution.keyword"}}},
        },
    )
    return {bucket["key"]: bucket["doc_count"] for bucket in resp["aggregations"]["resolutions"]["buckets"]}


def _file_has_largest_chunks(file_uri: str, index_name: str) -> bool:
    """Return True if any 'largest' chunks already exist for *file_id*."""
    es = _es_client()
    resp = es.search(
        index=index_name,
        body={
            "size": 1,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"metadata.uri": file_uri}},
                        {"term": {"metadata.chunk_resolution": "largest"}},
                    ]
                }
            },
        },
    )
    return resp["hits"]["total"]["value"] > 0


@admin.action(description="OpenSearch: check which chunk resolutions exist")
def check_chunk_resolutions(_, request, queryset):
    results = []

    for file in queryset:
        try:
            if file.status == "complete":
                resolutions = _get_resolutions_for_file(
                    file_uri=file.unique_name,
                    index_name=env.elastic_alias,
                )

                counts = list(resolutions.values())
                healthy = len(counts) > 0 and len(set(counts)) == 1
                max_count = max(counts) if counts else 0

                formatted_resolutions = [
                    {
                        "name": resolution,
                        "count": count,
                        "is_low": count < max_count,
                        "missing": max_count - count,
                    }
                    for resolution, count in sorted(resolutions.items())
                ]

                results.append(
                    {
                        "created_at_ts": int(file.created_at.timestamp()),
                        "created_at": file.created_at.strftime("%d %b %Y %H:%M"),
                        "file_id": str(file.pk),
                        "file_name": file.file_name,
                        "user": file.user.email,
                        "status": file.status,
                        "healthy": healthy,
                        "overall_ok": healthy and file.status == "complete",
                        "stored_name": file.unique_name,
                        "resolutions": formatted_resolutions,
                        "error": None,
                    }
                )
            else:
                results.append(
                    {
                        "created_at_ts": int(file.created_at.timestamp()),
                        "created_at": file.created_at.strftime("%d %b %Y %H:%M"),
                        "file_id": str(file.pk),
                        "file_name": file.file_name,
                        "user": file.user.email,
                        "status": file.status,
                        "healthy": False,
                        "overall_ok": False,
                        "stored_name": None,
                        "resolutions": None,
                        "error": None,
                    }
                )

        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "created_at_ts": int(file.created_at.timestamp()),
                    "created_at": file.created_at.strftime("%d %b %Y %H:%M"),
                    "file_id": str(file.pk),
                    "file_name": file.file_name,
                    "user": file.user.email,
                    "status": file.status,
                    "healthy": False,
                    "overall_ok": False,
                    "stored_name": file.unique_name,
                    "resolutions": None,
                    "error": str(exc),
                }
            )

    request.session["chunk_resolution_results"] = results

    return HttpResponseRedirect(reverse("admin:files_chunk_resolution_report"))
