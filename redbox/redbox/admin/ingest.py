from collections import defaultdict

from redbox.models.settings import get_settings
from redbox.models.file import ChunkResolution

env = get_settings()


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
    resolutions = {bucket["key"]: bucket["doc_count"] for bucket in resp["aggregations"]["resolutions"]["buckets"]}

    is_tabular = file_uri.endswith((".csv", ".tsv", ".xls", ".xlsx"))
    expected_resolutions = (
        [ChunkResolution.tabular] if is_tabular else [ChunkResolution.largest, ChunkResolution.normal]
    )
    for res in expected_resolutions:
        if res not in resolutions.keys():
            resolutions[res] = 0

    return resolutions


def _get_duplicate_chunks(file_uri: str, index_name: str) -> dict[str, dict]:
    """
    Returns per-resolution duplicate summary:
    {
        "large": {"avg_duplicates_per_page": 2.5, "affected_pages": 4, "total_duplicate_chunks": 10},
        ...
    }
    """
    es = _es_client()
    # resolution -> list of per-page doc_counts
    resolution_page_counts: dict[str, list[int]] = defaultdict(list)
    after = None

    while True:
        composite = {
            "sources": [
                {"uri": {"terms": {"field": "metadata.uri.keyword"}}},
                {"resolution": {"terms": {"field": "metadata.chunk_resolution.keyword"}}},
                {"page_number": {"terms": {"field": "metadata.page_number"}}},
            ],
            "size": 1000,
        }
        if after:
            composite["after"] = after

        resp = es.search(
            index=index_name,
            body={
                "size": 0,
                "query": {"term": {"metadata.uri.keyword": file_uri}},
                "aggs": {
                    "duplicate_groups": {
                        "composite": composite,
                        "aggs": {
                            "is_duplicate": {
                                "bucket_selector": {
                                    "buckets_path": {"count": "_count"},
                                    "script": "params.count > 1",
                                }
                            }
                        },
                    }
                },
            },
        )

        buckets = resp["aggregations"]["duplicate_groups"]["buckets"]
        for b in buckets:
            resolution_page_counts[b["key"]["resolution"]].append(b["doc_count"])

        after = resp["aggregations"]["duplicate_groups"].get("after_key")
        if not after:
            break

    return {
        resolution: {
            "avg_duplicates_per_page": round(sum(counts) / len(counts), 2),
            "affected_pages": len(counts),
            "total_duplicate_chunks": sum(counts),
        }
        for resolution, counts in resolution_page_counts.items()
    }
