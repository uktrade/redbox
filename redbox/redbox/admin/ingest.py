from dataclasses import dataclass
from collections import defaultdict

from redbox.models.settings import get_settings
from redbox.models.file import ChunkResolution

env = get_settings()


@dataclass
class ChunkResolutionDetail:
    name: str
    count: int


@dataclass
class ChunkDuplicateDetail:
    name: str
    avg_duplicates_per_page: float
    affected_pages: int
    total_duplicate_chunks: int


def _es_client():
    return env.elasticsearch_client()


def _get_resolutions_for_file(file_uri: str, index_name: str) -> list[ChunkResolutionDetail]:
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
    resolutions = {
        bucket["key"]: ChunkResolutionDetail(name=bucket["key"], count=bucket["doc_count"])
        for bucket in resp["aggregations"]["resolutions"]["buckets"]
    }

    is_tabular = file_uri.endswith((".csv", ".tsv", ".xls", ".xlsx"))
    expected_resolutions = (
        [ChunkResolution.tabular] if is_tabular else [ChunkResolution.largest, ChunkResolution.normal]
    )
    for res in expected_resolutions:
        if res not in resolutions.keys():
            resolutions[res] = ChunkResolutionDetail(name=res, count=0)

    return resolutions.values()


def _chunk_dedupe_key(hit: dict) -> tuple:
    meta = hit["_source"]["metadata"]

    return (
        meta["uri"],
        meta["chunk_resolution"],
        meta["page_number"],
        hit["_source"]["text"],
    )


def _get_duplicate_chunks(
    file_uri: str,
    index_name: str,
) -> list[ChunkDuplicateDetail]:
    """
    Returns per-resolution duplicate summary.

    A duplicate is defined as chunks having the same:
        (chunk_resolution, page_number, text)

    Metrics:
        affected_pages         = unique pages containing duplicates
        total_duplicate_chunks = extra duplicate chunks beyond the first
        avg_duplicates_per_page = total_duplicate_chunks / affected_pages
    """
    es = _es_client()

    resp = es.search(
        index=index_name,
        body={
            "size": 10000,
            "query": {
                "term": {
                    "metadata.uri.keyword": file_uri,
                }
            },
            "_source": [
                "text",
                "metadata.uri",
                "metadata.chunk_resolution",
                "metadata.page_number",
            ],
        },
    )

    seen: set[tuple[str, int, str]] = set()

    resolution_pages: dict[str, set[int]] = defaultdict(set)
    resolution_duplicate_counts: dict[str, int] = defaultdict(int)

    for hit in resp["hits"]["hits"]:
        key = _chunk_dedupe_key(hit)

        if key not in seen:
            seen.add(key)
            continue

        meta = hit["_source"]["metadata"]

        resolution = meta["chunk_resolution"]
        page_number = meta["page_number"]

        resolution_duplicate_counts[resolution] += 1
        resolution_pages[resolution].add(page_number)

    return [
        ChunkDuplicateDetail(
            name=resolution,
            affected_pages=len(resolution_pages[resolution]),
            total_duplicate_chunks=resolution_duplicate_counts[resolution],
            avg_duplicates_per_page=round(
                resolution_duplicate_counts[resolution] / len(resolution_pages[resolution]),
                2,
            )
            if resolution_pages[resolution]
            else 0,
        )
        for resolution in sorted(resolution_duplicate_counts)
    ]
