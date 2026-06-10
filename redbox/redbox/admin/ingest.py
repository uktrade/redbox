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
