from redbox.models.settings import get_settings

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
    return {bucket["key"]: bucket["doc_count"] for bucket in resp["aggregations"]["resolutions"]["buckets"]}
