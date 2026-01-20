import logging
from typing import Any

from langchain_core.documents import Document

from redbox.models.chain import AISettings, RedboxState
from redbox.models.file import ChunkResolution

log = logging.getLogger()


def build_file_filter(file_names: list[str]) -> dict[str, Any]:
    """Creates an Elasticsearch filter for file names."""
    return {"terms": {"metadata.uri.keyword": file_names}}


def build_resolution_filter(chunk_resolution: ChunkResolution) -> dict[str, Any]:
    """Creates an Elasticsearch filter for chunk resolutions."""
    return {"term": {"metadata.chunk_resolution.keyword": chunk_resolution.value}}


def build_query_filter(
    selected_files: list[str], permitted_files: list[str], chunk_resolution: ChunkResolution | None
) -> list[dict[str, Any]]:
    """Generic filter constructor for all queries.

    Warns if the selected S3 keys aren't in the permitted S3 keys.
    """
    selected_files = set(selected_files)
    permitted_files = set(permitted_files)

    if not selected_files <= permitted_files:
        log.warning(
            f"User has selected files they aren't permitted to access: \n{', '.join(selected_files - permitted_files)}"
        )

    file_names = list(selected_files & permitted_files)

    list_filters = []

    list_filters.append(build_file_filter(file_names=file_names))

    if chunk_resolution:
        list_filters.append(build_resolution_filter(chunk_resolution=chunk_resolution))

    query_filter = {
        "bool": {
            "must": list_filters
        }  # filter returns the results that matches all the listed filter. This is a logical AND operator. The results must match all queries in this clause.
    }

    return query_filter


def get_all(
    chunk_resolution: ChunkResolution | None,
    state: RedboxState,
) -> dict[str, Any]:
    """
    Returns a parameterised elastic query that will return everything it matches.

    As it's used in summarisation, it excludes embeddings.
    """
    query_filter = build_query_filter(
        selected_files=state.request.s3_keys,
        permitted_files=state.request.permitted_s3_keys,
        chunk_resolution=chunk_resolution,
    )

    return {
        "_source": {"excludes": ["vector_field"]},
        "query": {"bool": {"must": {"match_all": {}}, "filter": query_filter}},
    }


def get_knowledge_base(
    chunk_resolution: ChunkResolution | None,
    state: RedboxState,
) -> dict[str, Any]:
    """
    Returns a parameterised elastic query that will return everything it matches.

    Query against knowledge base
    """
    query_filter = build_query_filter(
        selected_files=state.request.knowledge_base_s3_keys,
        permitted_files=state.request.knowledge_base_s3_keys,
        chunk_resolution=chunk_resolution,
    )

    return {
        "_source": {"excludes": ["vector_field"]},
        "query": {"bool": {"must": {"match_all": {}}, "filter": query_filter}},
    }


def get_metadata(
    chunk_resolution: ChunkResolution | None,
    state: RedboxState,
) -> dict[str, Any]:
    query_filter = build_query_filter(
        selected_files=state.request.s3_keys,
        permitted_files=state.request.permitted_s3_keys,
        chunk_resolution=chunk_resolution,
    )

    return {
        "_source": {"excludes": ["vector_field", "text"]},
        "query": {"bool": {"must": {"match_all": {}}, "filter": query_filter}},
    }


def get_minimum_metadata(
    chunk_resolution: ChunkResolution | None,
    state: RedboxState,
) -> dict[str, Any]:
    """Retrive document metadata without page_content"""
    query_filter = build_query_filter(
        selected_files=state.request.s3_keys,
        permitted_files=state.request.permitted_s3_keys,
        chunk_resolution=chunk_resolution,
    )

    return {
        "_source": {"includes": ["metadata.name", "metadata.description", "metadata.keywords"]},
        "query": {"bool": {"must": {"match_all": {}}, "filter": query_filter}},
    }


def get_knowledge_base_metadata(
    chunk_resolution: ChunkResolution | None,
    state: RedboxState,
) -> dict[str, Any]:
    """Retrive knowledge base metadata without page_content"""
    query_filter = build_query_filter(
        selected_files=state.request.knowledge_base_s3_keys,
        permitted_files=state.request.knowledge_base_s3_keys,
        chunk_resolution=chunk_resolution,
    )

    return {
        "_source": {"includes": ["metadata.name", "metadata.description", "metadata.keywords"]},
        "query": {"bool": {"must": {"match_all": {}}, "filter": query_filter}},
    }


def get_knowledge_base_tabular_metadata(
    chunk_resolution: ChunkResolution | None,
    state: RedboxState,
) -> dict[str, Any]:
    """Retrive knowledge base metadata without page_content"""
    query_filter = build_query_filter(
        selected_files=state.request.knowledge_base_s3_keys,
        permitted_files=state.request.knowledge_base_s3_keys,
        chunk_resolution=chunk_resolution,
    )

    return {
        "_source": {
            "includes": [
                "metadata.uri",
                "metadata.name",
                "metadata.description",
                "metadata.keywords",
                "metadata.document_schema",
            ]
        },
        "query": {"bool": {"must": {"match_all": {}}, "filter": query_filter}},
    }


def get_k_value(file_list, desired_size=30):
    """
    Simple rule: more files filtered = lower k needed
    """
    num_files = len(file_list)

    if num_files <= 3:
        return desired_size * 3  # k=90 for very restrictive
    elif num_files <= 10:
        return desired_size * 2  # k=60 for restrictive
    elif num_files <= 30:
        return int(desired_size * 1.5)  # k=45 for moderate
    else:
        return desired_size  # k=30 for many files


def build_document_query(
    query: str,
    query_vector: list[float],
    embedding_field_name: str,
    ai_settings: AISettings,
    permitted_files: list[str],
    selected_files: list[str] | None = None,
    chunk_resolution: ChunkResolution | None = None,
) -> dict[str, Any]:
    """Builds a an Elasticsearch query that will return documents when called.

    Searches the document:
        * Text, as a keyword and similarity
    """

    query_filter = build_query_filter(
        selected_files=selected_files,
        permitted_files=permitted_files,
        chunk_resolution=chunk_resolution,
    )

    k_value = get_k_value(selected_files, desired_size=ai_settings.rag_num_candidates)
    return {
        "size": ai_settings.rag_k,
        "min_score": 0.6,
        "query": {
            "knn": {
                "vector_field": {
                    "vector": query_vector,
                    "k": k_value,
                    # "boost": ai_settings.knn_boost,
                    "filter": query_filter,
                }
            }
        },
        "_source": {"excludes": ["vector_field"]},
    }


def scale_score(score: float, old_min: float, old_max: float, new_min=1.1, new_max: float = 2.0):
    """Rescales an Elasticsearch score.

    Intended to turn the score into a multiplier to weight a Gauss function.

    If the old range is zero or undefined, returns new_min.
    """
    if old_max == old_min:
        return new_min

    return new_min + (score - old_min) * (new_max - new_min) / (old_max - old_min)


def add_document_filter_scores_to_query(
    elasticsearch_query: dict[str, Any],
    ai_settings: AISettings,
    centres: list[Document],
) -> dict[str, Any]:
    """
    Adds Gaussian function scores to a query centred on a list of documents.

    This function score will scale the centres' scores into a multiplier, and
    boost the score of documents with an index close to them.

    The result will be that documents with the same file name will have their
    score boosted in proportion to how close their index is to a file in the
    "centres" list.

    For example, if foo.txt index 9 with score 7 was passed in the centres list,
    if foo.txt index 10 would have scored 2, it will now be boosted to score 4.
    """
    gauss_functions: list[dict[str, Any]] = []
    gauss_scale = ai_settings.rag_gauss_scale_size
    gauss_decay = ai_settings.rag_gauss_scale_decay
    scores = [d.metadata["score"] for d in centres]
    old_min = min(scores)
    old_max = max(scores)
    new_min = ai_settings.rag_gauss_scale_min
    new_max = ai_settings.rag_gauss_scale_max

    for document in centres:
        gauss_functions.append(
            {
                "filter": {"term": {"metadata.file_name.keyword": document.metadata["uri"]}},
                "gauss": {
                    "metadata.index": {
                        "origin": document.metadata["index"],
                        "scale": gauss_scale,
                        "offset": 0,
                        "decay": gauss_decay,
                    }
                },
                "weight": scale_score(
                    score=document.metadata["score"],
                    old_min=old_min,
                    old_max=old_max,
                    new_min=new_min,
                    new_max=new_max,
                ),
            }
        )

    # The size should minimally capture changes to documents either
    # side of every Gauss function applied, including the document
    # itself (double + 1). Of course, this is a ranking, so most of
    # these results will be removed again later
    return {
        "size": elasticsearch_query.get("size") * ((gauss_scale * 2) + 1),
        "query": {
            "function_score": {
                "query": elasticsearch_query.get("query"),
                "functions": gauss_functions,
                "score_mode": "max",
                "boost_mode": "multiply",
            }
        },
    }
