import logging
import os
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, cast

from elasticsearch import Elasticsearch
from kneed import KneeLocator
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from opensearchpy import OpenSearch

# from elasticsearch.helpers import scan
from opensearchpy.helpers import scan

from redbox.models.chain import RedboxState
from redbox.models.file import ChunkResolution
from redbox.retriever.queries import (
    add_document_filter_scores_to_query,
    build_document_query,
    get_all,
    get_metadata,
    get_minimum_metadata,
)
from redbox.transform import merge_documents, sort_documents

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger()


class OpenSearchRetriever(BaseRetriever):
    """OpenSearch Retriever."""

    es_client: OpenSearch
    index_name: Union[str, Sequence[str]]
    body_func: Callable[[str], Dict]
    content_field: Optional[Union[str, Mapping[str, str]]] = None
    document_mapper: Optional[Callable[[Mapping], Document]] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.content_field = str(self.content_field)

        # if self.content_field is None and self.document_mapper is None:
        #     raise ValueError("Either content_field or document_mapper must be provided")

        # if self.content_field is not None and self.document_mapper is not None:
        #     raise ValueError("Only one of content_field or document_mapper can be provided")

        if not self.document_mapper:
            self.document_mapper = self._single_field_mapper
        elif isinstance(self.content_field, Mapping):
            self.document_mapper = self._multi_field_mapper
        # else:
        #     print(self.content_field)
        #     raise ValueError("content_field must be a string or a mapping")

        # self.os_client = create_opensearch_client(**kwargs)

    @staticmethod
    def from_os_params(
        self,
        index_name: Union[str, Sequence[str]],
        body_func: Callable[[str], Dict],
        content_field: Optional[Union[str, Mapping[str, str]]] = None,
        document_mapper: Optional[Callable[[Mapping], Document]] = None,
        opensearch_url: Optional[str] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "OpenSearchRetriever":
        es_client = self.es_client
        return OpenSearchRetriever(
            es_client=es_client,
            index_name=index_name,
            body_func=body_func,
            content_field=content_field,
            document_mapper=document_mapper,
        )

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        if not self.es_client or not self.document_mapper:
            raise ValueError("OpenSearch client or document mapper is not initialized")

        body = self.body_func(query)
        logger.info("query to opensearch: from get_relevant_documents")
        logger.info(str(body))
        response = self.es_client.search(index=self.index_name, body=body)
        logger.info("response from opensearch: from get_relevant_documents")
        logger.info(str([self.document_mapper(hit) for hit in response["hits"]["hits"]]))
        return [self.document_mapper(hit) for hit in response["hits"]["hits"]]

    def _single_field_mapper(self, hit: Mapping[str, Any]) -> Document:
        content = hit["_source"].pop(self.content_field)
        return Document(page_content=content, metadata=hit)

    def _multi_field_mapper(self, hit: Mapping[str, Any]) -> Document:
        self.content_field = cast(Mapping, self.content_field)
        field = self.content_field[hit["_index"]]
        content = hit["_source"].pop(field)
        return Document(page_content=content, metadata=hit)


def hit_to_doc(hit: dict[str, Any]) -> Document:
    """
    Backwards compatibility for Chunks and Documents.

    Chunks has two metadata fields in top-level: index and file_name. This moves them.
    """
    source = hit["_source"]
    c_meta = {
        "index": source.get("index"),
        "uri": source["metadata"].get(
            "uri", source["metadata"].get("file_name")
        ),  # Handle mapping previously ingested documents
        "score": hit["_score"],
        "uuid": hit["_id"],
    }
    return Document(
        page_content=source.get("text", ""),
        metadata={k: v for k, v in c_meta.items() if v is not None} | source["metadata"],
    )


def query_to_documents(
    es_client: Union[Elasticsearch, OpenSearch], index_name: str, query: dict[str, Any]
) -> list[Document]:
    """Runs an Elasticsearch query and returns Documents."""
    logger.info("query to opensearch: from query_to_documents")
    logger.info(str(query))
    response = es_client.search(index=index_name, body=query)
    logger.info("response from opensearch: from query_to_documents")
    logger.info(str([hit_to_doc(hit) for hit in response["hits"]["hits"]]))
    return [hit_to_doc(hit) for hit in response["hits"]["hits"]]


def filter_by_elbow(
    enabled: bool = True, sensitivity: float = 1, score_scaling_factor: float = 100
) -> Callable[[list[Document]], list[Document]]:
    """Filters a list of documents by the elbow point on the curve of their scores.

    Args:
        enabled (bool, optional): Whether to enable the filter. Defaults to True.
    Returns:
        callable: A function that takes a list of documents and returns a list of documents.
    """

    def _filter_by_elbow(docs: list[Document]) -> list[Document]:
        # If enabled, return only the documents up to the elbow point
        if enabled:
            if len(docs) == 0:
                return docs

            # *scaling because algorithm performs poorly on changes of ~1.0
            try:
                scores = [doc.metadata["score"] * score_scaling_factor for doc in docs]
            except AttributeError as exc:
                raise exc

            rank = range(len(scores))

            # Convex curve, decreasing direction as scores descend in a pareto-like fashion
            kn = KneeLocator(rank, scores, S=sensitivity, curve="convex", direction="decreasing")
            return docs[: kn.elbow]
        else:
            return docs

    return _filter_by_elbow


class ParameterisedElasticsearchRetriever(BaseRetriever):
    """A modified ElasticsearchRetriever that allows configuration from RedboxState."""

    es_client: Union[Elasticsearch, OpenSearch]
    index_name: str | Sequence[str]
    embedding_model: Embeddings
    embedding_field_name: str = "embedding"
    chunk_resolution: ChunkResolution = ChunkResolution.normal

    def _get_relevant_documents(
        self, query: RedboxState, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        query_text = query.last_message.content
        query_vector = self.embedding_model.embed_query(query_text)
        selected_files = query.request.s3_keys
        permitted_files = query.request.permitted_s3_keys
        ai_settings = query.request.ai_settings

        # Initial pass
        initial_query = build_document_query(
            query=query_text,
            query_vector=query_vector,
            selected_files=selected_files,
            permitted_files=permitted_files,
            embedding_field_name=self.embedding_field_name,
            chunk_resolution=self.chunk_resolution,
            ai_settings=ai_settings,
        )
        initial_documents = query_to_documents(
            es_client=self.es_client, index_name=self.index_name, query=initial_query
        )

        # Handle nothing found (as when no files are permitted)
        if not initial_documents:
            return []

        # Adjacent documents
        with_adjacent_query = add_document_filter_scores_to_query(
            elasticsearch_query=initial_query,
            ai_settings=ai_settings,
            centres=initial_documents,
        )
        adjacent_boosted = query_to_documents(
            es_client=self.es_client, index_name=self.index_name, query=with_adjacent_query
        )

        # Merge, sort, return
        merged_documents = merge_documents(initial=initial_documents, adjacent=adjacent_boosted)
        return sort_documents(documents=merged_documents)


class AllElasticsearchRetriever(OpenSearchRetriever):
    """A modified ElasticsearchRetriever that allows retrieving whole documents."""

    chunk_resolution: ChunkResolution = ChunkResolution.largest

    def __init__(self, es_client: Union[Elasticsearch, OpenSearch], **kwargs: Any) -> None:
        # Hack to pass validation before overwrite
        # Partly necessary due to how .with_config() interacts with a retriever
        kwargs["es_client"] = es_client
        kwargs["body_func"] = get_all
        kwargs["document_mapper"] = hit_to_doc
        super().__init__(**kwargs)
        self.body_func = partial(get_all, self.chunk_resolution)

    def _get_relevant_documents(
        self, query: RedboxState, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:  # noqa:ARG002
        # if not self.es_client or not self.document_mapper:
        #     msg = "faulty configuration"
        #     raise ValueError(msg)  # should not happen

        body = self.body_func(query)  # type: ignore

        results = [
            self.document_mapper(hit)
            for hit in scan(client=self.es_client, index=self.index_name, query=body, _source=True)
        ]

        return sorted(results, key=lambda result: result.metadata["index"])


class MetadataRetriever(OpenSearchRetriever):
    """A modified ElasticsearchRetriever that retrieves query metadata without any content"""

    chunk_resolution: ChunkResolution = ChunkResolution.largest

    def __init__(self, es_client: Union[Elasticsearch, OpenSearch], **kwargs: Any) -> None:
        # Hack to pass validation before overwrite
        # Partly necessary due to how .with_config() interacts with a retriever
        kwargs["body_func"] = get_metadata
        kwargs["document_mapper"] = hit_to_doc
        kwargs["es_client"] = es_client
        super().__init__(**kwargs)
        self.body_func = partial(get_metadata, self.chunk_resolution)

    def _get_relevant_documents(
        self, query: RedboxState, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:  # noqa:ARG002
        # if not self.es_client or not self.document_mapper:
        #     msg = "faulty configuration"
        #     raise ValueError(msg)  # should not happen

        body = self.body_func(query)  # type: ignore

        results = [
            self.document_mapper(hit)
            for hit in scan(client=self.es_client, index=self.index_name, query=body, _source=True)
        ]

        return sorted(results, key=lambda result: result.metadata["index"])


class BasicMetadataRetriever(OpenSearchRetriever):
    """A modified MetadataRetriever that retrieves only filename, keyword and description metadata"""

    chunk_resolution: ChunkResolution = ChunkResolution.largest

    def __init__(self, es_client: Union[Elasticsearch, OpenSearch], **kwargs: Any) -> None:
        # Hack to pass validation before overwrite
        # Partly necessary due to how .with_config() interacts with a retriever
        kwargs["body_func"] = get_minimum_metadata
        kwargs["es_client"] = es_client
        super().__init__(**kwargs)
        self.body_func = partial(get_minimum_metadata, self.chunk_resolution)

    def _get_relevant_documents(self, query: RedboxState, *, run_manager: CallbackManagerForRetrieverRun) -> list:  # noqa:ARG002
        # if not self.es_client or not self.document_mapper:
        #     msg = "faulty configuration"
        #     raise ValueError(msg)  # should not happen

        body = self.body_func(query)  # type: ignore
        response = self.es_client.search(index=self.index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        return [hit["_source"] for hit in hits]
