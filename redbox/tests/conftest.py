from collections.abc import Generator
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from _pytest.fixtures import FixtureRequest
from botocore.exceptions import ClientError
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.embeddings.fake import FakeEmbeddings
from opensearchpy import OpenSearch

from redbox.models.chain import AISettings, GeneratedMetadata, RedboxQuery, RedboxState
from redbox.models.settings import Settings
from redbox.retriever import (
    AllElasticsearchRetriever,
    MetadataRetriever,
    OpenSearchRetriever,
    ParameterisedElasticsearchRetriever,
)
from redbox.retriever.retrievers import KnowledgeBaseTabularMetadataRetriever
from redbox.test.data import RedboxChatTestCase
from redbox.transform import bedrock_tokeniser
from tests.retriever.data import (
    ALL_CHUNKS_RETRIEVER_CASES,
    KNOWLEDGE_BASE_CASES,
    METADATA_RETRIEVER_CASES,
    PARAMETERISED_RETRIEVER_CASES,
    TABULAR_RETRIEVER_CASES,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object


# ------------------#
# Clients and tools #
# ------------------#


@pytest.fixture(scope="session")
def env() -> Settings:
    return Settings(django_secret_key="", postgres_password="")


@pytest.fixture(scope="session")
def s3_client(env: Settings) -> S3Client:
    _client = env.s3_client()
    try:
        _client.create_bucket(
            Bucket=env.bucket_name,
            CreateBucketConfiguration={"LocationConstraint": env.aws_region},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
            raise

    return _client


@pytest.fixture(scope="session")
def tokeniser() -> callable:
    return bedrock_tokeniser


@pytest.fixture(scope="session")
def embedding_model_dim() -> int:
    return 1024


@pytest.fixture(scope="session")
def embedding_model(embedding_model_dim: int) -> FakeEmbeddings:
    return FakeEmbeddings(size=embedding_model_dim)


@pytest.fixture(scope="session")
def es_index(env: Settings) -> str:
    return f"{env.elastic_root_index}-chunk"


@pytest.fixture(scope="session")
def es_client(env: Settings) -> OpenSearch:
    return env.elasticsearch_client()


@pytest.fixture(scope="session")
def es_vector_store(es_index: str, embedding_model: FakeEmbeddings, env: Settings) -> OpenSearchVectorSearch:
    # return ElasticsearchStore(
    #     index_name=es_index,
    #     es_connection=es_client,
    #     query_field="text",
    #     vector_query_field=env.embedding_document_field_name,
    #     embedding=embedding_model,
    # )
    return OpenSearchVectorSearch(
        index_name=es_index,
        opensearch_url=env.elastic.collection_endpoint,
        embedding_function=embedding_model,
        query_field="text",
        vector_query_field=env.embedding_document_field_name,
    )


@pytest.fixture(autouse=True, scope="session")
def create_index(env: Settings, es_index: str) -> Generator[None, None, None]:
    es = env.elasticsearch_client()
    if not es.indices.exists(index=es_index):
        es.indices.create(index=es_index)
    yield
    es.indices.delete(index=es_index)


@pytest.fixture(scope="session")
def all_chunks_retriever(env: Settings) -> OpenSearchRetriever:
    return AllElasticsearchRetriever(
        es_client=env.elasticsearch_client(),
        index_name=env.elastic_chunk_alias,
    )


@pytest.fixture(scope="session")
def parameterised_retriever(
    env: Settings, es_client: OpenSearch, es_index: str, embedding_model: FakeEmbeddings
) -> OpenSearchRetriever:
    return ParameterisedElasticsearchRetriever(
        es_client=es_client,
        index_name=es_index,
        embedding_model=embedding_model,
        embedding_field_name=env.embedding_document_field_name,
    )


@pytest.fixture(scope="session")
def metadata_retriever(env: Settings) -> OpenSearchRetriever:
    return MetadataRetriever(es_client=env.elasticsearch_client(), index_name=env.elastic_chunk_alias)


@pytest.fixture(scope="session")
def kb_tabular_metadata_retriever(env: Settings) -> KnowledgeBaseTabularMetadataRetriever:
    return KnowledgeBaseTabularMetadataRetriever(
        es_client=env.elasticsearch_client(), index_name=env.elastic_chunk_alias
    )


@pytest.fixture(scope="session")
def fake_state() -> RedboxState:
    q = RedboxQuery(
        question="But seriously what is AI?",
        s3_keys=[],
        user_uuid=uuid4(),
        chat_history=[{"role": "user", "text": "what is AI?"}, {"role": "ai", "text": "AI is a lie."}],
        ai_settings=AISettings(),
        permitted_s3_keys=[],
    )

    return RedboxState(
        request=q,
    )


# -----#
# Data #
# -----#


@pytest.fixture(params=KNOWLEDGE_BASE_CASES)
def stored_file_knowledge_base(
    request: FixtureRequest, es_vector_store: OpenSearchVectorSearch
) -> Generator[RedboxChatTestCase, None, None]:
    test_case: RedboxChatTestCase = request.param
    doc_ids = es_vector_store.add_documents(test_case.docs)
    yield test_case
    es_vector_store.delete(doc_ids)


@pytest.fixture(params=ALL_CHUNKS_RETRIEVER_CASES)
def stored_file_all_chunks(
    request: FixtureRequest, es_vector_store: OpenSearchVectorSearch
) -> Generator[RedboxChatTestCase, None, None]:
    test_case: RedboxChatTestCase = request.param
    doc_ids = es_vector_store.add_documents(test_case.docs)
    yield test_case
    es_vector_store.delete(doc_ids)


@pytest.fixture(params=PARAMETERISED_RETRIEVER_CASES)
def stored_file_parameterised(
    request: FixtureRequest, es_vector_store: OpenSearchVectorSearch
) -> Generator[RedboxChatTestCase, None, None]:
    test_case: RedboxChatTestCase = request.param
    doc_ids = es_vector_store.add_documents(test_case.docs)
    yield test_case
    es_vector_store.delete(doc_ids)


@pytest.fixture(params=METADATA_RETRIEVER_CASES)
def stored_file_metadata(
    request: FixtureRequest, es_vector_store: OpenSearchVectorSearch
) -> Generator[RedboxChatTestCase, None, None]:
    test_case: RedboxChatTestCase = request.param
    doc_ids = es_vector_store.add_documents(test_case.docs)
    yield test_case
    es_vector_store.delete(doc_ids)


@pytest.fixture(params=TABULAR_RETRIEVER_CASES)
def stored_file_tabular(request: FixtureRequest, es_vector_store: OpenSearchVectorSearch):
    test_case: RedboxChatTestCase = request.param
    doc_ids = es_vector_store.add_documents(test_case.docs)
    yield test_case
    es_vector_store.delete(doc_ids)


@pytest.fixture
def mock_env():
    mock_env = MagicMock(spec=Settings)
    mock_env.unstructured_host = "localhost"
    mock_env.worker_ingest_min_chunk_size = 100
    mock_env.worker_ingest_max_chunk_size = 1000
    mock_env.bucket_name = "test-bucket"
    mock_env.max_retries = 3
    return mock_env


@pytest.fixture
def mock_metadata():
    return GeneratedMetadata(name="test", description="test desc", keywords=["test"])
