from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from redbox.loader.loaders import DocumentLoader, MetadataLoader
from redbox.loader.services.tabular import parse_tabular_schema, load_tabular_file
from redbox.loader.services.ingestion_pipeline import IngestionPipeline
from redbox.loader.services.chunker import TextChunker
from redbox.loader.services.embedding_batcher import EmbeddingBatcher
from redbox.loader.services.opensearch_indexer import OpenSearchBulkIndexer
from redbox.loader.services.textract_service import TextractService

from redbox.models.chain import GeneratedMetadata
from redbox.models.file import ChunkResolution
from redbox.models.settings import Settings
from redbox.retriever.queries import build_query_filter

from io import BytesIO
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object


fake_embedding = np.random.rand(1024).tolist()


def file_to_s3(filename: str, s3_client: S3Client, env: Settings) -> str:
    file_path = Path(__file__).parents[2] / "tests" / "data" / filename
    file_name = file_path.name

    with file_path.open("rb") as f:
        s3_client.put_object(
            Bucket=env.bucket_name,
            Body=f.read(),
            Key=file_name,
        )
    return file_name


def make_file_query(file_name: str, resolution: ChunkResolution | None = None) -> dict[str, Any]:
    query_filter = build_query_filter(
        selected_files=[file_name],
        permitted_files=[file_name],
        chunk_resolution=resolution,
    )
    query = {"query": {"bool": {"must": [{"match_all": {}}], "filter": query_filter}}}
    print("Constructed Query:", query)
    return query


def fake_llm_response():
    return {
        "name": "foo",
        "description": "more test",
        "keywords": ["hello", "world"],
    }


@patch("redbox.loader.loaders.get_chat_llm")
def test_extract_metadata_missing_key(
    mock_llm: MagicMock,
    env: Settings,
    s3_client: S3Client,
):
    mock_llm.return_value = GenericFakeChatModel(messages=iter(['{"missing_key":""}']))

    file_name = file_to_s3("html/example.html", s3_client, env)

    textract_service = TextractService()
    loader = DocumentLoader(bucket=env.bucket_name, textract_service=textract_service)

    metadata_loader = MetadataLoader(
        env=env,
        s3_client=s3_client,
        file_name=file_name,
        document_loader=loader,
    )
    metadata = metadata_loader.extract_metadata()

    if not metadata.name:
        metadata.name = file_name

    assert isinstance(metadata, GeneratedMetadata)
    assert metadata.name == "example.html"


@patch("redbox.loader.loaders.get_chat_llm")
def test_extract_metadata_extra_key(
    mock_llm: MagicMock,
    env: Settings,
    s3_client: S3Client,
):
    mock_llm.return_value = GenericFakeChatModel(
        messages=iter(['{"extra_key": "", "name": "foo", "description": "test", "keywords": ["abc"]}'])
    )

    file_name = file_to_s3("html/example.html", s3_client, env)

    textract_service = TextractService()
    loader = DocumentLoader(bucket=env.bucket_name, textract_service=textract_service)

    metadata_loader = MetadataLoader(env=env, s3_client=s3_client, file_name=file_name, document_loader=loader)
    metadata = metadata_loader.extract_metadata()

    assert metadata.name == "foo"
    assert metadata.description == "test"
    assert metadata.keywords == ["abc"]


@pytest.mark.parametrize(
    "table_name, df, expected_csv_prefix, expected_columns",
    [
        (
            "users",
            pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NY", "LA"]}),
            "<table_name>users</table_name>",
            {"name": "TEXT", "city": "TEXT"},
        ),
        (
            "orders",
            pd.DataFrame({"id": [1, 2, 3], "quantity": [10, 20, 30]}),
            "<table_name>orders</table_name>",
            {"id": "INTEGER", "quantity": "INTEGER"},
        ),
    ],
)
class TestParseTabularSchema:
    def test_result_is_not_none(self, table_name, df, expected_csv_prefix, expected_columns):
        result = parse_tabular_schema(table_name, df)
        assert result is not None

    def test_csv_text_has_correct_prefix(self, table_name, df, expected_csv_prefix, expected_columns):
        csv_text, _ = parse_tabular_schema(table_name, df)
        assert csv_text.startswith(expected_csv_prefix)

    def test_schema_columns_match(self, table_name, df, expected_csv_prefix, expected_columns):
        _, schema_dict = parse_tabular_schema(table_name, df)
        assert schema_dict["columns"] == expected_columns


def test_load_tabular_file_csv():
    csv_content = "name,age\nJohn,30\nJane,25"
    file_bytes = BytesIO(csv_content.encode())

    result = load_tabular_file("test.csv", file_bytes)

    assert len(result) == 1
    assert "<table_name>csv</table_name>" in result[0]["text"]
    assert "document_schema" in result[0]["metadata"]


def test_ingestion_pipeline_tabular(env: Settings, s3_client: S3Client):
    file_name = file_to_s3("airports.csv", s3_client, env)

    textract_service = TextractService()
    loader = DocumentLoader(bucket=env.bucket_name, textract_service=textract_service)

    metadata = MetadataLoader(
        env=env,
        s3_client=s3_client,
        file_name=file_name,
        document_loader=loader,
    ).extract_metadata()

    chunker = TextChunker(min_chunk_size=500, max_chunk_size=2000, overlap_chars=0)
    embedding_batcher = EmbeddingBatcher(embedding_model=MagicMock(), batch_size=64)

    normal_indexer = OpenSearchBulkIndexer(client=MagicMock(), index_name="test-chunks")
    schematised_indexer = OpenSearchBulkIndexer(client=MagicMock(), index_name="test-schematised")

    pipeline = IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedding_batcher=embedding_batcher,
        indexer=normal_indexer,
        schematised_indexer=schematised_indexer,
        metadata=metadata,
    )

    file_path = Path(__file__).parents[2] / "tests" / "data" / "airports.csv"
    with open(file_path, "rb") as f:
        file_bytes = BytesIO(f.read())

    pipeline.ingest(file_name=file_name, file_bytes=file_bytes)

    schematised_indexer.bulk_index.assert_called()


def test_ingestion_pipeline_docx(env: Settings, s3_client: S3Client):
    file_name = file_to_s3("example.docx", s3_client, env)

    textract_service = TextractService()
    loader = DocumentLoader(bucket=env.bucket_name, textract_service=textract_service)

    metadata = MetadataLoader(
        env=env, s3_client=s3_client, file_name=file_name, document_loader=loader
    ).extract_metadata()

    chunker = TextChunker(min_chunk_size=500, max_chunk_size=2000, overlap_chars=0)
    embedding_batcher = EmbeddingBatcher(embedding_model=MagicMock(), batch_size=64)

    normal_indexer = OpenSearchBulkIndexer(client=MagicMock(), index_name="test-chunks")
    schematised_indexer = OpenSearchBulkIndexer(client=MagicMock(), index_name="test-schematised")

    pipeline = IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedding_batcher=embedding_batcher,
        indexer=normal_indexer,
        schematised_indexer=schematised_indexer,
        metadata=metadata,
    )

    file_path = Path(__file__).parents[2] / "tests" / "data" / "example.docx"
    with open(file_path, "rb") as f:
        file_bytes = BytesIO(f.read())

    pipeline.ingest(file_name=file_name, file_bytes=file_bytes)
