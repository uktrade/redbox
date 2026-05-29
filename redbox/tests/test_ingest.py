from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from redbox.loader.loaders import DocumentLoader, MetadataLoader
from redbox.loader.services.tabular import load_tabular_file
from redbox.loader.services.ingestion_pipeline import IngestionPipeline
from redbox.loader.services.chunker import TextChunker
from redbox.loader.services.embedding_batcher import EmbeddingBatcher
from redbox.loader.services.opensearch_indexer import OpenSearchBulkIndexer
from redbox.loader.services.textract_service import TextractService

from redbox.models.chain import GeneratedMetadata
from redbox.models.settings import Settings

from io import BytesIO

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object


def make_test_csv() -> BytesIO:
    csv_content = """id,name,city,country
1,Adak Island Airport,Adak,United States
2,London Heathrow,London,United Kingdom"""
    return BytesIO(csv_content.encode())


def make_test_docx_content() -> BytesIO:
    return BytesIO(b"dummy docx content - this is a test document with some content.")


def fake_llm_response():
    return {"name": "test file", "description": "test description", "keywords": ["test"]}


@patch("redbox.loader.loaders.get_chat_llm")
def test_extract_metadata_missing_key(mock_llm, env: Settings, s3_client: S3Client):
    mock_llm.return_value = GenericFakeChatModel(messages=iter(['{"missing_key":""}']))

    file_name = "test-missing-key.html"
    s3_client.put_object(Bucket=env.bucket_name, Key=file_name, Body=b"dummy content")

    textract_service = TextractService()
    loader = DocumentLoader(bucket=env.bucket_name, textract_service=textract_service)

    metadata_loader = MetadataLoader(env=env, s3_client=s3_client, file_name=file_name, document_loader=loader)
    metadata = metadata_loader.extract_metadata()

    assert isinstance(metadata, GeneratedMetadata)


@patch("redbox.loader.loaders.get_chat_llm")
def test_extract_metadata_success(mock_llm, env: Settings, s3_client: S3Client):
    mock_llm.return_value = GenericFakeChatModel(
        messages=iter(['{"name": "foo", "description": "test", "keywords": ["abc"]}'])
    )

    file_name = "test-success.html"
    s3_client.put_object(Bucket=env.bucket_name, Key=file_name, Body=b"dummy content")

    textract_service = TextractService()
    loader = DocumentLoader(bucket=env.bucket_name, textract_service=textract_service)

    metadata_loader = MetadataLoader(env=env, s3_client=s3_client, file_name=file_name, document_loader=loader)
    metadata = metadata_loader.extract_metadata()

    assert metadata.name == "foo"
    assert metadata.description == "test"


def test_load_tabular_file_csv():
    result = load_tabular_file("airports.csv", make_test_csv())

    assert len(result) == 1
    assert "<table_name>csv</table_name>" in result[0]["text"]
    assert "document_schema" in result[0]["metadata"]


def test_ingestion_pipeline_tabular(env: Settings, s3_client: S3Client):
    file_name = "airports.csv"
    s3_client.put_object(Bucket=env.bucket_name, Key=file_name, Body=make_test_csv().getvalue())

    textract_service = TextractService()
    loader = DocumentLoader(bucket=env.bucket_name, textract_service=textract_service)

    metadata = MetadataLoader(
        env=env, s3_client=s3_client, file_name=file_name, document_loader=loader
    ).extract_metadata()

    chunker = TextChunker(min_chunk_size=100, max_chunk_size=2000, overlap_chars=0)

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

    pipeline.ingest(file_name=file_name, file_bytes=make_test_csv())

    schematised_indexer.bulk_index.assert_called_once()


def test_ingestion_pipeline_docx(env: Settings, s3_client: S3Client):
    file_name = "test-document.docx"
    s3_client.put_object(Bucket=env.bucket_name, Key=file_name, Body=make_test_docx_content().getvalue())

    textract_service = TextractService()
    loader = DocumentLoader(bucket=env.bucket_name, textract_service=textract_service)

    metadata = MetadataLoader(
        env=env, s3_client=s3_client, file_name=file_name, document_loader=loader
    ).extract_metadata()

    chunker = TextChunker(min_chunk_size=500, max_chunk_size=2000, overlap_chars=0)
    embedding_batcher = EmbeddingBatcher(embedding_model=MagicMock(), batch_size=64)

    normal_indexer = OpenSearchBulkIndexer(client=MagicMock(), index_name="test-chunks")

    pipeline = IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedding_batcher=embedding_batcher,
        indexer=normal_indexer,
        schematised_indexer=None,
        metadata=metadata,
    )

    pipeline.ingest(file_name=file_name, file_bytes=make_test_docx_content())

    normal_indexer.bulk_index.assert_called()
