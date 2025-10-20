import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch, call

import pytest
from _pytest.monkeypatch import MonkeyPatch
from opensearchpy import OpenSearch
from opensearchpy.helpers import scan
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_community.vectorstores import OpenSearchVectorSearch

from redbox.chains.ingest import document_loader, ingest_from_loader
from redbox.loader import ingester
from redbox.loader.ingester import ingest_file
from redbox.loader.loaders import MetadataLoader, UnstructuredChunkLoader
from redbox.models.chain import GeneratedMetadata


from redbox.loader.loaders import (
    is_large_pdf,
    split_pdf,
    read_csv_text,
    read_excel_file,
)
from redbox.models.file import ChunkResolution
from redbox.models.settings import Settings
from redbox.retriever.queries import build_query_filter
from io import BytesIO
import pandas as pd


if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object

import numpy as np

fake_embedding = np.random.rand(1024).tolist()


def file_to_s3(filename: str, s3_client: S3Client, env: Settings) -> str:
    file_path = Path(__file__).parents[2] / "tests" / "data" / filename
    file_name = file_path.name
    file_type = file_path.suffix

    with file_path.open("rb") as f:
        s3_client.put_object(
            Bucket=env.bucket_name,
            Body=f.read(),
            Key=file_name,
            Tagging=f"file_type={file_type}",
        )

    return file_name


def make_file_query(file_name: str, resolution: ChunkResolution | None = None) -> dict[str, Any]:
    query_filter = build_query_filter(
        selected_files=[file_name],
        permitted_files=[file_name],
        chunk_resolution=resolution,
    )
    query = {"query": {"bool": {"must": [{"match_all": {}}], "filter": query_filter}}}
    print("Constructed Query:", query)  # Debugging: Print the query
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
    requests_mock,
):
    mock_llm_response = mock_llm.return_value
    mock_llm_response.status_code = 200
    mock_llm_response.return_value = GenericFakeChatModel(messages=iter(['{"missing_key":""}']))

    requests_mock.post(
        f"http://{env.unstructured_host}:8000/general/v0/general",
        json=[{"text": "hello", "metadata": {}}],
    )

    """
    LLM replies but without one of the keys
    """

    # Upload file
    file_name = file_to_s3("html/example.html", s3_client, env)

    metadata_loader = MetadataLoader(env=env, s3_client=s3_client, file_name=file_name)
    metadata = metadata_loader.extract_metadata()

    if not metadata.name:
        metadata.name = file_name

    assert metadata == GeneratedMetadata(name="example.html")


@patch("redbox.loader.loaders.get_chat_llm")
def test_extract_metadata_extra_key(
    mock_llm: MagicMock,
    env: Settings,
    s3_client: S3Client,
    requests_mock,
):
    mock_llm_response = mock_llm.return_value
    mock_llm_response.status_code = 200
    mock_llm_response.return_value = GenericFakeChatModel(
        messages=iter(['{"extra_key": "", "name": "foo", "description": "test", "keywords": ["abc"]}'])
    )

    requests_mock.post(
        f"http://{env.unstructured_host}:8000/general/v0/general",
        json=[{"text": "hello", "metadata": {"filename": "something"}}],
    )

    """
    LLM replies with an extra key
    """

    # Upload file
    file_name = file_to_s3("html/example.html", s3_client, env)

    metadata_loader = MetadataLoader(env=env, s3_client=s3_client, file_name=file_name)
    metadata = metadata_loader.extract_metadata()

    assert metadata is not None
    assert metadata.name == "foo"
    assert metadata.description == "test"
    assert metadata.keywords == ["abc"]


@patch("redbox.loader.loaders.get_chat_llm")
@patch("redbox.loader.loaders.requests.post")
def test_document_loader(
    mock_post: MagicMock,
    mock_llm: MagicMock,
    s3_client: S3Client,
    env: Settings,
):
    """
    Given that I have written a text File to s3
    When I call document_loader
    I Expect to see this file chunked and embedded if appropriate
    """
    # Mock call to Unstructured
    mock_response = mock_post.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "type": "CompositeElement",
            "element_id": "1c493e1166a6e59ebe9e054c9c6c03db",
            "text": "Routing enables us to create bespoke responses according to user intent. Examples include:\n\n* RAG\n* Summarization\n* Plain chat",
            "metadata": {
                "languages": ["eng"],
                "orig_elements": "eJwVjsFOwzAQRH9l5SMiCEVtSXrjxI0D4lZVaGNPgtV4HdlrVKj679iXXe3szOidbgYrAkS/vDNHMnY89NPezd2Be9ft+mHfjfM4dOwwvDg4O++ezSOZAGXHyjVzMyvLUnhBrtfJQBZzvleP4qqtM8WiXhaC8LQiU8mkkWwCK2hC3uIFlNqWXN9sbUyuBaqrZCTyopXwiXDlsLUGL3YtDkd6oI/XtzpzCYET+z9WH6UK28peyH6zNlr93dBI3jml6vjBZ0O7n/8BhxNVfA==",
                "filename": "example.html",
                "filetype": "text/html",
            },
        }
    ]

    mock_llm_response = mock_llm.return_value
    mock_llm_response.status_code = 200
    mock_llm_response.return_value = GenericFakeChatModel(messages=iter([json.dumps(fake_llm_response())]))

    # Upload file
    file = file_to_s3("html/example.html", s3_client, env)

    metadata_loader = MetadataLoader(env=env, s3_client=s3_client, file_name=file)
    metadata = metadata_loader.extract_metadata()

    loader = UnstructuredChunkLoader(
        chunk_resolution=ChunkResolution.normal,
        env=env,
        min_chunk_size=env.worker_ingest_min_chunk_size,
        max_chunk_size=env.worker_ingest_max_chunk_size,
        metadata=metadata,
    )

    # Call loader
    loader = document_loader(loader, s3_client, env)
    chunks = list(loader.invoke(file))

    assert len(chunks) > 0

    # Verify that metadata has been attached to object
    for chuck in chunks:
        llm_response = fake_llm_response()
        assert chuck.metadata["name"] == llm_response["name"]
        assert chuck.metadata["description"] == llm_response["description"]
        assert chuck.metadata["keywords"] == llm_response["keywords"]


@patch("redbox.loader.loaders.get_chat_llm")
@patch("redbox.loader.loaders.requests.post")
@pytest.mark.parametrize(
    "resolution, has_embeddings",
    [
        (ChunkResolution.largest, False),
        (ChunkResolution.normal, True),
    ],
)
def test_ingest_from_loader(
    mock_post: MagicMock,
    mock_llm: MagicMock,
    resolution: ChunkResolution,
    has_embeddings: bool,
    monkeypatch: MonkeyPatch,
    es_client: OpenSearch,
    es_vector_store: OpenSearchVectorSearch,
    es_index: str,
    s3_client: S3Client,
    env: Settings,
):
    """
    Given that I have written a text File to s3
    When I call ingest_from_loader
    I Expect to see this file chunked and embedded if appropriate
    """

    # Mock call to Unstructured
    mock_response = mock_post.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "type": "CompositeElement",
            "element_id": "1c493e1166a6e59ebe9e054c9c6c03db",
            "text": "Routing enables us to create bespoke responses according to user intent. Examples include:\n\n* RAG\n* Summarization\n* Plain chat",
            "metadata": {
                "languages": ["eng"],
                "orig_elements": "eJwVjsFOwzAQRH9l5SMiCEVtSXrjxI0D4lZVaGNPgtV4HdlrVKj679iXXe3szOidbgYrAkS/vDNHMnY89NPezd2Be9ft+mHfjfM4dOwwvDg4O++ezSOZAGXHyjVzMyvLUnhBrtfJQBZzvleP4qqtM8WiXhaC8LQiU8mkkWwCK2hC3uIFlNqWXN9sbUyuBaqrZCTyopXwiXDlsLUGL3YtDkd6oI/XtzpzCYET+z9WH6UK28peyH6zNlr93dBI3jml6vjBZ0O7n/8BhxNVfA==",
                "filename": "example.html",
                "filetype": "text/html",
            },
        }
    ]

    mock_llm_response = mock_llm.return_value
    mock_llm_response.status_code = 200
    mock_llm_response.return_value = GenericFakeChatModel(messages=iter([json.dumps(fake_llm_response())]))

    # Upload file and call
    file_name = file_to_s3(filename="html/example.html", s3_client=s3_client, env=env)

    # Extract metadata
    metadata_loader = MetadataLoader(env=env, s3_client=s3_client, file_name=file_name)
    metadata = metadata_loader.extract_metadata()

    loader = UnstructuredChunkLoader(
        chunk_resolution=resolution,
        env=env,
        min_chunk_size=env.worker_ingest_min_chunk_size,
        max_chunk_size=env.worker_ingest_max_chunk_size,
        metadata=metadata,
    )

    mapping = {"properties": {"embedding": {"type": "knn_vector", "dimension": 1024}}}

    # Check if the index already exists and delete it if it does
    if es_client.indices.exists(index="my_index"):
        es_client.indices.delete(index="my_index")

    es_client.indices.create(index="my_index", body={"mappings": mapping})

    # Mock embeddings
    monkeypatch.setattr(ingester, "get_embeddings", lambda _: fake_embedding)

    ingest_chain = ingest_from_loader(loader=loader, s3_client=s3_client, vectorstore=es_vector_store, env=env)

    _ = ingest_chain.invoke(file_name)

    # Test it's written to Elastic
    file_query = make_file_query(file_name=file_name, resolution=resolution)

    chunks = list(scan(client=es_client, index=f"{es_index}-current", query=file_query))
    assert len(chunks) > 0

    # Debugging: Print chunks to inspect the output
    for chunk in chunks:
        print(chunk)

    def get_metadata(chunk: dict) -> dict:
        return chunk["_source"]["metadata"]

    # Verify that metadata has been attached to object
    if has_embeddings:
        for chunk in chunks:
            metadata = get_metadata(chunk)
            assert metadata["name"] == fake_llm_response()["name"]
            assert metadata["description"] == fake_llm_response()["description"]
            assert metadata["keywords"] == fake_llm_response()["keywords"]

    if has_embeddings:
        embeddings = chunks[0]["_source"].get("vector_field")
        print("Embeddings:", embeddings)  # Debugging: Print embeddings to inspect the output
        assert embeddings is not None
        assert len(embeddings) > 0

    # Teardown
    es_client.delete_by_query(index=es_index, body=file_query)


@patch("redbox.loader.loaders.get_chat_llm")
@patch("redbox.loader.loaders.requests.post")
@pytest.mark.parametrize(
    ("filename", "is_complete", "mock_json"),
    [
        (
            "html/example.html",
            True,
            [
                {
                    "type": "CompositeElement",
                    "element_id": "1c493e1166a6e59ebe9e054c9c6c03db",
                    "text": "Routing enables us to create bespoke responses according to user intent. Examples include:\n\n* RAG\n* Summarization\n* Plain chat",
                    "metadata": {
                        "languages": ["eng"],
                        "orig_elements": "eJwVjsFOwzAQRH9l5SMiCEVtSXrjxI0D4lZVaGNPgtV4HdlrVKj679iXXe3szOidbgYrAkS/vDNHMnY89NPezd2Be9ft+mHfjfM4dOwwvDg4O++ezSOZAGXHyjVzMyvLUnhBrtfJQBZzvleP4qqtM8WiXhaC8LQiU8mkkWwCK2hC3uIFlNqWXN9sbUyuBaqrZCTyopXwiXDlsLUGL3YtDkd6oI/XtzpzCYET+z9WH6UK28peyH6zNlr93dBI3jml6vjBZ0O7n/8BhxNVfA==",
                        "filename": "example.html",
                        "filetype": "text/html",
                    },
                }
            ],
        ),
        ("html/corrupt.html", False, None),
    ],
)
def test_ingest_file(
    mock_post: MagicMock,
    mock_llm: MagicMock,
    es_client: OpenSearch,
    s3_client: S3Client,
    monkeypatch: MonkeyPatch,
    env: Settings,
    es_index: str,
    filename: str,
    is_complete: bool,
    mock_json: list | None,
):
    """
    Given that I have written a text File to s3
    When I call ingest_file
    I Expect to see this file to be:
    1. chunked
    2. written to OpenSearch
    """
    # Mock call to Unstructured
    mock_response = mock_post.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = mock_json

    # Mock embeddings
    monkeypatch.setattr(ingester, "get_embeddings", lambda _: FakeEmbeddings(size=1024))

    # Upload file and call
    filename = file_to_s3(filename=filename, s3_client=s3_client, env=env)

    # Mock llm
    mock_llm_response = mock_llm.return_value
    mock_llm_response.status_code = 200
    mock_llm_response.return_value = GenericFakeChatModel(messages=iter([json.dumps(fake_llm_response())]))

    try:
        res = ingest_file(filename)
    except Exception as e:
        print(f"Exception occurred: {e}")
        raise

    if not is_complete:
        assert isinstance(res, str)
    else:
        assert res is None

        # Test it's written to Elastic
        file_query = make_file_query(file_name=filename)

        try:
            chunks = list(scan(client=es_client, index=f"{es_index}-current", query=file_query, _source=True))
        except Exception as e:
            print(f"Exception during scanning: {e}")
            raise

        assert len(chunks) > 0

        def get_metadata(chunk: dict) -> dict:
            return chunk["_source"]["metadata"]

        # Verify that metadata has been attached to document.
        for chunk in chunks:
            metadata = get_metadata(chunk)
            llm_response = fake_llm_response()
            assert metadata["name"] == llm_response["name"]
            assert metadata["description"] == llm_response["description"]
            assert metadata["keywords"] == llm_response["keywords"]

        def get_chunk_resolution(chunk: dict) -> str:
            return chunk["_source"]["metadata"]["chunk_resolution"]

        normal_resolution = [chunk for chunk in chunks if get_chunk_resolution(chunk) == "normal"]
        largest_resolution = [chunk for chunk in chunks if get_chunk_resolution(chunk) == "largest"]

        assert len(normal_resolution) > 0
        assert len(largest_resolution) > 0

        # Teardown
        es_client.delete_by_query(index=es_index, body=file_query)


def test_is_large_pdf_small_file():
    # Create a mock PDF with fewer pages than the threshold
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 50  # Less than default threshold of 150

    with patch("fitz.open", return_value=mock_doc):
        filebytes = BytesIO(b"mock pdf content")
        is_large, page_count = is_large_pdf("test.pdf", filebytes)

        assert is_large is False
        assert page_count == 50


def test_is_large_pdf_large_file():
    # Create a mock PDF with more pages than the threshold
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 200  # More than default threshold of 150

    with patch("fitz.open", return_value=mock_doc):
        filebytes = BytesIO(b"mock pdf content")
        is_large, page_count = is_large_pdf("test.pdf", filebytes)

        assert is_large is True
        assert page_count == 200


def test_is_large_pdf_custom_threshold():
    # Test with custom threshold
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 75

    with patch("fitz.open", return_value=mock_doc):
        filebytes = BytesIO(b"mock pdf content")
        is_large, page_count = is_large_pdf("test.pdf", filebytes, page_threshold=50)

        assert is_large is True
        assert page_count == 75


def test_is_large_pdf_non_pdf_file():
    # Test with a non-PDF file
    filebytes = BytesIO(b"mock content")
    is_large, page_count = is_large_pdf("test.txt", filebytes)

    assert is_large is False
    assert page_count == 0


def test_split_pdf_single_chunk():
    # Mock a PDF with fewer pages than chunk size
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 50

    # Mock the sub document
    mock_sub_doc = MagicMock()
    mock_sub_doc.tobytes.return_value = b"chunk content"
    mock_sub_doc.__len__.return_value = 50

    with patch("fitz.open", side_effect=[mock_doc, mock_sub_doc]):
        filebytes = BytesIO(b"mock pdf content")
        chunks = split_pdf(filebytes)

        assert len(chunks) == 1
        mock_sub_doc.insert_pdf.assert_has_calls([call(mock_doc, from_page=i, to_page=i) for i in range(50)])


def test_split_pdf_multiple_chunks():
    # Mock a PDF with more pages than chunk size
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 100

    # Mock the sub documents
    mock_sub_docs = [MagicMock(), MagicMock()]
    for doc in mock_sub_docs:
        doc.tobytes.return_value = b"chunk content"
        doc.__len__.return_value = 100

    with patch("fitz.open", side_effect=[mock_doc] + mock_sub_docs):
        filebytes = BytesIO(b"mock pdf content")
        chunks = split_pdf(filebytes, pages_per_chunk=50)

        assert len(chunks) == 2
        # First chunk should have pages 0-49
        mock_sub_docs[0].insert_pdf.assert_has_calls([call(mock_doc, from_page=i, to_page=i) for i in range(50)])
        # Second chunk should have pages 50-99
        mock_sub_docs[1].insert_pdf.assert_has_calls([call(mock_doc, from_page=i, to_page=i) for i in range(50, 100)])


def test_split_pdf_uneven_chunks():
    # Mock a PDF with pages that don't divide evenly by chunk size
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 80

    # Mock the sub documents
    mock_sub_docs = [MagicMock(), MagicMock()]
    for doc in mock_sub_docs:
        doc.tobytes.return_value = b"chunk content"
        doc.__len__.return_value = 80

    with patch("fitz.open", side_effect=[mock_doc] + mock_sub_docs):
        filebytes = BytesIO(b"mock pdf content")
        chunks = split_pdf(filebytes, pages_per_chunk=50)

        assert len(chunks) == 2
        # First chunk should have pages 0-49
        mock_sub_docs[0].insert_pdf.assert_has_calls([call(mock_doc, from_page=i, to_page=i) for i in range(50)])
        # Second chunk should have pages 50-79
        mock_sub_docs[1].insert_pdf.assert_has_calls([call(mock_doc, from_page=i, to_page=i) for i in range(50, 80)])


def test_read_csv_text_valid_file():
    # Create a valid CSV file
    csv_content = "name,age\nJohn,30\nJane,25"
    file_bytes = BytesIO(csv_content.encode())

    result = read_csv_text(file_bytes)

    assert len(result) == 1
    assert "name,age\nJohn,30\nJane,25" in result[0]["text"]
    assert result[0]["metadata"] == {}


def test_read_csv_text_pandas_error():
    # Test handling of pandas errors
    file_bytes = BytesIO(b"invalid csv content")

    with patch("pandas.read_csv", side_effect=Exception("CSV parsing error")):
        result = read_csv_text(file_bytes)
        assert result is None


def test_read_excel_file_multiple_sheets():
    # Mock pandas read_excel to return multiple sheets
    sheet1 = pd.DataFrame({"Col1": [1, 2], "Col2": [3, 4]})
    sheet2 = pd.DataFrame({"Col3": [5, 6], "Col4": [7, 8]})
    mock_sheets = {"Sheet1": sheet1, "Sheet2": sheet2}

    with patch("pandas.read_excel", return_value=mock_sheets):
        file_bytes = BytesIO(b"mock excel content")
        result = read_excel_file(file_bytes)

        assert len(result) == 2
        assert "<table_name>sheet1</table_name>" in result[0]["text"]
        assert "<table_name>sheet2</table_name>" in result[1]["text"]


def test_read_excel_file_empty_sheet():
    # Mock pandas read_excel to return one empty and one valid sheet
    sheet1 = pd.DataFrame  # Empty sheet
    sheet2 = pd.DataFrame({"Col1": [1, 2], "Col2": [3, 4]})
    mock_sheets = {"Sheet1": sheet1, "Sheet2": sheet2}

    with patch("pandas.read_excel", return_value=mock_sheets):
        file_bytes = BytesIO(b"mock excel content")
        result = read_excel_file(file_bytes)

        assert len(result) == 1
        assert "<table_name>sheet2</table_name>" in result[0]["text"]


def test_read_excel_file_sheet_error():
    # Mock pandas read_excel to return sheet with error
    sheet1 = MagicMock()
    sheet1.to_csv.side_effect = Exception("Sheet conversion error")
    sheet2 = pd.DataFrame({"Col1": [1, 2], "Col2": [3, 4]})
    mock_sheets = {"Sheet1": sheet1, "Sheet2": sheet2}

    with patch("pandas.read_excel", return_value=mock_sheets):
        file_bytes = BytesIO(b"mock excel content")
        result = read_excel_file(file_bytes)

        assert len(result) == 1
        assert "<table_name>sheet2</table_name>" in result[0]["text"]


def test_read_excel_file_all_empty_sheets():
    # Mock pandas read_excel to return all empty sheets
    sheet1 = pd.DataFrame
    sheet2 = pd.DataFrame
    mock_sheets = {"Sheet1": sheet1, "Sheet2": sheet2}

    with patch("pandas.read_excel", return_value=mock_sheets):
        file_bytes = BytesIO(b"mock excel content")
        result = read_excel_file(file_bytes)

        assert result is None


def test_read_excel_file_pandas_error():
    # Mock pandas read_excel to raise exception
    with patch("pandas.read_excel", side_effect=Exception("Excel parsing error")):
        file_bytes = BytesIO(b"mock excel content")
        result = read_excel_file(file_bytes)

        assert result is None
