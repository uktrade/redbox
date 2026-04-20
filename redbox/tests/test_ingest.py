from pathlib import Path

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch
import pytest

from redbox.loader.loaders import MetadataLoader, parse_tabular_schema
from redbox.models.chain import GeneratedMetadata


from redbox.loader.loaders import (
    is_large_pdf,
    split_pdf,
    read_csv_text,
    read_excel_file,
    _pdf_is_image_heavy,
    TextractChunkLoader,
)
from redbox.models.file import ChunkResolution
from redbox.models.settings import Settings
from redbox.retriever.queries import build_query_filter
from io import BytesIO
import pandas as pd

from redbox.test.data import GenericFakeChatModelWithTools


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
    mock_llm_response.return_value = GenericFakeChatModelWithTools(messages=iter(['{"missing_key":""}']))

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
    mock_llm_response.return_value = GenericFakeChatModelWithTools(
        messages=iter(['{"extra_key": "", "name": "foo", "description": "test", "keywords": ["abc"]}'])
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


def test_is_large_pdf_exception():
    mock_exception = Exception("PDF open error")

    with patch("fitz.open") as mock_open:
        mock_open.side_effect = mock_exception

        filebytes = BytesIO(b"mock pdf content")
        is_large, page_count = is_large_pdf("test.pdf", filebytes)

        assert is_large is False
        assert page_count == 0
        mock_open.assert_called_once_with(stream=b"mock pdf content", filetype="pdf")


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
        mock_sub_doc.insert_pdf.assert_called_once_with(mock_doc, from_page=0, to_page=50)


def test_split_pdf_multiple_chunks():
    # Mock a PDF with more pages than chunk size
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 100

    # Mock the sub documents
    mock_sub_docs = [MagicMock(), MagicMock()]
    for doc in mock_sub_docs:
        doc.tobytes.return_value = b"chunk content"
        doc.__len__.return_value = 50

    with patch("fitz.open", side_effect=[mock_doc] + mock_sub_docs):
        filebytes = BytesIO(b"mock pdf content")
        chunks = split_pdf(filebytes, pages_per_chunk=50)

        assert len(chunks) == 2
        # First chunk should have pages 0-49
        mock_sub_docs[0].insert_pdf.assert_called_once_with(mock_doc, from_page=0, to_page=50)
        # Second chunk should have pages 50-99
        mock_sub_docs[1].insert_pdf.assert_called_once_with(mock_doc, from_page=50, to_page=100)


def test_split_pdf_uneven_chunks():
    # Mock a PDF with pages that don't divide evenly by chunk size
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 80

    # Mock the sub documents
    mock_sub_docs = [MagicMock(), MagicMock()]
    mock_sub_docs[0].__len__.return_value = 50
    mock_sub_docs[1].__len__.return_value = 30
    for doc in mock_sub_docs:
        doc.tobytes.return_value = b"chunk content"

    with patch("fitz.open", side_effect=[mock_doc] + mock_sub_docs):
        filebytes = BytesIO(b"mock pdf content")
        chunks = split_pdf(filebytes, pages_per_chunk=50)

        assert len(chunks) == 2
        # First chunk should have pages 0-49
        mock_sub_docs[0].insert_pdf.assert_called_once_with(mock_doc, from_page=0, to_page=50)
        # Second chunk should have pages 50-79
        mock_sub_docs[1].insert_pdf.assert_called_once_with(mock_doc, from_page=50, to_page=80)


def test_split_pdf_empty_pdf():
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 0

    with patch("fitz.open", return_value=mock_doc):
        filebytes = BytesIO(b"mock pdf content")
        chunks = split_pdf(filebytes)

        assert len(chunks) == 0


def test_split_pdf_zero_chunk_size():
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 50
    mock_doc.select.return_value = mock_doc
    mock_doc.tobytes.return_value = b"mock pdf chunk content"
    mock_doc.close = MagicMock()

    with patch("fitz.open", return_value=mock_doc):
        filebytes = BytesIO(b"mock pdf content")
        chunks = split_pdf(filebytes, pages_per_chunk=5)

        assert len(chunks) == 10


def test_split_pdf_skip_empty_sub_doc():
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 50

    mock_sub_doc = MagicMock()
    mock_sub_doc.tobytes.return_value = b"chunk content"
    mock_sub_doc.__len__.return_value = 0

    with patch("fitz.open", side_effect=[mock_doc, mock_sub_doc]):
        filebytes = BytesIO(b"mock pdf content")
        chunks = split_pdf(filebytes)

        assert len(chunks) == 0
        mock_sub_doc.insert_pdf.assert_called_once_with(mock_doc, from_page=0, to_page=50)


def test_pdf_is_image_heavy_image_heavy():
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 3
    mock_pages = [MagicMock() for _ in range(3)]
    mock_pages[0].get_images.return_value = [{"image": "mock1"}]
    mock_pages[1].get_images.return_value = [{"image": "mock2"}]
    mock_pages[2].get_images.return_value = []
    mock_doc.__getitem__.side_effect = mock_pages

    with patch("fitz.open", return_value=mock_doc):
        filebytes = BytesIO(b"mock pdf content")
        is_image_heavy = _pdf_is_image_heavy(filebytes, sample_pages=3)

        assert is_image_heavy is True
        mock_doc.__getitem__.assert_has_calls([((0,),), ((1,),), ((2,),)])
        mock_pages[0].get_images.assert_called_once_with(full=True)
        mock_pages[1].get_images.assert_called_once_with(full=True)
        mock_pages[2].get_images.assert_called_once_with(full=True)


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
        (
            "metrics",
            pd.DataFrame({"score": [1.5, 2.7], "ratio": [0.1, 0.9]}),
            "<table_name>metrics</table_name>",
            {"score": "REAL", "ratio": "REAL"},
        ),
        (
            "products",
            pd.DataFrame({"name": ["apple"], "price": [1.99], "stock": [100], "active": [True]}),
            "<table_name>products</table_name>",
            {"name": "TEXT", "price": "REAL", "stock": "INTEGER", "active": "BOOLEAN"},
        ),
        (
            "labels",
            pd.DataFrame({"label": ["a", "b", "c"]}),
            "<table_name>labels</table_name>",
            {"label": "TEXT"},
        ),
        (
            "my table",
            pd.DataFrame({"x": [1]}),
            "<table_name>my table</table_name>",
            {"x": "INTEGER"},
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

    def test_schema_name_matches_table(self, table_name, df, expected_csv_prefix, expected_columns):
        _, schema_dict = parse_tabular_schema(table_name, df)
        assert schema_dict["name"] == table_name

    def test_schema_columns_match_expected(self, table_name, df, expected_csv_prefix, expected_columns):
        _, schema_dict = parse_tabular_schema(table_name, df)
        assert schema_dict["columns"] == expected_columns

    def test_csv_contains_column_headers(self, table_name, df, expected_csv_prefix, expected_columns):
        csv_text, _ = parse_tabular_schema(table_name, df)
        for col in df.columns:
            assert col in csv_text


class TestParseTabularSchemaErrors:
    @pytest.mark.parametrize(
        "error_target",
        [
            "pandas.core.frame.DataFrame.to_csv",
            "redbox.loader.loaders.TabularSchema",
        ],
    )
    def test_returns_none_on_exception(self, error_target):
        df = pd.DataFrame({"a": [1]})
        with patch(error_target, side_effect=Exception("mocked error")):
            assert parse_tabular_schema("fail_table", df) is None

    def test_empty_dataframe_still_returns_schema(self):
        df = pd.DataFrame({"col1": pd.Series([], dtype="int64")})
        result = parse_tabular_schema("empty_table", df)
        assert result is not None
        _, schema_dict = result
        assert schema_dict["columns"] == {"col1": "INTEGER"}


def test_read_csv_text_valid_file():
    # Create a valid CSV file
    csv_content = "name,age\nJohn,30\nJane,25"
    file_bytes = BytesIO(csv_content.encode())

    result = read_csv_text(file_bytes)

    assert len(result) == 1
    assert "name,age\nJohn,30\nJane,25" in result[0]["text"]
    assert result[0]["metadata"] == {
        "document_schema": {"columns": {"age": "INTEGER", "name": "TEXT"}, "name": "csv", "type": "tabular"}
    }


def test_read_csv_text_pandas_error():
    # Test handling of pandas errors
    file_bytes = BytesIO(b"invalid csv content")

    with patch("pandas.read_csv", side_effect=Exception("CSV parsing error")):
        result = read_csv_text(file_bytes)
        assert result == [{"metadata": {}, "text": "invalid csv content"}]


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
        assert result[0]["metadata"] == {
            "document_schema": {"columns": {"Col1": "INTEGER", "Col2": "INTEGER"}, "name": "sheet1", "type": "tabular"}
        }
        assert "<table_name>sheet2</table_name>" in result[1]["text"]
        assert result[1]["metadata"] == {
            "document_schema": {"columns": {"Col3": "INTEGER", "Col4": "INTEGER"}, "name": "sheet2", "type": "tabular"}
        }


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
        assert result[0]["metadata"] == {
            "document_schema": {"columns": {"Col1": "INTEGER", "Col2": "INTEGER"}, "name": "sheet2", "type": "tabular"}
        }


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
        assert result[0]["metadata"] == {
            "document_schema": {"columns": {"Col1": "INTEGER", "Col2": "INTEGER"}, "name": "sheet2", "type": "tabular"}
        }


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

        assert result == [{"metadata": {}, "text": "mock excel content"}]


class TestTextractChunkLoaderInit:
    @patch("redbox.loader.loaders.boto3.client")
    def test_init_default_parameters(self, mock_boto_client: MagicMock):

        loader = TextractChunkLoader(bucket="test-bucket")

        assert loader.bucket == "test-bucket"
        assert loader.min_chunk_size == 500
        assert loader.max_chunk_size == 2000
        assert loader.overlap_chars == 200
        assert loader.metadata.name == ""
        assert loader.metadata.description == ""
        assert loader.metadata.keywords == []
        mock_boto_client.assert_called()

    @patch("redbox.loader.loaders.boto3.client")
    def test_init_custom_parameters(self, mock_boto_client: MagicMock):

        custom_metadata = MagicMock(name="test.pdf", description="Test file", keywords=["test"])

        loader = TextractChunkLoader(
            bucket="custom-bucket",
            min_chunk_size=300,
            max_chunk_size=3000,
            overlap_chars=100,
            region="eu-west-2",
            metadata=custom_metadata,
        )

        assert loader.bucket == "custom-bucket"
        assert loader.min_chunk_size == 300
        assert loader.max_chunk_size == 3000
        assert loader.overlap_chars == 100
        assert loader.metadata == custom_metadata

    @patch("redbox.loader.loaders.boto3.client")
    def test_init_creates_boto_clients(self, mock_boto_client: MagicMock):

        TextractChunkLoader(bucket="test-bucket")

        assert mock_boto_client.call_count >= 2


class TestWaitForJob:
    @patch("redbox.loader.loaders.boto3.client")
    @patch("time.sleep")
    def test_wait_for_job_success(self, mock_sleep: MagicMock, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.get_document_text_detection.return_value = {"JobStatus": "SUCCEEDED"}
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract

        result = loader._wait_for_job("test-job-id")

        assert result == "SUCCEEDED"
        mock_textract.get_document_text_detection.assert_called_once_with(JobId="test-job-id")

    @patch("redbox.loader.loaders.boto3.client")
    @patch("time.sleep")
    def test_wait_for_job_failed(self, mock_sleep: MagicMock, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.get_document_text_detection.return_value = {"JobStatus": "FAILED"}
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract

        result = loader._wait_for_job("test-job-id")

        assert result == "FAILED"

    @patch("redbox.loader.loaders.boto3.client")
    @patch("time.sleep")
    def test_wait_for_job_polling(self, mock_sleep: MagicMock, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.get_document_text_detection.side_effect = [
            {"JobStatus": "IN_PROGRESS"},
            {"JobStatus": "IN_PROGRESS"},
            {"JobStatus": "SUCCEEDED"},
        ]
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract

        result = loader._wait_for_job("test-job-id")

        assert result == "SUCCEEDED"
        assert mock_textract.get_document_text_detection.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("redbox.loader.loaders.boto3.client")
    def test_wait_for_job_api_error(self, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.get_document_text_detection.side_effect = Exception("API Error")
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract

        with pytest.raises(Exception):
            loader._wait_for_job("test-job-id")


class TestGetTextractResults:
    @patch("redbox.loader.loaders.boto3.client")
    def test_get_textract_results_single_page(self, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.get_document_text_detection.return_value = {
            "Blocks": [
                {"BlockType": "LINE", "Text": "Line 1", "Page": 1},
                {"BlockType": "LINE", "Text": "Line 2", "Page": 1},
            ],
            "NextToken": None,
        }
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract

        result = loader._get_textract_results("test-job-id")

        assert len(result) == 1
        assert "Line 1" in result[0]
        assert "Line 2" in result[0]

    @patch("redbox.loader.loaders.boto3.client")
    def test_get_textract_results_multiple_pages(self, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.get_document_text_detection.side_effect = [
            {
                "Blocks": [
                    {"BlockType": "LINE", "Text": "Page 1 Line 1", "Page": 1},
                    {"BlockType": "LINE", "Text": "Page 1 Line 2", "Page": 1},
                ],
                "NextToken": "token123",
            },
            {
                "Blocks": [
                    {"BlockType": "LINE", "Text": "Page 2 Line 1", "Page": 2},
                    {"BlockType": "LINE", "Text": "Page 2 Line 2", "Page": 2},
                ],
                "NextToken": None,
            },
        ]
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract

        result = loader._get_textract_results("test-job-id")

        assert len(result) == 2
        assert "Page 1 Line 1" in result[0]
        assert "Page 2 Line 1" in result[1]

    @patch("redbox.loader.loaders.boto3.client")
    def test_get_textract_results_no_blocks(self, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.get_document_text_detection.return_value = {
            "Blocks": [],
            "NextToken": None,
        }
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract

        result = loader._get_textract_results("test-job-id")

        assert result == []

    @patch("redbox.loader.loaders.boto3.client")
    def test_get_textract_results_ignores_non_line_blocks(self, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.get_document_text_detection.return_value = {
            "Blocks": [
                {"BlockType": "LINE", "Text": "Valid Line", "Page": 1},
                {"BlockType": "WORD", "Text": "Word Block", "Page": 1},
                {"BlockType": "PAGE", "Text": "Page Block", "Page": 1},
            ],
            "NextToken": None,
        }
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract

        result = loader._get_textract_results("test-job-id")

        assert len(result) == 1
        assert result[0] == "Valid Line"

    @patch("redbox.loader.loaders.boto3.client")
    def test_get_textract_results_pagination(self, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.get_document_text_detection.side_effect = [
            {
                "Blocks": [{"BlockType": "LINE", "Text": "Batch 1", "Page": 1}],
                "NextToken": "token1",
            },
            {
                "Blocks": [{"BlockType": "LINE", "Text": "Batch 2", "Page": 1}],
                "NextToken": "token2",
            },
            {
                "Blocks": [{"BlockType": "LINE", "Text": "Batch 3", "Page": 1}],
                "NextToken": None,
            },
        ]
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract

        result = loader._get_textract_results("test-job-id")

        assert mock_textract.get_document_text_detection.call_count == 3
        assert "Batch 1" in result[0]
        assert "Batch 2" in result[0]
        assert "Batch 3" in result[0]

    @patch("redbox.loader.loaders.boto3.client")
    def test_get_textract_results_api_error(self, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.get_document_text_detection.side_effect = Exception("API Error")
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract

        with pytest.raises(Exception):
            loader._get_textract_results("test-job-id")


class TestExtractDocx:
    @patch("redbox.loader.loaders.boto3.client")
    @patch("redbox.loader.loaders.partition_docx")
    def test_extract_docx_single_page(self, mock_partition: MagicMock, mock_boto_client: MagicMock):

        mock_element = MagicMock()
        mock_element.__str__.return_value = "Test content"
        mock_element.metadata.page_number = 1

        mock_partition.return_value = [mock_element]
        mock_boto_client.return_value = MagicMock()

        loader = TextractChunkLoader(bucket="test-bucket")
        result = loader._extract_docx(BytesIO(b"fake docx content"))

        assert len(result) == 1
        assert "Test content" in result[0]

    @patch("redbox.loader.loaders.boto3.client")
    @patch("redbox.loader.loaders.partition_docx")
    def test_extract_docx_multiple_pages(self, mock_partition: MagicMock, mock_boto_client: MagicMock):

        mock_elements = []
        for page in [1, 1, 2, 2, 3]:
            el = MagicMock()
            el.__str__.return_value = f"Content page {page}"
            el.metadata.page_number = page
            mock_elements.append(el)

        mock_partition.return_value = mock_elements
        mock_boto_client.return_value = MagicMock()

        loader = TextractChunkLoader(bucket="test-bucket")
        result = loader._extract_docx(BytesIO(b"fake docx content"))

        assert len(result) == 3

    @patch("redbox.loader.loaders.boto3.client")
    @patch("redbox.loader.loaders.partition_docx")
    def test_extract_docx_no_elements(self, mock_partition: MagicMock, mock_boto_client: MagicMock):

        mock_partition.return_value = []
        mock_boto_client.return_value = MagicMock()

        loader = TextractChunkLoader(bucket="test-bucket")

        with pytest.raises(ValueError, match="unstructured returned no elements"):
            loader._extract_docx(BytesIO(b"fake docx content"))

    @patch("redbox.loader.loaders.boto3.client")
    @patch("redbox.loader.loaders.partition_docx")
    def test_extract_docx_partition_error(self, mock_partition: MagicMock, mock_boto_client: MagicMock):

        mock_partition.side_effect = Exception("Partition failed")
        mock_boto_client.return_value = MagicMock()

        loader = TextractChunkLoader(bucket="test-bucket")

        with pytest.raises(Exception):
            loader._extract_docx(BytesIO(b"fake docx content"))

    @patch("redbox.loader.loaders.boto3.client")
    @patch("redbox.loader.loaders.partition_docx")
    def test_extract_docx_element_without_page_number(self, mock_partition: MagicMock, mock_boto_client: MagicMock):

        mock_element = MagicMock()
        mock_element.__str__.return_value = "Test content"
        mock_element.metadata.page_number = None

        mock_partition.return_value = [mock_element]
        mock_boto_client.return_value = MagicMock()

        loader = TextractChunkLoader(bucket="test-bucket")
        result = loader._extract_docx(BytesIO(b"fake docx content"))

        assert len(result) == 1
        assert "Test content" in result[0]


class TestExtractPdfFromS3:
    @patch("redbox.loader.loaders.boto3.client")
    def test_extract_pdf_success(self, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.start_document_text_detection.return_value = {"JobId": "job-123"}
        mock_textract.get_document_text_detection.return_value = {"JobStatus": "SUCCEEDED"}
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract
        loader._wait_for_job = MagicMock(return_value="SUCCEEDED")
        loader._get_textract_results = MagicMock(return_value=["Page 1 content"])

        result = loader._extract_pdf_from_s3("test-bucket", "test.pdf")

        assert result == ["Page 1 content"]
        mock_textract.start_document_text_detection.assert_called_once()
        loader._wait_for_job.assert_called_once_with("job-123")
        loader._get_textract_results.assert_called_once_with("job-123")

    @patch("redbox.loader.loaders.boto3.client")
    def test_extract_pdf_job_failed(self, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.start_document_text_detection.return_value = {"JobId": "job-123"}
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract
        loader._wait_for_job = MagicMock(return_value="FAILED")

        with pytest.raises(RuntimeError, match="Textract failed"):
            loader._extract_pdf_from_s3("test-bucket", "test.pdf")

    @patch("redbox.loader.loaders.boto3.client")
    def test_extract_pdf_api_error(self, mock_boto_client: MagicMock):

        mock_textract = MagicMock()
        mock_textract.start_document_text_detection.side_effect = Exception("API Error")
        mock_boto_client.return_value = mock_textract

        loader = TextractChunkLoader(bucket="test-bucket")
        loader.textract = mock_textract

        with pytest.raises(Exception):
            loader._extract_pdf_from_s3("test-bucket", "test.pdf")


class TestLazyLoad:
    @patch("redbox.loader.loaders.boto3.client")
    @patch("redbox.loader.loaders.load_tabular_file")
    def test_lazy_load_csv_file(self, mock_load_tabular: MagicMock, mock_boto_client: MagicMock):

        mock_load_tabular.return_value = [
            {
                "text": "<table_name>test</table_name>col1,col2\n1,2",
                "metadata": {"document_schema": {"name": "test", "type": "tabular", "columns": {}}},
            }
        ]
        mock_boto_client.return_value = MagicMock()

        loader = TextractChunkLoader(bucket="test-bucket")
        docs = list(loader.lazy_load("test.csv", BytesIO(b"csv content")))

        assert len(docs) == 1
        assert docs[0].page_content.startswith("<table_name>")
        assert docs[0].metadata["chunk_resolution"] == ChunkResolution.tabular

    @patch("redbox.loader.loaders.boto3.client")
    @patch("redbox.loader.loaders.load_tabular_file")
    def test_lazy_load_excel_file(self, mock_load_tabular: MagicMock, mock_boto_client: MagicMock):

        mock_load_tabular.return_value = [
            {
                "text": "<table_name>Sheet1</table_name>col1,col2\n1,2",
                "metadata": {"document_schema": {"name": "Sheet1"}},
            }
        ]
        mock_boto_client.return_value = MagicMock()

        loader = TextractChunkLoader(bucket="test-bucket")
        docs = list(loader.lazy_load("test.xlsx", BytesIO(b"excel content")))

        assert len(docs) == 1
        assert docs[0].metadata["chunk_resolution"] == ChunkResolution.tabular

    @patch("redbox.loader.loaders.boto3.client")
    @patch("redbox.loader.loaders.load_tabular_file")
    def test_lazy_load_empty_tabular_file(self, mock_load_tabular: MagicMock, mock_boto_client: MagicMock):

        mock_load_tabular.return_value = []
        mock_boto_client.return_value = MagicMock()

        loader = TextractChunkLoader(bucket="test-bucket")
        docs = list(loader.lazy_load("test.csv", BytesIO(b"csv content")))

        assert len(docs) == 0
