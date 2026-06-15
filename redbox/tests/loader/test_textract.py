import pytest
from io import BytesIO
from unittest.mock import MagicMock, patch

from redbox.loader.textract import (
    TextractChunkLoader,
    LayoutBlock,
)
from redbox.models.file import ChunkResolution


@pytest.fixture
def loader():
    """
    Builds a real TextractChunkLoader then immediately replaces the three
    external-facing attributes with mocks so no AWS calls are made and no
    real chunker logic runs unless the test explicitly enables it.
    """
    loader = TextractChunkLoader(bucket="test-bucket")
    loader.textract = MagicMock()
    loader.s3 = MagicMock()
    loader.chunker = MagicMock()
    return loader


@pytest.fixture
def fake_pdf():
    return BytesIO(b"%PDF-fake")


@pytest.fixture
def fake_docx():
    return BytesIO(b"fake-docx")


@pytest.fixture
def default_chunk():
    """Minimal chunk returned by the mocked chunker."""
    chunk = MagicMock()
    chunk.text = "chunk text"
    chunk.page_start = 1
    return chunk


class TestTextractChunkLoaderInit:
    @patch("redbox.loader.textract.boto3.client")
    def test_init_default_parameters(self, mock_boto_client):
        loader = TextractChunkLoader(bucket="test-bucket")

        assert loader.bucket == "test-bucket"
        assert loader.chunker.min_chunk_size == 500
        assert loader.chunker.max_chunk_size == 2000
        assert loader.chunker.overlap_chars == 200
        assert loader.metadata.name == ""
        assert loader.metadata.description == ""
        assert loader.metadata.keywords == []
        mock_boto_client.assert_called()

    @patch("redbox.loader.textract.boto3.client")
    def test_init_custom_parameters(self, mock_boto_client):
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
        assert loader.chunker.min_chunk_size == 300
        assert loader.chunker.max_chunk_size == 3000
        assert loader.chunker.overlap_chars == 100
        assert loader.metadata == custom_metadata

    @patch("redbox.loader.textract.boto3.client")
    def test_init_creates_boto_clients(self, mock_boto_client):
        TextractChunkLoader(bucket="test-bucket")

        # One client for textract, one for s3.
        assert mock_boto_client.call_count >= 2


class TestWaitForLayoutJob:
    """
    _wait_for_job (which polled get_document_text_detection) was removed during
    the Textract LAYOUT migration.  The equivalent is _wait_for_layout_job,
    which polls get_document_analysis.  Throttling / retry logic lives entirely
    in _retry_textract_request (covered by TestRetryLogic), so these tests only
    verify the polling state-machine itself.
    """

    def test_returns_succeeded_immediately(self, loader):
        loader.textract.get_document_analysis.return_value = {"JobStatus": "SUCCEEDED"}

        result = loader._wait_for_layout_job("test-job-id")

        assert result == "SUCCEEDED"
        loader.textract.get_document_analysis.assert_called_once_with(JobId="test-job-id")

    def test_returns_failed_immediately(self, loader):
        loader.textract.get_document_analysis.return_value = {"JobStatus": "FAILED"}

        result = loader._wait_for_layout_job("test-job-id")

        assert result == "FAILED"

    @patch("time.sleep")
    def test_polls_until_succeeded(self, mock_sleep, loader):
        loader.textract.get_document_analysis.side_effect = [
            {"JobStatus": "IN_PROGRESS"},
            {"JobStatus": "IN_PROGRESS"},
            {"JobStatus": "SUCCEEDED"},
        ]

        result = loader._wait_for_layout_job("test-job-id")

        assert result == "SUCCEEDED"
        assert loader.textract.get_document_analysis.call_count == 3
        # One sleep per non-terminal response — two sleeps before the third poll.
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    def test_polls_until_failed(self, mock_sleep, loader):
        loader.textract.get_document_analysis.side_effect = [
            {"JobStatus": "IN_PROGRESS"},
            {"JobStatus": "FAILED"},
        ]

        result = loader._wait_for_layout_job("test-job-id")

        assert result == "FAILED"
        assert loader.textract.get_document_analysis.call_count == 2
        assert mock_sleep.call_count == 1

    def test_propagates_unexpected_api_error(self, loader):
        loader.textract.get_document_analysis.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            loader._wait_for_layout_job("test-job-id")


class TestRetryLogic:
    def test_non_retryable_error_is_raised(self, loader):
        fn = MagicMock()
        fn.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            loader._retry_textract_request(fn)

    def test_retryable_error_retries(self, loader, monkeypatch):
        monkeypatch.setattr(loader, "_is_retryable_textract_error", lambda e: True)

        fn = MagicMock()
        fn.side_effect = [Exception(), Exception(), {"ok": True}]

        result = loader._retry_textract_request(fn, max_attempts=5)

        assert result == {"ok": True}
        assert fn.call_count == 3

    def test_exhausted_retries_raises(self, loader, monkeypatch):
        monkeypatch.setattr(loader, "_is_retryable_textract_error", lambda e: True)

        fn = MagicMock()
        fn.side_effect = Exception("throttled")

        with pytest.raises(Exception, match="throttled"):
            loader._retry_textract_request(fn, max_attempts=3)

        assert fn.call_count == 3


class TestGetTextractLayoutResults:
    """
    The old _get_textract_results read LINE blocks via get_document_text_detection
    and returned List[str] grouped by page.  The new _get_textract_layout_results
    reads LAYOUT_* blocks via get_document_analysis and returns List[LayoutBlock].
    """

    def test_filters_to_layout_blocks_only(self, loader):
        loader.textract.get_document_analysis.return_value = {
            "Blocks": [
                {"BlockType": "LAYOUT_TITLE", "Text": "Introduction", "Page": 1},
                {"BlockType": "LAYOUT_TEXT", "Text": "Body", "Page": 1},
                # LINE / WORD are sub-blocks and must be excluded.
                {"BlockType": "LINE", "Text": "ignore me", "Page": 1},
                {"BlockType": "WORD", "Text": "ignore me too", "Page": 1},
            ]
        }

        blocks = loader._get_textract_layout_results("job-id")

        assert len(blocks) == 2
        assert blocks[0].is_title is True
        assert blocks[0].text == "Introduction"
        assert blocks[1].is_title is False
        assert blocks[1].text == "Body"

    def test_skips_header_footer_figure_blocks(self, loader):
        loader.textract.get_document_analysis.return_value = {
            "Blocks": [
                {"BlockType": "LAYOUT_HEADER", "Text": "My header", "Page": 1},
                {"BlockType": "LAYOUT_FOOTER", "Text": "Page 1 of 10", "Page": 1},
                {"BlockType": "LAYOUT_PAGE_NUMBER", "Text": "1", "Page": 1},
                {"BlockType": "LAYOUT_FIGURE", "Text": "Figure caption", "Page": 1},
                {"BlockType": "LAYOUT_TEXT", "Text": "Real content", "Page": 1},
            ]
        }

        blocks = loader._get_textract_layout_results("job-id")

        assert len(blocks) == 1
        assert blocks[0].text == "Real content"

    def test_skips_empty_text_blocks(self, loader):
        loader.textract.get_document_analysis.return_value = {
            "Blocks": [
                {"BlockType": "LAYOUT_TEXT", "Text": "", "Page": 1},
                {"BlockType": "LAYOUT_TEXT", "Text": "   ", "Page": 1},
                {"BlockType": "LAYOUT_TEXT", "Text": "Has content", "Page": 1},
            ]
        }

        blocks = loader._get_textract_layout_results("job-id")

        assert len(blocks) == 1
        assert blocks[0].text == "Has content"

    def test_returns_empty_list_when_no_blocks(self, loader):
        loader.textract.get_document_analysis.return_value = {"Blocks": []}

        blocks = loader._get_textract_layout_results("job-id")

        assert blocks == []

    def test_paginates_across_multiple_responses(self, loader):
        loader.textract.get_document_analysis.side_effect = [
            {
                "Blocks": [{"BlockType": "LAYOUT_TEXT", "Text": "one", "Page": 1}],
                "NextToken": "abc",
            },
            {
                "Blocks": [{"BlockType": "LAYOUT_TEXT", "Text": "two", "Page": 2}],
            },
        ]

        blocks = loader._get_textract_layout_results("job-id")

        assert len(blocks) == 2
        assert blocks[0].text == "one"
        assert blocks[1].text == "two"

    def test_paginates_through_three_pages(self, loader):
        loader.textract.get_document_analysis.side_effect = [
            {
                "Blocks": [{"BlockType": "LAYOUT_TEXT", "Text": "batch-1", "Page": 1}],
                "NextToken": "tok1",
            },
            {
                "Blocks": [{"BlockType": "LAYOUT_TEXT", "Text": "batch-2", "Page": 1}],
                "NextToken": "tok2",
            },
            {
                "Blocks": [{"BlockType": "LAYOUT_TEXT", "Text": "batch-3", "Page": 1}],
            },
        ]

        blocks = loader._get_textract_layout_results("job-id")

        assert loader.textract.get_document_analysis.call_count == 3
        texts = [b.text for b in blocks]
        assert "batch-1" in texts
        assert "batch-2" in texts
        assert "batch-3" in texts

    def test_propagates_api_error(self, loader):
        loader.textract.get_document_analysis.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            loader._get_textract_layout_results("job-id")


class TestExtractPdfLayoutFromS3:
    """
    The old _extract_pdf_from_s3 used start_document_text_detection and returned
    List[str].  The new _extract_pdf_layout_from_s3 uses start_document_analysis
    with FeatureTypes=["LAYOUT"] and returns List[LayoutBlock].
    """

    def test_success_returns_layout_blocks(self, loader):
        loader.textract.start_document_analysis.return_value = {"JobId": "job-123"}
        loader._wait_for_layout_job = MagicMock(return_value="SUCCEEDED")
        expected = [LayoutBlock(text="Title", block_type="LAYOUT_TITLE", page_number=1, is_title=True)]
        loader._get_textract_layout_results = MagicMock(return_value=expected)

        result = loader._extract_pdf_layout_from_s3("test-bucket", "test.pdf")

        assert result == expected
        loader.textract.start_document_analysis.assert_called_once_with(
            DocumentLocation={"S3Object": {"Bucket": "test-bucket", "Name": "test.pdf"}},
            FeatureTypes=["LAYOUT"],
        )
        loader._wait_for_layout_job.assert_called_once_with("job-123")
        loader._get_textract_layout_results.assert_called_once_with("job-123")

    def test_failed_job_raises_runtime_error(self, loader):
        loader.textract.start_document_analysis.return_value = {"JobId": "job-123"}
        loader._wait_for_layout_job = MagicMock(return_value="FAILED")

        with pytest.raises(RuntimeError, match="Textract LAYOUT job failed"):
            loader._extract_pdf_layout_from_s3("test-bucket", "test.pdf")

    def test_api_error_propagates(self, loader):
        loader.textract.start_document_analysis.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            loader._extract_pdf_layout_from_s3("test-bucket", "test.pdf")


class TestPdfFallbacks:
    def test_uses_textract_when_available(self, loader, fake_pdf):
        expected = [LayoutBlock(text="hello", block_type="LAYOUT_TEXT", page_number=1, is_title=False)]
        loader._extract_pdf_layout_from_s3 = MagicMock(return_value=expected)

        result = loader._extract_pdf_layout(fake_pdf, "key", "file.pdf")

        assert result == expected

    def test_falls_back_to_direct_text_on_error(self, loader, fake_pdf):
        loader._extract_pdf_layout_from_s3 = MagicMock(side_effect=Exception("boom"))
        loader._extract_pdf_text_direct = MagicMock(return_value=["page1", "page2"])

        result = loader._extract_pdf_layout(fake_pdf, "key", "file.pdf")

        assert len(result) == 2
        assert all(isinstance(b, LayoutBlock) for b in result)
        assert result[0].page_number == 1
        assert result[1].page_number == 2
        assert result[0].is_title is False

    def test_fallback_filters_empty_pages(self, loader, fake_pdf):
        loader._extract_pdf_layout_from_s3 = MagicMock(side_effect=Exception("boom"))
        loader._extract_pdf_text_direct = MagicMock(return_value=["real content", "   "])

        result = loader._extract_pdf_layout(fake_pdf, "key", "file.pdf")

        assert len(result) == 1
        assert result[0].text == "real content"


class TestExtractDocx:
    @patch("redbox.loader.textract.partition_docx")
    def test_single_page(self, mock_partition, loader):
        el = MagicMock()
        el.__str__.return_value = "Test content"
        el.metadata.page_number = 1
        mock_partition.return_value = [el]

        result = loader._extract_docx(BytesIO(b"fake docx"))

        assert len(result) == 1
        assert "Test content" in result[0]

    @patch("redbox.loader.textract.partition_docx")
    def test_multiple_pages(self, mock_partition, loader):
        elements = []
        for page in [1, 1, 2, 2, 3]:
            el = MagicMock()
            el.__str__.return_value = f"Content page {page}"
            el.metadata.page_number = page
            elements.append(el)
        mock_partition.return_value = elements

        result = loader._extract_docx(BytesIO(b"fake docx"))

        assert len(result) == 3

    @patch("redbox.loader.textract.partition_docx")
    def test_no_elements_raises(self, mock_partition, loader):
        mock_partition.return_value = []

        with pytest.raises(ValueError, match="unstructured returned no elements"):
            loader._extract_docx(BytesIO(b"fake docx"))

    @patch("redbox.loader.textract.partition_docx")
    def test_partition_error_propagates(self, mock_partition, loader):
        mock_partition.side_effect = Exception("Partition failed")

        with pytest.raises(Exception, match="Partition failed"):
            loader._extract_docx(BytesIO(b"fake docx"))

    @patch("redbox.loader.textract.partition_docx")
    def test_element_without_page_number(self, mock_partition, loader):
        el = MagicMock()
        el.__str__.return_value = "Test content"
        el.metadata.page_number = None
        mock_partition.return_value = [el]

        result = loader._extract_docx(BytesIO(b"fake docx"))

        assert len(result) == 1
        assert "Test content" in result[0]


class TestLayoutBlockExtraction:
    def test_pdf_routes_to_pdf_layout(self, loader, fake_pdf):
        expected = [LayoutBlock(text="Title", block_type="LAYOUT_TITLE", page_number=1, is_title=True)]
        loader._extract_pdf_layout = MagicMock(return_value=expected)

        result = loader._extract_layout_blocks("file.pdf", fake_pdf, "key")

        loader._extract_pdf_layout.assert_called_once_with(fake_pdf, "key", "file.pdf")
        assert result == expected

    @pytest.mark.parametrize(
        ("filename", "method"),
        [
            ("file.docx", "_extract_docx"),
            ("file.pptx", "_extract_pptx"),
        ],
    )
    def test_routes_to_correct_handler(self, loader, filename, method):
        setattr(loader, method, MagicMock(return_value=["page one", "page two"]))

        result = loader._extract_layout_blocks(filename, BytesIO(), "key")

        assert isinstance(result, list)
        assert len(result) == 2

    def test_falls_back_to_unstructured(self, loader):
        loader._extract_with_unstructured = MagicMock(return_value=["one"])

        result = loader._extract_layout_blocks("file.py", BytesIO(), "key")

        loader._extract_with_unstructured.assert_called_once()
        assert isinstance(result, list)
        assert len(result) == 1


class TestLazyLoadPdf:
    def test_pdf_uses_chunker(self, loader, fake_pdf, default_chunk):
        loader._extract_pdf_layout = MagicMock(
            return_value=[LayoutBlock(text="Intro", block_type="LAYOUT_TITLE", page_number=1, is_title=True)]
        )
        loader.chunker.chunk = MagicMock(return_value=[default_chunk])

        docs = list(loader.lazy_load("file.pdf", fake_pdf))

        assert len(docs) == 1
        loader.chunker.chunk.assert_called_once()


class TestLazyLoadNonPdf:
    def test_docx_produces_documents(self, loader, fake_docx, default_chunk):
        loader._extract_layout_blocks = MagicMock(
            return_value=[
                LayoutBlock(text="page one", block_type="LAYOUT_TEXT", page_number=1, is_title=False),
                LayoutBlock(text="page two", block_type="LAYOUT_TEXT", page_number=2, is_title=False),
            ]
        )
        loader.chunker.chunk = MagicMock(return_value=[default_chunk])

        docs = list(loader.lazy_load("file.docx", fake_docx))

        assert docs
        loader.chunker.chunk.assert_called_once()


class TestLazyLoadTabular:
    def test_csv_bypasses_chunker(self, loader, monkeypatch):
        monkeypatch.setattr(
            "redbox.loader.textract.load_tabular_file",
            lambda *_: [{"text": "row1"}],
        )

        docs = list(loader.lazy_load("file.csv", BytesIO(b"csv")))

        assert len(docs) == 1
        assert docs[0].page_content == "row1"
        assert docs[0].metadata["chunk_resolution"] == ChunkResolution.tabular
        loader.chunker.chunk.assert_not_called()

    def test_xlsx_bypasses_chunker(self, loader, monkeypatch):
        monkeypatch.setattr(
            "redbox.loader.textract.load_tabular_file",
            lambda *_: [{"text": "<table_name>Sheet1</table_name>col1,col2\n1,2"}],
        )

        docs = list(loader.lazy_load("file.xlsx", BytesIO(b"excel")))

        assert len(docs) == 1
        assert docs[0].metadata["chunk_resolution"] == ChunkResolution.tabular
        loader.chunker.chunk.assert_not_called()

    def test_empty_tabular_file_yields_no_documents(self, loader, monkeypatch):
        monkeypatch.setattr(
            "redbox.loader.textract.load_tabular_file",
            lambda *_: [],
        )

        docs = list(loader.lazy_load("file.csv", BytesIO(b"csv")))

        assert docs == []
        loader.chunker.chunk.assert_not_called()


class TestEndToEnd:
    def test_pdf_pipeline(self, loader, fake_pdf):
        loader._extract_pdf_layout = MagicMock(
            return_value=[
                LayoutBlock(text="Intro", block_type="LAYOUT_TITLE", page_number=1, is_title=True),
                LayoutBlock(text="Body", block_type="LAYOUT_TEXT", page_number=1, is_title=False),
            ]
        )
        chunk = MagicMock()
        chunk.text = "Intro\n\nBody"
        chunk.page_start = 1
        loader.chunker.chunk = MagicMock(return_value=[chunk])

        docs = list(loader.lazy_load("file.pdf", fake_pdf))

        assert len(docs) >= 1
        assert "Body" in docs[0].page_content
