import pytest
from io import BytesIO
from unittest.mock import MagicMock

from redbox.loader.textract import (
    TextractChunkLoader,
    LayoutBlock,
)


@pytest.fixture
def loader():
    loader = TextractChunkLoader(
        bucket="test-bucket",
    )

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
    """A minimal chunk returned by the mocked chunker."""
    chunk = MagicMock()
    chunk.text = "chunk text"
    chunk.page_start = 1
    return chunk


class TestRetryLogic:
    def test_non_retryable_error_is_raised(self, loader):
        fn = MagicMock()
        fn.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            loader._retry_textract_request(fn)

    def test_retryable_error_retries(self, loader, monkeypatch):
        monkeypatch.setattr(
            loader,
            "_is_retryable_textract_error",
            lambda e: True,
        )

        fn = MagicMock()
        fn.side_effect = [
            Exception(),
            Exception(),
            {"ok": True},
        ]

        result = loader._retry_textract_request(fn, max_attempts=5)

        assert result == {"ok": True}
        assert fn.call_count == 3


class TestTextractLayoutExtraction:
    def test_get_textract_layout_results_filters_blocks(self, loader):
        loader.textract.get_document_analysis.return_value = {
            "Blocks": [
                {"BlockType": "LAYOUT_TITLE", "Text": "Introduction", "Page": 1},
                {"BlockType": "LAYOUT_TEXT", "Text": "Body", "Page": 1},
                # LINE blocks must be filtered out — they are sub-blocks
                {"BlockType": "LINE", "Text": "ignore", "Page": 1},
            ]
        }

        blocks = loader._get_textract_layout_results("job")

        assert len(blocks) == 2
        assert blocks[0].is_title is True
        assert blocks[1].is_title is False

    def test_paginates_results(self, loader):
        loader.textract.get_document_analysis.side_effect = [
            {
                "Blocks": [{"BlockType": "LAYOUT_TEXT", "Text": "one", "Page": 1}],
                "NextToken": "abc",
            },
            {"Blocks": [{"BlockType": "LAYOUT_TEXT", "Text": "two", "Page": 2}]},
        ]

        blocks = loader._get_textract_layout_results("job")

        assert len(blocks) == 2


class TestPdfFallbacks:
    def test_pdf_layout_uses_textract_when_available(self, loader, fake_pdf):
        expected = [
            LayoutBlock(
                text="hello",
                block_type="LAYOUT_TEXT",
                page_number=1,
                is_title=False,
            )
        ]

        loader._extract_pdf_layout_from_s3 = MagicMock(return_value=expected)

        result = loader._extract_pdf_layout(fake_pdf, "key", "file.pdf")

        assert result == expected

    def test_pdf_layout_falls_back_to_direct_text(self, loader, fake_pdf):
        loader._extract_pdf_layout_from_s3 = MagicMock(side_effect=Exception("boom"))

        # Direct extraction returns a list of page strings; the fallback wraps
        # each non-empty page in a LayoutBlock with is_title=False.
        loader._extract_pdf_text_direct = MagicMock(return_value=["page1", "page2"])

        result = loader._extract_pdf_layout(fake_pdf, "key", "file.pdf")

        assert len(result) == 2
        assert all(isinstance(b, LayoutBlock) for b in result)
        assert result[0].page_number == 1
        assert result[1].page_number == 2
        assert result[0].is_title is False


class TestLayoutBlockExtraction:
    """
    Tests for _extract_layout_blocks, which routes to the correct handler
    based on the file extension and always returns list[LayoutBlock].
    """

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
    def test_extract_layout_blocks_uses_expected_handler(self, loader, filename, method):
        raw_pages = ["page one", "page two"]
        setattr(loader, method, MagicMock(return_value=raw_pages))

        result = loader._extract_layout_blocks(filename, BytesIO(), "key")

        # Handlers return List[str]; _extract_layout_blocks wraps them into
        # LayoutBlock objects before returning.
        assert isinstance(result, list)
        assert len(result) == 2

    def test_extract_layout_blocks_falls_back_to_unstructured(self, loader):
        loader._extract_with_unstructured = MagicMock(return_value=["one"])

        result = loader._extract_layout_blocks("file.md", BytesIO(), "key")

        loader._extract_with_unstructured.assert_called_once()
        assert isinstance(result, list)
        assert len(result) == 1


class TestLazyLoadPdf:
    def test_pdf_uses_chunker(self, loader, fake_pdf, default_chunk):
        loader._extract_pdf_layout = MagicMock(
            return_value=[
                LayoutBlock(
                    text="Intro",
                    block_type="LAYOUT_TITLE",
                    page_number=1,
                    is_title=True,
                )
            ]
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
    def test_tabular_bypasses_chunker(self, loader, monkeypatch):
        # Patch the name as it is imported inside the textract module.
        monkeypatch.setattr(
            "redbox.loader.textract.load_tabular_file",
            lambda *_: [{"text": "row1"}],
        )

        docs = list(loader.lazy_load("file.csv", BytesIO(b"csv")))

        assert len(docs) == 1
        assert docs[0].page_content == "row1"
        # Chunker must not be called for tabular files.
        loader.chunker.chunk.assert_not_called()


class TestEndToEnd:
    def test_pdf_pipeline(self, loader, fake_pdf):
        layout_blocks = [
            LayoutBlock(text="Intro", block_type="LAYOUT_TITLE", page_number=1, is_title=True),
            LayoutBlock(text="Body", block_type="LAYOUT_TEXT", page_number=1, is_title=False),
        ]
        loader._extract_pdf_layout = MagicMock(return_value=layout_blocks)

        # Provide a realistic chunk so the pipeline can yield a Document.
        chunk = MagicMock()
        chunk.text = "Intro\n\nBody"
        chunk.page_start = 1
        loader.chunker.chunk = MagicMock(return_value=[chunk])

        docs = list(loader.lazy_load("file.pdf", fake_pdf))

        assert len(docs) >= 1
        assert "Body" in docs[0].page_content
