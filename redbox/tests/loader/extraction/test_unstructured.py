import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock

from redbox.loader.extraction.unstructured import UnstructuredService


def make_element(text: str, page_number: int | None = None, slide_number: int | None = None) -> MagicMock:
    el = MagicMock()
    el.__str__ = lambda self: text
    el.metadata.page_number = page_number
    el.metadata.slide_number = slide_number
    return el


def make_elements(*specs: tuple) -> list:
    """specs: (text, page_number) tuples."""
    return [make_element(text, page) for text, page in specs]


DUMMY_BYTES = BytesIO(b"fake file content")
SERVICE = UnstructuredService()


class TestExtractDocx:
    @pytest.mark.parametrize(
        "elements, expected_pages",
        [
            # single page
            (make_elements(("Hello", 1), ("World", 1)), ["Hello\nWorld"]),
            # two pages
            (make_elements(("Page1", 1), ("Page2", 2)), ["Page1", "Page2"]),
            # three pages, multiple elements per page
            (make_elements(("A", 1), ("B", 1), ("C", 2), ("D", 3)), ["A\nB", "C", "D"]),
            # elements with no page metadata all land on one page
            (make_elements(("X", None), ("Y", None)), ["X\nY"]),
        ],
    )
    @patch("redbox.loader.extraction.unstructured.partition_docx")
    def test_page_grouping(self, mock_partition, elements, expected_pages):
        mock_partition.return_value = elements
        result = SERVICE._extract_docx(DUMMY_BYTES)
        assert result == expected_pages

    @patch("redbox.loader.extraction.unstructured.partition_docx", return_value=[])
    def test_raises_on_no_elements(self, _mock):
        with pytest.raises(ValueError, match="no elements"):
            SERVICE._extract_docx(DUMMY_BYTES)

    @patch("redbox.loader.extraction.unstructured.partition_docx", return_value=[make_element("   ", 1)])
    def test_raises_on_no_readable_text(self, _mock):
        with pytest.raises(ValueError, match="no readable text"):
            SERVICE._extract_docx(DUMMY_BYTES)

    @patch("redbox.loader.extraction.unstructured.partition_docx", side_effect=RuntimeError("corrupt"))
    def test_propagates_partition_exceptions(self, _mock):
        with pytest.raises(RuntimeError, match="corrupt"):
            SERVICE._extract_docx(DUMMY_BYTES)

    @patch("redbox.loader.extraction.unstructured.partition_docx")
    def test_seeks_to_zero_before_partition(self, mock_partition):
        mock_partition.return_value = make_elements(("text", 1))
        buf = BytesIO(b"data")
        buf.read()  # advance position
        SERVICE._extract_docx(buf)
        mock_partition.assert_called_once()
        assert mock_partition.call_args.kwargs["file"].read() == b"data"


class TestExtractPptx:
    @pytest.mark.parametrize(
        "elements, expected_pages",
        [
            # single slide
            (make_elements(("Title", 1), ("Body", 1)), ["Title\nBody"]),
            # two slides
            (make_elements(("Slide1", 1), ("Slide2", 2)), ["Slide1", "Slide2"]),
            # multiple elements per slide across three slides
            (make_elements(("A", 1), ("B", 1), ("C", 2), ("D", 3)), ["A\nB", "C", "D"]),
        ],
    )
    @patch("redbox.loader.extraction.unstructured.partition_pptx")
    def test_slide_grouping(self, mock_partition, elements, expected_pages):
        mock_partition.return_value = elements
        result = SERVICE._extract_pptx(DUMMY_BYTES)
        assert result == expected_pages

    @patch("redbox.loader.extraction.unstructured.partition_pptx", return_value=[])
    def test_raises_on_no_elements(self, _mock):
        with pytest.raises(ValueError, match="no elements"):
            SERVICE._extract_pptx(DUMMY_BYTES)

    @patch("redbox.loader.extraction.unstructured.partition_pptx")
    def test_no_page_metadata_falls_back_to_single_page(self, mock_partition):
        # Elements with no page_number -> all joined as one page
        mock_partition.return_value = make_elements(("A", None), ("B", None))
        result = SERVICE._extract_pptx(DUMMY_BYTES)
        assert result == ["A\nB"]

    @patch("redbox.loader.extraction.unstructured.partition_pptx", side_effect=ImportError("missing extra"))
    def test_raises_import_error_for_missing_extra(self, _mock):
        with pytest.raises(ImportError):
            SERVICE._extract_pptx(DUMMY_BYTES)

    @patch("redbox.loader.extraction.unstructured.partition_pptx", side_effect=RuntimeError("bad file"))
    def test_propagates_partition_exceptions(self, _mock):
        with pytest.raises(RuntimeError, match="bad file"):
            SERVICE._extract_pptx(DUMMY_BYTES)


class TestExtract:
    @pytest.mark.parametrize(
        "elements, file_name, expected_pages",
        [
            # page_number used for grouping
            (make_elements(("A", 1), ("B", 1), ("C", 2)), "doc.pdf", ["A\nB", "C"]),
            # single page document
            (make_elements(("Only", 1)), "doc.pdf", ["Only"]),
            # no page metadata -> single fallback page
            (make_elements(("X", None), ("Y", None)), "doc.txt", ["X\nY"]),
            # slide_number used when page_number absent
            (
                [
                    make_element("S1", page_number=None, slide_number=1),
                    make_element("S2", page_number=None, slide_number=2),
                ],
                "deck.pptx",
                ["S1", "S2"],
            ),
        ],
    )
    @patch("redbox.loader.extraction.unstructured.partition")
    def test_page_grouping(self, mock_partition, elements, file_name, expected_pages):
        mock_partition.return_value = elements
        result = SERVICE._extract(DUMMY_BYTES, file_name)
        assert result == expected_pages

    @patch("redbox.loader.extraction.unstructured.partition", return_value=[])
    def test_raises_on_no_elements(self, _mock):
        with pytest.raises(ValueError, match="no elements"):
            SERVICE._extract(DUMMY_BYTES, "empty.pdf")

    @patch("redbox.loader.extraction.unstructured.partition")
    def test_seeks_to_zero_before_partition(self, mock_partition):
        mock_partition.return_value = make_elements(("text", 1))
        buf = BytesIO(b"data")
        buf.read()
        SERVICE._extract(buf, "file.pdf")
        mock_partition.assert_called_once()
        assert mock_partition.call_args.kwargs["file"].read() == b"data"

    @patch("redbox.loader.extraction.unstructured.partition")
    def test_page_number_takes_precedence_over_slide_number(self, mock_partition):
        # element has both; page_number should win (it's checked first in _extract)
        el = make_element("content", page_number=2, slide_number=99)
        mock_partition.return_value = [make_element("first", page_number=1), el]
        result = SERVICE._extract(DUMMY_BYTES, "mixed.pptx")
        assert len(result) == 2  # grouped by page_number (1 and 2), not slide_number
