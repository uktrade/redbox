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
        "elements",
        [
            make_elements(("Hello", 1), ("World", 1)),
            make_elements(("Page1", 1), ("Page2", 2)),
            make_elements(("A", 1), ("B", 1), ("C", 2), ("D", 3)),
            make_elements(("X", None), ("Y", None)),
        ],
    )
    @patch("redbox.loader.extraction.unstructured.partition_docx")
    def test_returns_elements(self, mock_partition, elements):
        mock_partition.return_value = elements

        result = SERVICE._extract_docx(DUMMY_BYTES)

        assert result is elements

    @patch("redbox.loader.extraction.unstructured.partition_docx", return_value=[])
    def test_raises_on_no_elements(self, _mock):
        with pytest.raises(ValueError, match="no elements"):
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

    @patch("redbox.loader.extraction.unstructured.partition_docx")
    def test_preserves_element_order(self, mock_partition):
        elements = make_elements(
            ("A1", 1),
            ("A2", 1),
            ("A3", 1),
        )

        mock_partition.return_value = elements

        result = SERVICE._extract_docx(DUMMY_BYTES)

        assert result == elements


class TestExtractPptx:
    @pytest.mark.parametrize(
        "elements",
        [
            make_elements(("Title", 1), ("Body", 1)),
            make_elements(("Slide1", 1), ("Slide2", 2)),
            make_elements(("A", 1), ("B", 1), ("C", 2), ("D", 3)),
        ],
    )
    @patch("redbox.loader.extraction.unstructured.partition_pptx")
    def test_returns_elements(self, mock_partition, elements):
        mock_partition.return_value = elements

        result = SERVICE._extract_pptx(DUMMY_BYTES)

        assert result is elements

    @patch("redbox.loader.extraction.unstructured.partition_pptx", return_value=[])
    def test_raises_on_no_elements(self, _mock):
        with pytest.raises(ValueError, match="no elements"):
            SERVICE._extract_pptx(DUMMY_BYTES)

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
        "elements, file_name",
        [
            (make_elements(("A", 1), ("B", 1), ("C", 2)), "doc.pdf"),
            (make_elements(("Only", 1)), "doc.pdf"),
            (make_elements(("X", None), ("Y", None)), "doc.txt"),
            (
                [
                    make_element("S1", page_number=None, slide_number=1),
                    make_element("S2", page_number=None, slide_number=2),
                ],
                "deck.pptx",
            ),
        ],
    )
    @patch("redbox.loader.extraction.unstructured.partition")
    def test_returns_elements(self, mock_partition, elements, file_name):
        mock_partition.return_value = elements

        result = SERVICE._extract(DUMMY_BYTES, file_name)

        assert result is elements

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
    def test_returns_same_elements_from_partition(self, mock_partition):
        elements = make_elements(("A", 1), ("B", 2))
        mock_partition.return_value = elements

        result = SERVICE._extract(DUMMY_BYTES, "file.pdf")

        assert result is elements
