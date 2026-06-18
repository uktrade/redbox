import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock, ANY, call

from redbox.loader.extraction.base import DocumentExtractionService


BUCKET = "test-bucket"
PAGES = ["page one", "page two"]
TABULAR = [{"text": "col1,col2", "metadata": {}}]


def make_service() -> DocumentExtractionService:
    with (
        patch("redbox.loader.extraction.base.boto3.client"),
        patch("redbox.loader.extraction.base.TextractService"),
        patch("redbox.loader.extraction.base.UnstructuredService"),
    ):
        return DocumentExtractionService(bucket=BUCKET)


def make_s3_body(content: bytes = b"fake") -> MagicMock:
    body = MagicMock()
    body.read.return_value = content
    return body


def patch_s3(svc: DocumentExtractionService, content: bytes = b"fake"):
    svc.s3.get_object = MagicMock(return_value={"Body": make_s3_body(content)})


class TestExtractInit:
    @patch("redbox.loader.extraction.base.boto3.client")
    def test_init_default_parameters(self, mock_boto_client: MagicMock):
        extractor = DocumentExtractionService(bucket="test-bucket")

        assert extractor.bucket == "test-bucket"
        assert extractor.region == "eu-west-2"

        assert extractor.textract.bucket == extractor.bucket
        assert extractor.textract.region == extractor.region

        assert mock_boto_client.call_count == 2
        mock_boto_client.assert_has_calls(
            [
                call("s3", region_name="eu-west-2"),
                call("textract", region_name="eu-west-2", config=ANY),
            ]
        )

    @patch("redbox.loader.extraction.base.boto3.client")
    def test_init_custom_parameters(self, mock_boto_client: MagicMock):
        bucket, region = "test-bucket-2", "eu-west-1"
        extractor = DocumentExtractionService(
            bucket=bucket,
            region=region,
        )

        assert extractor.bucket == bucket
        assert extractor.region == region

        assert extractor.textract.bucket == extractor.bucket
        assert extractor.textract.region == extractor.region

        assert mock_boto_client.call_count == 2
        mock_boto_client.assert_has_calls(
            [
                call("s3", region_name=region),
                call("textract", region_name=region, config=ANY),
            ]
        )


class TestExtractPdfTextDirect:
    @patch("redbox.loader.extraction.base.fitz.open")
    def test_returns_non_empty_pages(self, mock_fitz):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "  page text  "
        mock_fitz.return_value.__iter__ = MagicMock(return_value=iter([mock_page, mock_page]))
        result = make_service()._extract_pdf_text_direct(BytesIO(b"pdf"))
        assert result == ["page text", "page text"]

    @patch("redbox.loader.extraction.base.fitz.open")
    def test_skips_blank_pages(self, mock_fitz):
        pages = [MagicMock(), MagicMock(), MagicMock()]
        pages[0].get_text.return_value = "real text"
        pages[1].get_text.return_value = "   "  # blank
        pages[2].get_text.return_value = "more text"
        mock_fitz.return_value.__iter__ = MagicMock(return_value=iter(pages))
        result = make_service()._extract_pdf_text_direct(BytesIO(b"pdf"))
        assert result == ["real text", "more text"]

    @patch("redbox.loader.extraction.base.fitz.open")
    def test_raises_when_all_pages_blank(self, mock_fitz):
        page = MagicMock()
        page.get_text.return_value = "   "
        mock_fitz.return_value.__iter__ = MagicMock(return_value=iter([page]))
        with pytest.raises(ValueError, match="no extractable text"):
            make_service()._extract_pdf_text_direct(BytesIO(b"pdf"))

    @patch("redbox.loader.extraction.base.fitz.open")
    def test_raises_when_no_pages(self, mock_fitz):
        mock_fitz.return_value.__iter__ = MagicMock(return_value=iter([]))
        with pytest.raises(ValueError, match="no extractable text"):
            make_service()._extract_pdf_text_direct(BytesIO(b"pdf"))


class TestExtractTabular:
    @pytest.mark.parametrize(
        "file_name",
        [
            "data/report.csv",
            "data/report.tsv",
            "data/report.xls",
            "data/report.xlsx",
        ],
    )
    @patch("redbox.loader.extraction.base.load_tabular_file", return_value=TABULAR)
    def test_routes_to_load_tabular(self, mock_load, file_name):
        svc = make_service()
        patch_s3(svc)
        result = svc.extract(file_name)
        assert result == TABULAR
        mock_load.assert_called_once()


class TestExtractOffice:
    @pytest.mark.parametrize(
        "file_name, method",
        [
            ("deck.pptx", "_extract_pptx"),
            ("deck.ppt", "_extract_pptx"),
            ("document.docx", "_extract_docx"),
        ],
    )
    def test_routes_to_unstructured(self, file_name, method):
        svc = make_service()
        patch_s3(svc)
        getattr(svc.unstructured, method).return_value = PAGES
        result = svc.extract(file_name)
        assert result == PAGES
        getattr(svc.unstructured, method).assert_called_once()


class TestExtractPdf:
    @pytest.mark.parametrize(
        "is_image_heavy, expected_method",
        [
            (True, "document_analysis"),  # large + image-heavy -> Textract analysis
            (False, None),  # large + text -> direct PyMuPDF
        ],
    )
    @patch("redbox.loader.extraction.base._pdf_is_image_heavy")
    @patch("redbox.loader.extraction.base.is_large_pdf", return_value=(True, 200))
    @patch("redbox.loader.extraction.base.fitz.open")
    def test_large_pdf_routing(self, mock_fitz, _mock_large, mock_image_heavy, is_image_heavy, expected_method):
        mock_image_heavy.return_value = is_image_heavy
        page = MagicMock()
        page.get_text.return_value = "text"
        mock_fitz.return_value.__iter__ = MagicMock(return_value=iter([page]))

        svc = make_service()
        patch_s3(svc)
        svc.textract.document_analysis.return_value = PAGES

        result = svc.extract("report.pdf")

        if expected_method:
            svc.textract.document_analysis.assert_called_once_with(key="report.pdf")
            assert result == PAGES
        else:
            svc.textract.document_analysis.assert_not_called()
            assert result == ["text"]

    @patch("redbox.loader.extraction.base.is_large_pdf", return_value=(False, 5))
    def test_small_pdf_tries_textract_first(self, _mock_large):
        svc = make_service()
        patch_s3(svc)
        svc.textract.document_analysis.return_value = PAGES
        result = svc.extract("small.pdf")
        svc.textract.document_analysis.assert_called_once_with(key="small.pdf")
        assert result == PAGES

    @patch("redbox.loader.extraction.base.is_large_pdf", return_value=(False, 5))
    @patch("redbox.loader.extraction.base.fitz.open")
    def test_small_pdf_falls_back_to_direct_on_textract_failure(self, mock_fitz, _mock_large):
        page = MagicMock()
        page.get_text.return_value = "fallback text"
        mock_fitz.return_value.__iter__ = MagicMock(return_value=iter([page]))

        svc = make_service()
        patch_s3(svc)
        svc.textract.document_analysis.side_effect = RuntimeError("textract down")

        result = svc.extract("small.pdf")
        assert result == ["fallback text"]

    @patch("redbox.loader.extraction.base.is_large_pdf", return_value=(False, 5))
    @patch("redbox.loader.extraction.base.fitz.open")
    def test_fallback_raises_if_direct_also_fails(self, mock_fitz, _mock_large):
        mock_fitz.side_effect = RuntimeError("fitz broken")
        svc = make_service()
        patch_s3(svc)
        svc.textract.document_analysis.side_effect = RuntimeError("textract down")
        with pytest.raises(RuntimeError, match="fitz broken"):
            svc.extract("small.pdf")


class TestExtractGeneric:
    @pytest.mark.parametrize(
        "file_name",
        [
            "notes.txt",
            "archive/report.html",
            "image.png",
        ],
    )
    def test_unknown_types_route_to_unstructured_extract(self, file_name):
        svc = make_service()
        patch_s3(svc)
        svc.unstructured._extract.return_value = PAGES
        result = svc.extract(file_name)
        assert result == PAGES
        svc.unstructured._extract.assert_called_once()


class TestExtractS3:
    @patch("redbox.loader.extraction.base.is_large_pdf", return_value=(False, 5))
    def test_fetches_from_correct_bucket_and_key(self, _mock_large):
        svc = make_service()
        patch_s3(svc)
        svc.textract.document_analysis.return_value = PAGES
        svc.extract("path/to/report.pdf")
        svc.s3.get_object.assert_called_once_with(Bucket=BUCKET, Key="path/to/report.pdf")

    def test_propagates_s3_exception(self):
        svc = make_service()
        svc.s3.get_object = MagicMock(side_effect=RuntimeError("S3 unavailable"))
        with pytest.raises(RuntimeError, match="S3 unavailable"):
            svc.extract("any/file.pdf")
