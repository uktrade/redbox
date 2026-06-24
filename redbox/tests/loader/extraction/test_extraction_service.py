import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock, ANY, call

from redbox.loader.extraction.service import DocumentExtractionService


def make_element(text: str, page_number: int | None = None, slide_number: int | None = None) -> MagicMock:
    el = MagicMock()
    el.__str__ = lambda self: text
    el.metadata.page_number = page_number
    el.metadata.slide_number = slide_number
    return el


def make_elements(*specs: tuple) -> list:
    """specs: (text, page_number) tuples."""
    return [make_element(text, page) for text, page in specs]


BUCKET = "test-bucket"
PAGES = ["page one", "page two"]
TABULAR = [{"text": "col1,col2", "metadata": {}}]


def make_service() -> DocumentExtractionService:
    with (
        patch("redbox.loader.extraction.service.boto3.client"),
        patch("redbox.loader.extraction.service.TextractService"),
        patch("redbox.loader.extraction.service.UnstructuredService"),
    ):
        return DocumentExtractionService(bucket=BUCKET)


def make_s3_body(content: bytes = b"fake") -> MagicMock:
    body = MagicMock()
    body.read.return_value = content
    return body


def patch_s3(svc: DocumentExtractionService, content: bytes = b"fake"):
    svc.s3.get_object = MagicMock(return_value={"Body": make_s3_body(content)})


class TestExtractInit:
    @patch("redbox.loader.extraction.service.boto3.client")
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

    @patch("redbox.loader.extraction.service.boto3.client")
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
    @patch("redbox.loader.extraction.service.fitz.open")
    def test_returns_non_empty_pages(self, mock_fitz):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "  page text  "
        mock_fitz.return_value.__iter__ = MagicMock(return_value=iter([mock_page, mock_page]))
        result = make_service()._extract_pdf_text_direct(BytesIO(b"pdf"))
        assert result == ["page text", "page text"]

    @patch("redbox.loader.extraction.service.fitz.open")
    def test_skips_blank_pages(self, mock_fitz):
        pages = [MagicMock(), MagicMock(), MagicMock()]
        pages[0].get_text.return_value = "real text"
        pages[1].get_text.return_value = "   "  # blank
        pages[2].get_text.return_value = "more text"
        mock_fitz.return_value.__iter__ = MagicMock(return_value=iter(pages))
        result = make_service()._extract_pdf_text_direct(BytesIO(b"pdf"))
        assert result == ["real text", "more text"]

    @patch("redbox.loader.extraction.service.fitz.open")
    def test_raises_when_all_pages_blank(self, mock_fitz):
        page = MagicMock()
        page.get_text.return_value = "   "
        mock_fitz.return_value.__iter__ = MagicMock(return_value=iter([page]))
        with pytest.raises(ValueError, match="no extractable text"):
            make_service()._extract_pdf_text_direct(BytesIO(b"pdf"))

    @patch("redbox.loader.extraction.service.fitz.open")
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
    @patch("redbox.loader.extraction.service.load_tabular_file", return_value=TABULAR)
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
        "auto, fast, textract, direct, expected, raises",
        [
            # auto success
            ([make_element(text="auto-page")], None, None, None, [make_element(text="auto-page")], False),
            # fast success
            (
                RuntimeError("auto"),
                [make_element(text="fast-page")],
                None,
                None,
                [make_element(text="fast-page")],
                False,
            ),
            # textract success
            (RuntimeError("auto"), RuntimeError("fast"), ["textract-page"], None, ["textract-page"], False),
            # direct fallback success
            (
                RuntimeError("auto"),
                RuntimeError("fast"),
                RuntimeError("textract"),
                ["direct-page"],
                ["direct-page"],
                False,
            ),
            # direct fails -> FINAL raise
            (RuntimeError("auto"), RuntimeError("fast"), RuntimeError("textract"), RuntimeError("direct"), None, True),
        ],
    )
    def test_pdf_matrix(self, auto, fast, textract, direct, expected, raises):

        svc = make_service()
        patch_s3(svc)

        # unstructured mock behaviour (auto + fast)
        def unstructured_mock(file_bytes, key, strategy):
            if strategy == "auto":
                if isinstance(auto, Exception):
                    raise auto
                return auto
            if strategy == "fast":
                if isinstance(fast, Exception):
                    raise fast
                return fast

        svc.unstructured._extract.side_effect = unstructured_mock

        # textract
        if isinstance(textract, Exception):
            svc.textract.document_analysis.side_effect = textract
        else:
            svc.textract.document_analysis.return_value = textract

        # direct fallback
        direct_patch = patch.object(
            svc,
            "_extract_pdf_text_direct",
            side_effect=(direct if isinstance(direct, Exception) else None),
            return_value=None if isinstance(direct, Exception) else direct,
        )

        with direct_patch as mock_direct:
            if raises:
                with pytest.raises(RuntimeError, match="All extraction strategies"):
                    svc.extract("file.pdf")
                return

            result = svc.extract("file.pdf")

            assert [str(r) for r in result] == [str(e) for e in expected]

            if direct == "direct-page":
                mock_direct.assert_called_once()


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
    @pytest.mark.parametrize(
        "s3_error, body_bytes, expected_exception",
        [
            # happy path - S3 returns bytes
            (None, b"pdf-bytes", None),
            # S3 failure propagates
            (RuntimeError("S3 unavailable"), None, RuntimeError),
        ],
    )
    def test_s3_fetch_matrix(self, s3_error, body_bytes, expected_exception):
        svc = make_service()

        if s3_error:
            svc.s3.get_object = MagicMock(side_effect=s3_error)
        else:
            body = MagicMock()
            body.read.return_value = body_bytes
            svc.s3.get_object = MagicMock(return_value={"Body": body})

        if expected_exception:
            with pytest.raises(RuntimeError, match="S3 unavailable"):
                svc.extract("any/file.pdf")
            return

        # we just verify extraction runs without S3 error
        svc.unstructured._extract.return_value = ["ok"]

        result = svc.extract("any/file.pdf")
        assert result == ["ok"]

        svc.s3.get_object.assert_called_once_with(
            Bucket=BUCKET,
            Key="any/file.pdf",
        )
