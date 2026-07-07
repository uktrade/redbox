import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock, ANY, call
import sys
import types

from redbox_app.redbox_core.enums import IngestExtractionStrategy
from redbox.loader.extraction.textract import TextractTimeout

_orig_cache = sys.modules.get("django.core.cache")

fake_cache_module = types.ModuleType("django.core.cache")
fake_cache_module.cache = MagicMock()
fake_cache_module.caches = MagicMock()
fake_cache_module.InvalidCacheBackendError = type(
    "InvalidCacheBackendError",
    (Exception,),
    {},
)

sys.modules["django.core.cache"] = fake_cache_module
try:
    from redbox.loader.extraction.service import (
        DocumentExtractionService,
        STRATEGIES,
        INGEST_LOCK_TIMEOUT_SECONDS,
        LARGE_PDF_BYTES_THRESHOLD,
        IngestionAlreadyInProgress,
    )
finally:
    if _orig_cache is None:
        sys.modules.pop("django.core.cache", None)
    else:
        sys.modules["django.core.cache"] = _orig_cache


@pytest.fixture(autouse=True)
def mock_cache():
    with patch("redbox.loader.extraction.service.cache") as cache:
        cache.add.return_value = True
        cache.delete.return_value = None
        yield cache


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


def patch_s3(
    svc: DocumentExtractionService,
    content: bytes = b"fake",
    file_size: int | None = None,
):
    svc.s3.get_object = MagicMock(return_value={"Body": make_s3_body(content)})
    svc.s3.head_object = MagicMock(return_value={"ContentLength": file_size if file_size is not None else len(content)})


class TestExtractInit:
    @patch("redbox.loader.extraction.service.boto3.client")
    def test_init_default_parameters(self, mock_boto_client: MagicMock):
        extractor = DocumentExtractionService(bucket="test-bucket")

        assert extractor.bucket == "test-bucket"
        assert extractor.region == "eu-west-2"

        assert extractor.textract.bucket == extractor.bucket
        assert extractor.textract.region == extractor.region

        assert mock_boto_client.call_count == 3
        mock_boto_client.assert_has_calls(
            [
                call("s3", region_name="eu-west-2"),
                call("textract", region_name="eu-west-2", config=ANY),
                call("s3", region_name="eu-west-2"),
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

        assert mock_boto_client.call_count == 3
        mock_boto_client.assert_has_calls(
            [
                call("s3", region_name=region),
                call("textract", region_name=region, config=ANY),
                call("s3", region_name=region),
            ]
        )


class TestExtractPdfTextDirect:
    @patch("redbox.loader.extraction.service.fitz.open")
    def test_returns_non_empty_pages(self, mock_fitz):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "  page text  "
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([mock_page, mock_page])
        mock_fitz.return_value.__enter__.return_value = mock_doc

        result = make_service()._extract_pdf_text_direct(BytesIO(b"pdf"), "[log]")
        assert result == ["page text", "page text"]

    @patch("redbox.loader.extraction.service.fitz.open")
    def test_skips_blank_pages(self, mock_fitz):
        pages = [MagicMock(), MagicMock(), MagicMock()]
        pages[0].get_text.return_value = "real text"
        pages[1].get_text.return_value = "   "  # blank
        pages[2].get_text.return_value = "more text"
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter(pages)
        mock_fitz.return_value.__enter__.return_value = mock_doc

        result = make_service()._extract_pdf_text_direct(BytesIO(b"pdf"), "[log]")
        assert result == ["real text", "more text"]

    @patch("redbox.loader.extraction.service.fitz.open")
    def test_raises_when_all_pages_blank(self, mock_fitz):
        page = MagicMock()
        page.get_text.return_value = "   "
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([page])
        mock_fitz.return_value.__enter__.return_value = mock_doc

        with pytest.raises(ValueError, match="no extractable text"):
            make_service()._extract_pdf_text_direct(BytesIO(b"pdf"), "[log]")

    @patch("redbox.loader.extraction.service.fitz.open")
    def test_raises_when_no_pages(self, mock_fitz):
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = iter([])
        mock_fitz.return_value.__enter__.return_value = mock_doc

        with pytest.raises(ValueError, match="no extractable text"):
            make_service()._extract_pdf_text_direct(BytesIO(b"pdf"), "[log]")


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
        strategy, result = svc.extract(file_name)
        assert strategy == IngestExtractionStrategy.tabular
        assert result == TABULAR
        mock_load.assert_called_once()


class TestOfficeToPdfRouting:
    @pytest.mark.parametrize(
        "file_name, converter",
        [
            ("document.docx", "docx_to_pdf"),
            ("document.doc", "docx_to_pdf"),
            ("deck.pptx", "pptx_to_pdf"),
            ("deck.ppt", "pptx_to_pdf"),
        ],
    )
    def test_office_documents_are_converted_then_extracted(self, file_name, converter):
        svc = make_service()
        patch_s3(svc)

        extracted_pdf = MagicMock()

        with (
            patch.object(svc, converter, return_value=extracted_pdf) as mock_converter,
            patch.object(
                svc,
                "_extract_pdf",
                return_value=(IngestExtractionStrategy.pymupdf, ["page"]),
            ) as mock_extract,
        ):
            strategy, result = svc.extract(file_name)

        assert strategy == IngestExtractionStrategy.pymupdf
        assert result == ["page"]

        mock_converter.assert_called_once()

        mock_extract.assert_called_once_with(
            s3_key=file_name,
            pdf=extracted_pdf,
            log_stub=ANY,
            use_s3_textract=False,
        )


class TestOfficeConversion:
    def test_docx_to_pdf_delegates(self):
        svc = make_service()

        expected = MagicMock()

        with patch.object(
            svc,
            "_convert_office_to_pdf",
            return_value=expected,
        ) as convert:
            result = svc.docx_to_pdf(BytesIO(b"doc"), "[log]")

        assert result is expected

        convert.assert_called_once_with(
            file_bytes=ANY,
            suffix=".docx",
            log_stub="[log]",
        )

    def test_pptx_to_pdf_delegates(self):
        svc = make_service()

        expected = MagicMock()

        with patch.object(
            svc,
            "_convert_office_to_pdf",
            return_value=expected,
        ) as convert:
            result = svc.pptx_to_pdf(BytesIO(b"ppt"), "[log]")

        assert result is expected

        convert.assert_called_once_with(
            file_bytes=ANY,
            suffix=".pptx",
            log_stub="[log]",
        )


class TestPdfRouting:
    @pytest.mark.parametrize(
        "page_count,file_size,expected_large",
        [
            (1, LARGE_PDF_BYTES_THRESHOLD, False),  # normal PDF
            (201, 1024, True),  # large by page count
            (10, LARGE_PDF_BYTES_THRESHOLD + 1, True),  # large by file size
        ],
    )
    def test_routes_based_on_pdf_size_or_page_count(
        self,
        page_count,
        file_size,
        expected_large,
    ):
        svc = make_service()

        patch_s3(svc, content=b"x" * file_size, file_size=file_size)

        mock_doc = MagicMock()
        mock_doc.page_count = page_count

        with (
            patch("redbox.loader.extraction.service.fitz.open") as mock_open,
            patch.object(
                svc,
                "_run_with_fallbacks",
                return_value=(
                    IngestExtractionStrategy.unstructured_auto,
                    ["page"],
                ),
            ) as mock_fallback,
            patch.object(
                svc.textract,
                "document_analysis_large",
                return_value=["page"],
            ) as mock_large,
        ):
            mock_open.return_value.__enter__.return_value = mock_doc

            strategy, _ = svc.extract("file.pdf")

        if expected_large:
            assert strategy == IngestExtractionStrategy.textract_document_analysis_large
            mock_large.assert_called_once()
            mock_fallback.assert_not_called()
        else:
            assert strategy == IngestExtractionStrategy.unstructured_auto
            mock_fallback.assert_called_once()
            mock_large.assert_not_called()


class TestExtractPdf:
    @pytest.mark.parametrize(
        "results, raises",
        [
            # First configured strategy succeeds
            (
                {
                    "unstructured_auto": [make_element(text="auto-page")],
                    "unstructured_fast": RuntimeError("fast"),
                    "textract_document_analysis": RuntimeError("textract"),
                    "pymupdf": ["direct-page"],
                },
                False,
            ),
            # Second configured strategy succeeds
            (
                {
                    "unstructured_auto": RuntimeError("auto"),
                    "unstructured_fast": [make_element(text="fast-page")],
                    "textract_document_analysis": RuntimeError("textract"),
                    "pymupdf": ["direct-page"],
                },
                False,
            ),
            # Third configured strategy succeeds
            (
                {
                    "unstructured_auto": RuntimeError("auto"),
                    "unstructured_fast": RuntimeError("fast"),
                    "textract_document_analysis": ["textract-page"],
                    "pymupdf": ["direct-page"],
                },
                False,
            ),
            # All configured strategies fail, direct fallback succeeds
            (
                {
                    "unstructured_auto": RuntimeError("auto"),
                    "unstructured_fast": RuntimeError("fast"),
                    "textract_document_analysis": RuntimeError("textract"),
                    "pymupdf": ["direct-page"],
                },
                False,
            ),
            # Everything fails
            (
                {
                    "unstructured_auto": RuntimeError("auto"),
                    "unstructured_fast": RuntimeError("fast"),
                    "textract_document_analysis": RuntimeError("textract"),
                    "pymupdf": RuntimeError("direct"),
                },
                True,
            ),
        ],
    )
    def test_pdf_fallbacks(self, results, raises):
        svc = make_service()

        def get_result(name):
            return results.get(name, RuntimeError(name))

        def unstructured_mock(file_bytes, key, strategy, **kwargs):
            result = get_result(f"unstructured_{strategy}")
            if isinstance(result, Exception):
                raise result
            return result

        svc.unstructured._extract.side_effect = unstructured_mock

        textract = get_result("textract_document_analysis")
        if isinstance(textract, Exception):
            svc.textract.document_analysis.side_effect = textract
        else:
            svc.textract.document_analysis.return_value = textract

        direct = get_result("pymupdf")

        with patch.object(
            svc,
            "_extract_pdf_text_direct",
            side_effect=direct if isinstance(direct, Exception) else None,
            return_value=None if isinstance(direct, Exception) else direct,
        ) as mock_direct:
            if raises:
                with pytest.raises(RuntimeError, match="All extraction strategies"):
                    svc._run_with_fallbacks(
                        file_bytes=BytesIO(b"pdf"),
                        s3_key="file.pdf",
                        log_stub="[log]",
                    )
                return

            for strategy in STRATEGIES:
                result = get_result(strategy)
                if not isinstance(result, Exception):
                    expected_strategy = IngestExtractionStrategy(strategy)
                    expected = result
                    expected_direct = False
                    break
            else:
                expected_strategy = IngestExtractionStrategy.pymupdf
                expected = direct
                expected_direct = True

            strategy, result = svc._run_with_fallbacks(
                file_bytes=BytesIO(b"pdf"),
                s3_key="file.pdf",
                log_stub="[log]",
            )

            assert strategy == expected_strategy
            assert [str(r) for r in result] == [str(r) for r in expected]

            if expected_direct:
                mock_direct.assert_called_once_with(ANY, "[log]")
            else:
                mock_direct.assert_not_called()


class TestExtractGeneric:
    @pytest.mark.parametrize(
        "file_name",
        [
            "image.png",
            "archive/data.json",
        ],
    )
    def test_unknown_types_route_to_unstructured_extract(self, file_name):
        svc = make_service()
        patch_s3(svc)
        svc.unstructured._extract.return_value = PAGES
        strategy, result = svc.extract(file_name)
        assert strategy == IngestExtractionStrategy.unstructured_auto
        assert result == PAGES
        svc.unstructured._extract.assert_called_once()


class TestExtractMarkupTypes:
    @pytest.mark.parametrize(
        "file_name",
        [
            "notes.md",
            "notes.markdown",
        ],
    )
    def test_markdown_routes_to_extract_markdown(self, file_name):
        svc = make_service()
        patch_s3(svc)
        svc.unstructured._extract_markdown.return_value = PAGES

        strategy, result = svc.extract(file_name)

        assert strategy == IngestExtractionStrategy.unstructured_auto
        assert result == PAGES
        svc.unstructured._extract_markdown.assert_called_once()
        assert svc.unstructured._extract_markdown.call_args.args[1] == 300
        svc.unstructured._extract.assert_not_called()

    def test_markdown_extraction_failure_propagates(self):
        svc = make_service()
        patch_s3(svc)
        svc.unstructured._extract_markdown.side_effect = RuntimeError("bad markdown")

        with pytest.raises(RuntimeError, match="bad markdown"):
            svc.extract("notes.md")

    @pytest.mark.parametrize(
        "file_name",
        [
            "page.html",
            "page.htm",
        ],
    )
    def test_html_routes_to_extract_html(self, file_name):
        svc = make_service()
        patch_s3(svc)
        svc.unstructured._extract_html.return_value = PAGES

        strategy, result = svc.extract(file_name)

        assert strategy == IngestExtractionStrategy.unstructured_auto
        assert result == PAGES
        svc.unstructured._extract_html.assert_called_once()
        assert svc.unstructured._extract_html.call_args.args[1] == 300
        svc.unstructured._extract.assert_not_called()

    def test_html_extraction_failure_propagates(self):
        svc = make_service()
        patch_s3(svc)
        svc.unstructured._extract_html.side_effect = RuntimeError("malformed markup")

        with pytest.raises(RuntimeError, match="malformed markup"):
            svc.extract("page.html")

    def test_txt_routes_to_extract_text(self):
        svc = make_service()
        patch_s3(svc)
        svc.unstructured._extract_text.return_value = PAGES

        strategy, result = svc.extract("notes.txt")

        assert strategy == IngestExtractionStrategy.unstructured_auto
        assert result == PAGES
        svc.unstructured._extract_text.assert_called_once()
        assert svc.unstructured._extract_text.call_args.args[1] == 300
        svc.unstructured._extract.assert_not_called()

    def test_txt_extraction_failure_propagates(self):
        svc = make_service()
        patch_s3(svc)
        svc.unstructured._extract_text.side_effect = RuntimeError("bad encoding")

        with pytest.raises(RuntimeError, match="bad encoding"):
            svc.extract("notes.txt")


class TestExtractS3:
    def test_fetches_pdf_from_s3(self):
        svc = make_service()
        patch_s3(svc, content=b"pdf-bytes", file_size=1024)

        with (
            patch("redbox.loader.extraction.service.fitz.open") as mock_open,
            patch.object(
                svc,
                "_extract_pdf",
                return_value=(IngestExtractionStrategy.unstructured_auto, ["ok"]),
            ) as mock_extract,
        ):
            mock_doc = MagicMock()
            mock_doc.page_count = 1
            mock_open.return_value.__enter__.return_value = mock_doc

            strategy, result = svc.extract(
                "any/file.pdf",
            )

        assert strategy == IngestExtractionStrategy.unstructured_auto
        assert result == ["ok"]

        svc.s3.get_object.assert_called_once_with(
            Bucket=BUCKET,
            Key="any/file.pdf",
        )

        mock_extract.assert_called_once()

    def test_s3_get_object_failure_propagates(self):
        svc = make_service()

        patch_s3(svc, file_size=1024)
        svc.s3.get_object.side_effect = RuntimeError("S3 unavailable")

        with pytest.raises(RuntimeError, match="S3 unavailable"):
            svc.extract("any/file.pdf")

        svc.s3.get_object.assert_called_once_with(
            Bucket=BUCKET,
            Key="any/file.pdf",
        )


class TestLargePdfExtraction:
    def make_pdf(self):
        pdf = MagicMock()
        pdf.bytes = BytesIO(b"pdf")
        pdf.page_count = 300
        pdf.file_size = 1024
        return pdf

    def test_large_pdf_timeout_falls_back_to_pymupdf(self):
        svc = make_service()

        pdf = self.make_pdf()

        svc.textract.document_analysis_large.side_effect = TextractTimeout("timeout")

        with patch.object(
            svc,
            "_extract_pdf_text_direct",
            return_value=["page"],
        ) as mock_direct:
            strategy, result = svc._extract_pdf(
                s3_key="file.pdf",
                pdf=pdf,
                log_stub="[log]",
            )

        assert strategy == IngestExtractionStrategy.pymupdf
        assert result == ["page"]

        svc.textract.document_analysis_large.assert_called_once()
        mock_direct.assert_called_once_with(pdf.bytes, "[log]")

    def test_large_pdf_uses_pdf_bytes_when_not_using_s3_textract(self):
        svc = make_service()

        pdf = self.make_pdf()

        svc.textract.document_analysis_large.return_value = ["page"]

        strategy, result = svc._extract_pdf(
            s3_key="document.docx",
            pdf=pdf,
            log_stub="[log]",
            use_s3_textract=False,
        )

        assert strategy == IngestExtractionStrategy.textract_document_analysis_large
        assert result == ["page"]

        svc.textract.document_analysis_large.assert_called_once_with(
            key="document.docx",
            file_bytes=pdf.bytes,
            timeout=ANY,
        )

    def test_large_pdf_uses_s3_reference_when_enabled(self):
        svc = make_service()

        pdf = self.make_pdf()

        svc.textract.document_analysis_large.return_value = ["page"]

        strategy, result = svc._extract_pdf(
            s3_key="file.pdf",
            pdf=pdf,
            log_stub="[log]",
            use_s3_textract=True,
        )

        assert strategy == IngestExtractionStrategy.textract_document_analysis_large
        assert result == ["page"]

        svc.textract.document_analysis_large.assert_called_once_with(
            key="file.pdf",
            file_bytes=None,
            timeout=ANY,
        )


class TestExtractLocking:
    def test_acquires_and_releases_lock(self, mock_cache):
        svc = make_service()
        patch_s3(svc)

        svc.unstructured._extract.return_value = PAGES

        strategy, result = svc.extract("notes.json")

        assert strategy == IngestExtractionStrategy.unstructured_auto
        assert result == PAGES

        mock_cache.add.assert_called_once_with(
            "ingest-lock:notes.json",
            "1",
            timeout=INGEST_LOCK_TIMEOUT_SECONDS,
        )
        mock_cache.delete.assert_called_once_with(
            "ingest-lock:notes.json",
        )

    def test_releases_lock_when_extraction_fails(self, mock_cache):
        svc = make_service()
        patch_s3(svc)

        svc.unstructured._extract.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            svc.extract("notes.json")

        mock_cache.add.assert_called_once_with(
            "ingest-lock:notes.json",
            "1",
            timeout=INGEST_LOCK_TIMEOUT_SECONDS,
        )
        mock_cache.delete.assert_called_once_with(
            "ingest-lock:notes.json",
        )

    def test_raises_when_lock_already_exists(self, mock_cache):
        mock_cache.add.return_value = False

        svc = make_service()

        with pytest.raises(IngestionAlreadyInProgress):
            svc.extract("notes.json")

        mock_cache.add.assert_called_once_with(
            "ingest-lock:notes.json",
            "1",
            timeout=INGEST_LOCK_TIMEOUT_SECONDS,
        )
        mock_cache.delete.assert_not_called()
