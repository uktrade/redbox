import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock, ANY, call
import types
import sys

from redbox.models.file import ChunkResolution
from redbox_app.redbox_core.enums import IngestExtractionStrategy

fake_cache_module = types.ModuleType("django.core.cache")
fake_cache_module.cache = MagicMock()
sys.modules["django.core.cache"] = fake_cache_module

from redbox.loader.extraction.service import (  # noqa: E402
    DocumentExtractionService,
    STRATEGIES,
    INGEST_LOCK_TIMEOUT_SECONDS,
    IngestionAlreadyInProgress,
)


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
        strategy, result = svc.extract(file_name, ChunkResolution.tabular)
        assert strategy == IngestExtractionStrategy.tabular
        assert result == TABULAR
        mock_load.assert_called_once()


class TestExtractOffice:
    @pytest.mark.parametrize(
        "file_name, method, expected_strategy",
        [
            ("deck.pptx", "_extract_pptx", IngestExtractionStrategy.unstructured_pptx),
            ("deck.ppt", "_extract_pptx", IngestExtractionStrategy.unstructured_pptx),
            ("document.docx", "_extract_docx", IngestExtractionStrategy.unstructured_docx),
        ],
    )
    def test_routes_to_unstructured(self, file_name, method, expected_strategy):
        svc = make_service()
        patch_s3(svc)
        getattr(svc.unstructured, method).return_value = PAGES
        strategy, result = svc.extract(file_name, ChunkResolution.normal)
        assert strategy == expected_strategy
        assert result == PAGES
        getattr(svc.unstructured, method).assert_called_once()


class TestPdfRouting:
    @pytest.mark.parametrize(
        "file_size, expected_large",
        [
            (5 * 1024 * 1024, False),  # threshold
            (5 * 1024 * 1024 + 1, True),  # above threshold
        ],
    )
    def test_routes_based_on_pdf_size(self, file_size, expected_large):
        svc = make_service()

        patch_s3(svc, file_size=file_size)

        with (
            patch.object(
                svc,
                "_run_with_fallbacks",
                return_value=(IngestExtractionStrategy.unstructured_auto, ["page"]),
            ) as mock_fallback,
            patch.object(
                svc.textract,
                "document_analysis_large",
                return_value=["page"],
            ) as mock_large,
        ):
            strategy, _ = svc.extract("file.pdf", ChunkResolution.normal)

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
    def test_pdf_fallbacks_on_normal_resolution(self, results, raises):
        svc = make_service()
        patch_s3(svc, file_size=1024)

        def get_result(name):
            return results.get(name, RuntimeError(name))

        # unstructured
        def unstructured_mock(file_bytes, key, strategy, **kwargs):
            result = get_result(f"unstructured_{strategy}")
            if isinstance(result, Exception):
                raise result
            return result

        svc.unstructured._extract.side_effect = unstructured_mock

        # textract
        textract = get_result("textract_document_analysis")
        if isinstance(textract, Exception):
            svc.textract.document_analysis.side_effect = textract
        else:
            svc.textract.document_analysis.return_value = textract

        # direct fallback
        direct = get_result("pymupdf")
        direct_patch = patch.object(
            svc,
            "_extract_pdf_text_direct",
            side_effect=direct if isinstance(direct, Exception) else None,
            return_value=None if isinstance(direct, Exception) else direct,
        )

        with direct_patch as mock_direct:
            if raises:
                with pytest.raises(RuntimeError, match="All extraction strategies"):
                    svc.extract("file.pdf", ChunkResolution.normal)
                return

            # Determine expected winner from configured strategy order
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

            strategy, result = svc.extract("file.pdf", ChunkResolution.normal)

            assert strategy == expected_strategy
            assert [str(r) for r in result] == [str(r) for r in expected]

            if expected_direct:
                mock_direct.assert_called_once()
            else:
                mock_direct.assert_not_called()

    @pytest.mark.parametrize(
        "file_name, pages",
        [("notes.pdf", 2), ("file.PDF", 1), ("file.PDF", 0)],
    )
    def test_pdf_on_largest_resolution(self, file_name, pages):
        expected_page = "page text"

        svc = make_service()

        with patch.object(svc, "_extract_pdf_text_direct", return_value=[expected_page] * pages) as mock_direct:
            with patch.object(svc, "_run_with_fallbacks", return_value=[expected_page] * pages) as mock_fallback:
                patch_s3(svc)
                strategy, result = svc.extract(file_name, ChunkResolution.largest)

                assert strategy == IngestExtractionStrategy.pymupdf
                assert result == [expected_page] * pages
                mock_direct.assert_called_once()
                mock_fallback.assert_not_called()


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
        strategy, result = svc.extract(file_name, ChunkResolution.normal)
        assert strategy == IngestExtractionStrategy.unstructured_auto
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
            patch_s3(svc, file_size=1024)
            svc.s3.get_object.side_effect = s3_error
        else:
            patch_s3(svc, content=body_bytes, file_size=1024)

        if expected_exception:
            with pytest.raises(RuntimeError, match="S3 unavailable"):
                svc.extract("any/file.pdf", ChunkResolution.normal)
            return

        # we just verify extraction runs without S3 error
        if STRATEGIES[0].startswith("unstructured"):
            svc.unstructured._extract.return_value = ["ok"]
        else:
            svc.textract.document_analysis.return_value = ["ok"]

        strategy, result = svc.extract("any/file.pdf", ChunkResolution.normal)
        assert strategy == STRATEGIES[0]
        assert result == ["ok"]

        svc.s3.get_object.assert_called_once_with(
            Bucket=BUCKET,
            Key="any/file.pdf",
        )


class TestExtractLocking:
    def test_acquires_and_releases_lock(self, mock_cache):
        svc = make_service()
        patch_s3(svc)

        svc.unstructured._extract.return_value = PAGES

        strategy, result = svc.extract("notes.txt", ChunkResolution.normal)

        assert strategy == IngestExtractionStrategy.unstructured_auto
        assert result == PAGES

        mock_cache.add.assert_called_once_with(
            "ingest-lock:notes.txt:normal",
            "1",
            timeout=INGEST_LOCK_TIMEOUT_SECONDS,
        )
        mock_cache.delete.assert_called_once_with(
            "ingest-lock:notes.txt:normal",
        )

    def test_releases_lock_when_extraction_fails(self, mock_cache):
        svc = make_service()
        patch_s3(svc)

        svc.unstructured._extract.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            svc.extract("notes.txt", ChunkResolution.normal)

        mock_cache.add.assert_called_once_with(
            "ingest-lock:notes.txt:normal",
            "1",
            timeout=INGEST_LOCK_TIMEOUT_SECONDS,
        )
        mock_cache.delete.assert_called_once_with(
            "ingest-lock:notes.txt:normal",
        )

    def test_raises_when_lock_already_exists(self, mock_cache):
        mock_cache.add.return_value = False

        svc = make_service()

        with pytest.raises(IngestionAlreadyInProgress):
            svc.extract("notes.txt", ChunkResolution.normal)

        mock_cache.add.assert_called_once_with(
            "ingest-lock:notes.txt:normal",
            "1",
            timeout=INGEST_LOCK_TIMEOUT_SECONDS,
        )
        mock_cache.delete.assert_not_called()
