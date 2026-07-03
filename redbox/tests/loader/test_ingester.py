from unittest.mock import MagicMock, patch

import pytest

from redbox.loader.extraction.service import IngestionAlreadyInProgress
from redbox.loader.ingester import (
    FileIngestionResponse,
    _ingest_file,
    ingest_file,
)
from redbox.models.file import ChunkResolution
from redbox_app.redbox_core.enums import (
    IngestChunkingStrategy,
    IngestExtractionStrategy,
)


class TestIngestFile:
    @pytest.mark.parametrize(
        ("side_effect", "expected_result", "expected_error"),
        [
            (
                FileIngestionResponse(
                    normal_extraction_strategy=IngestExtractionStrategy.textract_document_analysis,
                    normal_chunking_strategy=IngestChunkingStrategy.unstructured_chunk_by_title,
                    largest_extraction_strategy=IngestExtractionStrategy.textract_document_analysis_large,
                    largest_chunking_strategy=IngestChunkingStrategy.unstructured_chunk_by_title,
                ),
                "response",
                None,
            ),
            (
                RuntimeError("boom"),
                None,
                "RuntimeError",
            ),
        ],
    )
    def test_returns_response_or_error(
        self,
        side_effect,
        expected_result,
        expected_error,
    ):
        if isinstance(side_effect, Exception):
            patch_kwargs = {"side_effect": side_effect}
        else:
            patch_kwargs = {"return_value": side_effect}

        with patch("redbox.loader.ingester._ingest_file", **patch_kwargs):
            result, error = ingest_file("file.csv")

        if expected_result == "response":
            assert result == side_effect
            assert error is None
        else:
            assert result is None
            assert expected_error in error

    def test_propagates_ingestion_already_in_progress(self):
        with patch(
            "redbox.loader.ingester._ingest_file",
            side_effect=IngestionAlreadyInProgress("file.pdf"),
        ):
            with pytest.raises(IngestionAlreadyInProgress):
                ingest_file("file.pdf")


class TestInternalIngestFile:
    def _mock_parallel_result(self):
        runnable = MagicMock()
        runnable.invoke.return_value = {
            "normal": {
                "strategy": IngestChunkingStrategy.unstructured_chunk_by_title,
                "documents": [],
            },
            "largest": {
                "strategy": IngestChunkingStrategy.overlapping_pages,
                "documents": [],
            },
            "tabular": {
                "strategy": IngestChunkingStrategy.tabular,
                "documents": [],
            },
            "schematised_tabular": {
                "strategy": IngestChunkingStrategy.tabular,
                "documents": [],
            },
        }
        return runnable

    def _setup(
        self,
        extraction_service_cls,
        metadata_cls,
        ingest_chunks,
        ingest_tabular_chunks,
        runnable_parallel,
        *,
        pdf: bool,
    ):
        extraction = MagicMock()

        if pdf:
            extraction.extract.side_effect = [
                (
                    IngestExtractionStrategy.textract_document_analysis,
                    ["normal"],
                ),
                (
                    IngestExtractionStrategy.pymupdf,
                    ["largest"],
                ),
            ]
        else:
            extraction.extract.return_value = (
                IngestExtractionStrategy.tabular,
                [{"text": "table"}],
            )

        extraction_service_cls.return_value = extraction

        metadata = MagicMock()
        metadata.extract.return_value = MagicMock()
        metadata_cls.return_value = metadata

        ingest_chunks.return_value = MagicMock()
        ingest_tabular_chunks.return_value = MagicMock()

        runnable = self._mock_parallel_result()
        runnable_parallel.return_value = runnable

        return extraction, metadata, runnable

    @patch("redbox.loader.ingester.RunnableParallel")
    @patch("redbox.loader.ingester.ingest_tabular_chunks")
    @patch("redbox.loader.ingester.ingest_chunks")
    @patch("redbox.loader.ingester.MetadataExtraction")
    @patch("redbox.loader.ingester.DocumentExtractionService")
    @pytest.mark.parametrize(
        ("filename", "expected_calls"),
        [
            ("test.csv", [ChunkResolution.normal]),
            ("test.pdf", [ChunkResolution.normal, ChunkResolution.largest]),
        ],
    )
    def test_extraction_strategy(
        self,
        extraction_service_cls,
        metadata_cls,
        ingest_chunks,
        ingest_tabular_chunks,
        runnable_parallel,
        filename,
        expected_calls,
    ):
        extraction, _, _ = self._setup(
            extraction_service_cls,
            metadata_cls,
            ingest_chunks,
            ingest_tabular_chunks,
            runnable_parallel,
            pdf=filename.endswith(".pdf"),
        )

        _ingest_file(filename)

        assert extraction.extract.call_count == len(expected_calls)

        for resolution in expected_calls:
            extraction.extract.assert_any_call(
                file_name=filename,
                chunk_resolution=resolution,
            )

    @patch("redbox.loader.ingester.RunnableParallel")
    @patch("redbox.loader.ingester.ingest_tabular_chunks")
    @patch("redbox.loader.ingester.ingest_chunks")
    @patch("redbox.loader.ingester.MetadataExtraction")
    @patch("redbox.loader.ingester.DocumentExtractionService")
    def test_metadata_uses_normal_elements(
        self,
        extraction_service_cls,
        metadata_cls,
        ingest_chunks,
        ingest_tabular_chunks,
        runnable_parallel,
    ):
        _, metadata, _ = self._setup(
            extraction_service_cls,
            metadata_cls,
            ingest_chunks,
            ingest_tabular_chunks,
            runnable_parallel,
            pdf=True,
        )

        _ingest_file("test.pdf")

        metadata.extract.assert_called_once_with(
            file_name="test.pdf",
            elements=["normal"],
        )

    @patch("redbox.loader.ingester.RunnableParallel")
    @patch("redbox.loader.ingester.ingest_tabular_chunks")
    @patch("redbox.loader.ingester.ingest_chunks")
    @patch("redbox.loader.ingester.MetadataExtraction")
    @patch("redbox.loader.ingester.DocumentExtractionService")
    def test_returns_ingestion_response(
        self,
        extraction_service_cls,
        metadata_cls,
        ingest_chunks,
        ingest_tabular_chunks,
        runnable_parallel,
    ):
        self._setup(
            extraction_service_cls,
            metadata_cls,
            ingest_chunks,
            ingest_tabular_chunks,
            runnable_parallel,
            pdf=True,
        )

        response = _ingest_file("test.pdf")

        assert response == FileIngestionResponse(
            normal_extraction_strategy=IngestExtractionStrategy.textract_document_analysis,
            normal_chunking_strategy=IngestChunkingStrategy.unstructured_chunk_by_title,
            largest_extraction_strategy=IngestExtractionStrategy.pymupdf,
            largest_chunking_strategy=IngestChunkingStrategy.overlapping_pages,
        )

    @patch("redbox.loader.ingester.RunnableParallel")
    @patch("redbox.loader.ingester.ingest_tabular_chunks")
    @patch("redbox.loader.ingester.ingest_chunks")
    @patch("redbox.loader.ingester.MetadataExtraction")
    @patch("redbox.loader.ingester.DocumentExtractionService")
    @pytest.mark.parametrize(
        ("filename", "contains_tabular"),
        [
            ("test.csv", True),
            ("test.xlsx", True),
            ("test.tsv", True),
            ("test.xls", True),
            ("test.pdf", False),
        ],
    )
    def test_tabular_pipeline_selection(
        self,
        extraction_service_cls,
        metadata_cls,
        ingest_chunks,
        ingest_tabular_chunks,
        runnable_parallel,
        filename,
        contains_tabular,
    ):
        self._setup(
            extraction_service_cls,
            metadata_cls,
            ingest_chunks,
            ingest_tabular_chunks,
            runnable_parallel,
            pdf=filename.endswith(".pdf"),
        )

        _ingest_file(filename)

        assert ingest_chunks.call_count == 2
        assert ingest_tabular_chunks.call_count == 2

        chains = runnable_parallel.call_args.args[0]

        assert "normal" in chains
        assert "largest" in chains
        if contains_tabular:
            assert "tabular" in chains
            assert "schematised_tabular" in chains
        else:
            assert "tabular" not in chains
            assert "schematised_tabular" not in chains
