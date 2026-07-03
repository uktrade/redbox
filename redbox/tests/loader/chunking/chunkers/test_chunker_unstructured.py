import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from redbox.loader.chunking.chunkers.unstructured import (
    UnstructuredDocumentChunker,
)
from redbox.models.file import ChunkResolution


def make_chunker(**kwargs):
    defaults = dict(
        chunk_resolution=ChunkResolution.normal,
        min_chunk_size=10,
        max_chunk_size=50,
        overlap_chars=5,
    )
    return UnstructuredDocumentChunker(**{**defaults, **kwargs})


def make_generated_metadata(name="doc.pdf", description="A doc", keywords=None):
    meta = MagicMock()
    meta.name = name
    meta.description = description
    meta.keywords = keywords or ["kw1"]
    return meta


def make_chunk(text, pages):
    return SimpleNamespace(
        text=text,
        metadata=SimpleNamespace(
            orig_elements=[SimpleNamespace(metadata=SimpleNamespace(page_number=p)) for p in pages]
        ),
    )


class TestInitValidation:
    @pytest.mark.parametrize("value", [0, -1])
    def test_invalid_min_chunk_size(self, value):
        with pytest.raises(ValueError):
            make_chunker(min_chunk_size=value)

    def test_max_chunk_size_must_be_ge_min(self):
        with pytest.raises(ValueError):
            make_chunker(min_chunk_size=100, max_chunk_size=99)

    @pytest.mark.parametrize("value", [-1, -100])
    def test_negative_overlap_rejected(self, value):
        with pytest.raises(ValueError):
            make_chunker(overlap_chars=value)


class TestChunkingBehaviour:
    def test_empty_elements_produce_no_chunks(self):
        chunker = make_chunker()

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [],
                make_generated_metadata(),
            )
        )

        assert docs == []

    @patch("redbox.loader.chunking.chunkers.unstructured.chunk_by_title")
    def test_blank_chunks_are_skipped(self, mock_chunk):
        mock_chunk.return_value = [
            make_chunk("", [1]),
            make_chunk("   ", [1]),
        ]

        chunker = make_chunker()

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [MagicMock()],
                make_generated_metadata(),
            )
        )

        assert docs == []

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    @patch("redbox.loader.chunking.chunkers.unstructured.chunk_by_title")
    def test_page_number_taken_from_first_element(self, mock_chunk, _):
        mock_chunk.return_value = [
            make_chunk("hello", [3, 4, 5]),
        ]

        chunker = make_chunker()

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [MagicMock()],
                make_generated_metadata(),
            )
        )

        assert docs[0].metadata["page_number"] == 3

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    @patch("redbox.loader.chunking.chunkers.unstructured.chunk_by_title")
    def test_indices_are_global(self, mock_chunk, _):
        mock_chunk.return_value = [
            make_chunk("one", [1]),
            make_chunk("two", [2]),
            make_chunk("three", [3]),
        ]

        chunker = make_chunker()

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [MagicMock()],
                make_generated_metadata(),
            )
        )

        assert [d.metadata["index"] for d in docs] == [0, 1, 2]


class TestChunkByTitleIntegration:
    @patch("redbox.loader.chunking.chunkers.unstructured.chunk_by_title")
    def test_chunk_by_title_called_with_expected_arguments(self, mock_chunk):
        mock_chunk.return_value = []

        elements = [MagicMock()]

        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=123,
            overlap_chars=7,
        )

        list(
            chunker.chunks(
                "s3://x.pdf",
                elements,
                make_generated_metadata(),
            )
        )

        mock_chunk.assert_called_once_with(
            elements=elements,
            max_characters=123,
            new_after_n_chars=123,
            overlap=7,
            multipage_sections=True,
            overlap_all=True,  # or False depending on your test settings
            include_orig_elements=True,
        )


class TestMetadata:
    @patch("redbox.loader.chunking.base.tokeniser", return_value=5)
    @patch("redbox.loader.chunking.chunkers.unstructured.chunk_by_title")
    def test_metadata_fields(self, mock_chunk, _):
        mock_chunk.return_value = [
            make_chunk("hello world", [2]),
        ]

        chunker = make_chunker()

        docs = list(
            chunker.chunks(
                "s3://my/file.pdf",
                [MagicMock()],
                make_generated_metadata(name="file.pdf"),
            )
        )

        m = docs[0].metadata

        assert docs[0].page_content == "hello world"
        assert m["uri"] == "s3://my/file.pdf"
        assert m["name"] == "file.pdf"
        assert m["token_count"] == 5
        assert m["index"] == 0
        assert m["page_number"] == 2
        assert m["chunk_resolution"] == ChunkResolution.normal

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    @patch("redbox.loader.chunking.chunkers.unstructured.chunk_by_title")
    def test_created_datetime_is_shared(self, mock_chunk, _):
        mock_chunk.return_value = [
            make_chunk("one", [1]),
            make_chunk("two", [2]),
        ]

        chunker = make_chunker()

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [MagicMock()],
                make_generated_metadata(),
            )
        )

        timestamps = {d.metadata["created_datetime"] for d in docs}

        assert len(timestamps) == 1
