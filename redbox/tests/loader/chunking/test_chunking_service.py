import pytest
from unittest.mock import MagicMock, patch, ANY

from unstructured.documents.elements import Element

from redbox.loader.chunking.service import DocumentChunkingService
from redbox.models.file import ChunkResolution
from redbox_app.redbox_core.enums import IngestChunkingStrategy


def make_generated_metadata():
    meta = MagicMock()
    meta.name = "doc.pdf"
    meta.description = "desc"
    meta.keywords = ["kw"]
    return meta


def make_service():
    return DocumentChunkingService(
        chunk_resolution=ChunkResolution.normal,
        min_chunk_size=10,
        max_chunk_size=20,
        overlap_chars=5,
    )


class DummyElement(Element):
    pass


class TestTabularChunks:
    def test_tabular_chunker_is_called(self):
        service = make_service()

        expected = iter([MagicMock()])
        service.chunker_tabular.tabular_chunks = MagicMock(return_value=expected)

        strategy, docs = service.tabular_chunks(
            s3_key="s3://file",
            tabular_elements=[{"text": "row"}],
            generated_metadata=make_generated_metadata(),
            include_schema_metadata=True,
        )

        assert strategy == IngestChunkingStrategy.tabular
        assert docs is expected

        service.chunker_tabular.tabular_chunks.assert_called_once_with(
            s3_key="s3://file",
            tabular_elements=[{"text": "row"}],
            generated_metadata=ANY,
            include_schema_metadata=True,
        )


class TestChunks:
    def test_empty_elements_raise(self):
        service = make_service()

        with pytest.raises(ValueError):
            service.chunks(
                "s3://file",
                [],
                make_generated_metadata(),
                chunks_overlap_pages=False,
            )

    def test_page_by_page_chunker_selected(self):
        service = make_service()

        expected = iter([MagicMock()])
        service.chunker_page_by_page.chunks = MagicMock(return_value=expected)

        strategy, docs = service.chunks(
            "s3://file",
            ["page1", "page2"],
            make_generated_metadata(),
            chunks_overlap_pages=False,
        )

        assert strategy == IngestChunkingStrategy.page_by_page
        assert docs is expected

        service.chunker_page_by_page.chunks.assert_called_once()

    def test_joined_pages_chunker_selected(self):
        service = make_service()

        expected = iter([MagicMock()])
        service.chunker_joined_pages.chunks = MagicMock(return_value=expected)

        strategy, docs = service.chunks(
            "s3://file",
            ["page1", "page2"],
            make_generated_metadata(),
            chunks_overlap_pages=True,
        )

        assert strategy == IngestChunkingStrategy.overlapping_pages
        assert docs is expected

        service.chunker_joined_pages.chunks.assert_called_once()

    @patch(
        "redbox.loader.chunking.service.Element",
        DummyElement,
    )
    def test_unstructured_chunker_selected(self):
        service = make_service()

        expected = iter([MagicMock()])
        service.chunker_unstructured.chunks = MagicMock(return_value=expected)

        elements = [DummyElement()]

        strategy, docs = service.chunks(
            "s3://file",
            elements,
            make_generated_metadata(),
            chunks_overlap_pages=False,
        )

        assert strategy == IngestChunkingStrategy.unstructured_chunk_by_title
        assert docs is expected

        service.chunker_unstructured.chunks.assert_called_once_with(
            s3_key="s3://file",
            elements=elements,
            generated_metadata=ANY,
        )

    def test_mixed_input_types_raise(self):
        service = make_service()

        with pytest.raises(TypeError):
            service.chunks(
                "s3://file",
                ["page1", MagicMock(spec=Element)],
                make_generated_metadata(),
                chunks_overlap_pages=False,
            )


@patch("redbox.loader.chunking.service.PageByPageDocumentChunker")
@patch("redbox.loader.chunking.service.JoinedPagesDocumentChunker")
@patch("redbox.loader.chunking.service.UnstructuredDocumentChunker")
@patch("redbox.loader.chunking.service.TabularDocumentChunker")
def test_initialises_chunkers(
    mock_tabular,
    mock_unstructured,
    mock_joined,
    mock_page,
):
    DocumentChunkingService(
        chunk_resolution=ChunkResolution.normal,
        min_chunk_size=10,
        max_chunk_size=20,
        overlap_chars=5,
    )

    mock_page.assert_called_once()
    mock_joined.assert_called_once()
    mock_unstructured.assert_called_once()
    mock_tabular.assert_called_once_with(
        chunk_resolution=ChunkResolution.normal,
    )
