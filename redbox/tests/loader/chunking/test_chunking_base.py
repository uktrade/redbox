import pytest
from datetime import UTC, datetime

from redbox.loader.chunking.base import BaseDocumentChunker
from redbox.models.file import ChunkResolution


class DummyGeneratedMetadata:
    def __init__(self):
        self.name = "test-name"
        self.description = "test-description"
        self.keywords = ["a", "b"]


class DummyChunker(BaseDocumentChunker):
    def chunks(self, s3_key, data, generated_metadata):
        return iter([])


class TestValidation:
    @pytest.mark.parametrize(
        "min_chunk_size,max_chunk_size,overlap_chars,expected_error",
        [
            (0, 10, 0, "min_chunk_size must be > 0"),
            (10, 5, 0, "max_chunk_size must be >= min_chunk_size"),
            (1, 10, -1, "overlap_chars must be >= 0"),
        ],
    )
    def test_base_document_chunker_validation_errors(
        self,
        min_chunk_size,
        max_chunk_size,
        overlap_chars,
        expected_error,
    ):
        with pytest.raises(ValueError) as exc:
            DummyChunker(
                chunk_resolution=ChunkResolution.normal,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                overlap_chars=overlap_chars,
            )

        assert expected_error in str(exc.value)

    def test_base_document_chunker_is_abstract(self):
        with pytest.raises(TypeError):
            BaseDocumentChunker(
                chunk_resolution=ChunkResolution.normal,
                min_chunk_size=1,
                max_chunk_size=10,
                overlap_chars=0,
            )

    def test__chunk_not_implemented(self):
        chunker = DummyChunker(
            chunk_resolution=ChunkResolution.normal,
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=0,
        )

        with pytest.raises(NotImplementedError):
            chunker._chunk()


class TestMetadataBuild:
    @pytest.mark.parametrize(
        "token_count,text,page_number,index",
        [
            (42, "hello world", 1, 0),
            (0, "", 5, 99),
            (123, "some longer text", 2, 3),
        ],
    )
    def test_build_metadata(self, monkeypatch, token_count, text, page_number, index):
        monkeypatch.setattr("redbox.loader.chunking.base.tokeniser", lambda x: token_count)

        chunker = DummyChunker(
            chunk_resolution=ChunkResolution.normal,
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=0,
        )

        generated_metadata = DummyGeneratedMetadata()
        created_datetime = datetime(2026, 1, 1, tzinfo=UTC)

        result = chunker._build_metadata(
            index=index,
            s3_key="s3://bucket/file.txt",
            page_number=page_number,
            created_datetime=created_datetime,
            text=text,
            generated_metadata=generated_metadata,
        )

        assert result["token_count"] == token_count
        assert result["index"] == index
        assert result["page_number"] == page_number
