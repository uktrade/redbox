from unittest.mock import MagicMock, patch

from redbox.loader.chunking.chunkers.tabular import TabularDocumentChunker
from redbox.models.file import ChunkResolution


def make_chunker(**kwargs):
    defaults = dict(
        chunk_resolution=ChunkResolution.normal,
    )
    return TabularDocumentChunker(**{**defaults, **kwargs})


def make_generated_metadata(name="doc.pdf", description="A doc", keywords=None):
    meta = MagicMock()
    meta.name = name
    meta.description = description
    meta.keywords = keywords or ["kw1"]
    return meta


class TestTabularChunks:
    def test_empty_elements_produce_no_docs(self):
        chunker = make_chunker()

        docs = list(
            chunker.tabular_chunks(
                "s3://x.pdf",
                [],
                make_generated_metadata(),
                include_schema_metadata=False,
            )
        )

        assert docs == []

    def test_none_elements_produce_no_docs(self):
        chunker = make_chunker()

        docs = list(
            chunker.tabular_chunks(
                "s3://x.pdf",
                None,
                make_generated_metadata(),
                include_schema_metadata=False,
            )
        )

        assert docs == []

    @patch("redbox.loader.chunking.base.tokeniser", return_value=5)
    def test_metadata_fields(self, _):
        chunker = make_chunker()

        docs = list(
            chunker.tabular_chunks(
                "s3://my/file.pdf",
                [{"text": "hello world"}],
                make_generated_metadata(name="file.pdf"),
                include_schema_metadata=False,
            )
        )

        m = docs[0].metadata

        assert docs[0].page_content == "hello world"
        assert m["uri"] == "s3://my/file.pdf"
        assert m["name"] == "file.pdf"
        assert m["token_count"] == 5
        assert m["index"] == 0
        assert m["page_number"] == 1
        assert m["chunk_resolution"] == ChunkResolution.normal

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_indices_are_incremented(self, _):
        chunker = make_chunker()

        docs = list(
            chunker.tabular_chunks(
                "s3://x.pdf",
                [
                    {"text": "row1"},
                    {"text": "row2"},
                    {"text": "row3"},
                ],
                make_generated_metadata(),
                include_schema_metadata=False,
            )
        )

        assert [d.metadata["index"] for d in docs] == [0, 1, 2]

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_created_datetime_is_shared(self, _):
        chunker = make_chunker()

        docs = list(
            chunker.tabular_chunks(
                "s3://x.pdf",
                [
                    {"text": "row1"},
                    {"text": "row2"},
                ],
                make_generated_metadata(),
                include_schema_metadata=False,
            )
        )

        timestamps = {d.metadata["created_datetime"] for d in docs}
        assert len(timestamps) == 1

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_schema_metadata_is_included(self, _):
        chunker = make_chunker()

        docs = list(
            chunker.tabular_chunks(
                "s3://x.pdf",
                [
                    {
                        "text": "row1",
                        "metadata": {
                            "table_name": "customers",
                            "column_count": 4,
                        },
                    }
                ],
                make_generated_metadata(),
                include_schema_metadata=True,
            )
        )

        metadata = docs[0].metadata

        assert metadata["table_name"] == "customers"
        assert metadata["column_count"] == 4

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_schema_metadata_is_not_included(self, _):
        chunker = make_chunker()

        docs = list(
            chunker.tabular_chunks(
                "s3://x.pdf",
                [
                    {
                        "text": "row1",
                        "metadata": {
                            "table_name": "customers",
                        },
                    }
                ],
                make_generated_metadata(),
                include_schema_metadata=False,
            )
        )

        assert "table_name" not in docs[0].metadata

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_missing_schema_metadata_is_handled(self, _):
        chunker = make_chunker()

        docs = list(
            chunker.tabular_chunks(
                "s3://x.pdf",
                [{"text": "row1"}],
                make_generated_metadata(),
                include_schema_metadata=True,
            )
        )

        assert docs[0].metadata["index"] == 0
        assert docs[0].page_content == "row1"
