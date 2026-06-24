import pytest
from unittest.mock import patch, MagicMock

from redbox.loader.chunking.chunkers.page_by_page import PageByPageDocumentChunker
from redbox.models.file import ChunkResolution


def make_chunker(**kwargs) -> PageByPageDocumentChunker:
    defaults = dict(
        chunk_resolution=ChunkResolution.normal,
        min_chunk_size=10,
        max_chunk_size=50,
        overlap_chars=5,
    )
    return PageByPageDocumentChunker(**{**defaults, **kwargs})


def make_generated_metadata(
    name="doc.pdf",
    description="A doc",
    keywords=None,
):
    meta = MagicMock()
    meta.name = name
    meta.description = description
    meta.keywords = keywords or ["kw1"]
    return meta


class TestInitValidation:
    @pytest.mark.parametrize("value", [0, -1])
    def test_invalid_min_chunk_size(self, value):
        with pytest.raises(ValueError):
            make_chunker(min_chunk_size=value)

    def test_max_chunk_size_must_be_greater_than_or_equal_to_min(self):
        with pytest.raises(ValueError):
            make_chunker(
                min_chunk_size=100,
                max_chunk_size=99,
            )

    @pytest.mark.parametrize("value", [-1, -100])
    def test_negative_overlap_chars_rejected(self, value):
        with pytest.raises(ValueError):
            make_chunker(overlap_chars=value)


class TestChunkText:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            " ",
            "\n",
            "\n\n\n",
            "\t",
            "   \n\t   ",
        ],
    )
    def test_blank_text_returns_no_chunks(self, text):
        assert make_chunker()._chunk_text(text) == []

    @pytest.mark.parametrize(
        "text, min_size, max_size, overlap, expected_count",
        [
            ("hello", 1, 100, 0, 1),
            ("abcde", 1, 5, 0, 1),
            ("a" * 100, 1, 50, 0, 2),
            ("a" * 100, 1, 50, 10, 3),
            ("a" * 10, 1, 50, 0, 1),
        ],
    )
    def test_chunk_count(self, text, min_size, max_size, overlap, expected_count):
        chunker = make_chunker(
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            overlap_chars=overlap,
        )

        assert len(chunker._chunk_text(text)) == expected_count

    @pytest.mark.parametrize("overlap", [2, 3, 5])
    def test_overlap_produces_shared_content(self, overlap):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=overlap,
        )

        text = "0123456789ABCDEFGHIJ"

        chunks = chunker._chunk_text(text)

        assert len(chunks) >= 2

        for chunk_a, chunk_b in zip(chunks, chunks[1:]):
            assert chunk_b.startswith(chunk_a[-overlap:])

    # def test_trailing_chunk_below_min_is_merged(self):
    #     chunker = make_chunker(
    #         min_chunk_size=5,
    #         max_chunk_size=10,
    #         overlap_chars=0,
    #     )
    #     chunks = chunker._chunk_text("a" * 12)

    #     # The 2-char tail should be merged into the previous chunk.
    #     assert [len(chunk) for chunk in chunks] == [12]

    # def test_no_text_is_lost_when_final_chunk_is_merged(self):
    #     chunker = make_chunker(
    #         min_chunk_size=5,
    #         max_chunk_size=10,
    #         overlap_chars=2,
    #     )
    #     text = "0123456789AB"

    #     # 12 chars
    #     chunks = chunker._chunk_text(text)

    #     # Every character from the original text should appear
    #     # in at least one chunk.
    #     reconstructed = "".join(chunks)

    #     for char in text:
    #         assert char in reconstructed

    #     # More importantly, the unique tail ("AB") must survive.
    #     assert reconstructed.endswith("AB")

    @pytest.mark.parametrize(
        "overlap",
        [
            10,
            11,
        ],
    )
    def test_overlap_must_be_less_than_max_chunk_size(self, overlap):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=overlap,
        )

        with pytest.raises(ValueError):
            chunker._chunk_text("a" * 100)


class TestChunks:
    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_empty_pages_produce_no_documents(self, _mock_tok):
        docs = list(
            make_chunker().chunks(
                "s3://x.pdf",
                [],
                make_generated_metadata(),
            )
        )

        assert docs == []

    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_chunking_is_page_local(self, _mock_tok):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=0,
        )

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [
                    "a" * 25,
                    "b" * 25,
                ],
                make_generated_metadata(),
            )
        )

        assert [d.metadata["page_number"] for d in docs] == [
            1,
            1,
            1,
            2,
            2,
            2,
        ]

    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_chunk_indices_are_global_across_pages(self, _mock_tok):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=0,
        )

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [
                    "a" * 25,
                    "b" * 25,
                ],
                make_generated_metadata(),
            )
        )

        assert [d.metadata["index"] for d in docs] == list(range(len(docs)))

    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_doc_count_scales_with_content(self, _mock_tok):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=100,
            overlap_chars=0,
        )

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [
                    "a" * 200,
                    "b" * 200,
                ],
                make_generated_metadata(),
            )
        )

        assert len(docs) == 4

    @pytest.mark.parametrize(
        "resolution",
        [
            ChunkResolution.normal,
            ChunkResolution.large,
        ],
    )
    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_chunk_resolution_in_metadata(self, _mock_tok, resolution):
        chunker = make_chunker(chunk_resolution=resolution)

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                ["some content"],
                make_generated_metadata(),
            )
        )

        assert docs[0].metadata["chunk_resolution"] == resolution

    @patch("redbox.loader.chunker.tokeniser", return_value=5)
    def test_metadata_fields(self, _mock_tok):
        docs = list(
            make_chunker(
                min_chunk_size=1,
                max_chunk_size=100,
                overlap_chars=0,
            ).chunks(
                "s3://my/file.pdf",
                ["hello world"],
                make_generated_metadata(name="file.pdf"),
            )
        )

        meta = docs[0].metadata

        assert meta["uri"] == "s3://my/file.pdf"
        assert meta["name"] == "file.pdf"
        assert meta["token_count"] == 5
        assert meta["index"] == 0
        assert meta["page_number"] == 1

    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_all_chunks_share_same_created_datetime(self, _mock_tok):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=0,
        )

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                ["a" * 25],
                make_generated_metadata(),
            )
        )

        timestamps = {d.metadata["created_datetime"] for d in docs}

        assert len(timestamps) == 1


class TestTabularChunks:
    @pytest.mark.parametrize(
        "elements, expected_count",
        [
            (None, 0),
            ([], 0),
            ([{"text": "row 1"}], 1),
            ([{"text": "row 1"}, {"text": "row 2"}], 2),
        ],
    )
    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_doc_count(self, _mock_tok, elements, expected_count):
        docs = list(
            make_chunker().tabular_chunks(
                "s3://x.csv",
                elements,
                make_generated_metadata(),
            )
        )

        assert len(docs) == expected_count

    @pytest.mark.parametrize(
        "include_schema, extra_meta, expect_merged",
        [
            (True, {"col_name": "revenue"}, True),
            (False, {"col_name": "revenue"}, False),
        ],
    )
    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_schema_metadata_merging(
        self,
        _mock_tok,
        include_schema,
        extra_meta,
        expect_merged,
    ):
        chunker = make_chunker(
            include_schema_metadata=include_schema,
        )

        docs = list(
            chunker.tabular_chunks(
                "s3://x.csv",
                [{"text": "data", "metadata": extra_meta}],
                make_generated_metadata(),
            )
        )

        for key in extra_meta:
            assert (key in docs[0].metadata) == expect_merged

    @patch("redbox.loader.chunker.tokeniser", return_value=2)
    def test_metadata_fields(self, _mock_tok):
        docs = list(
            make_chunker().tabular_chunks(
                "s3://my/table.csv",
                [{"text": "cell content"}],
                make_generated_metadata(name="table.csv"),
            )
        )

        meta = docs[0].metadata

        assert meta["uri"] == "s3://my/table.csv"
        assert meta["name"] == "table.csv"
        assert meta["chunk_resolution"] == ChunkResolution.tabular
        assert meta["page_number"] == 1
        assert meta["token_count"] == 2
        assert meta["index"] == 0

    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_index_increments(self, _mock_tok):
        docs = list(
            make_chunker().tabular_chunks(
                "s3://x.csv",
                [{"text": f"row {i}"} for i in range(5)],
                make_generated_metadata(),
            )
        )

        assert [d.metadata["index"] for d in docs] == list(range(5))

    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_all_rows_share_same_created_datetime(self, _mock_tok):
        docs = list(
            make_chunker().tabular_chunks(
                "s3://x.csv",
                [
                    {"text": "row 1"},
                    {"text": "row 2"},
                    {"text": "row 3"},
                ],
                make_generated_metadata(),
            )
        )

        timestamps = {d.metadata["created_datetime"] for d in docs}

        assert len(timestamps) == 1
