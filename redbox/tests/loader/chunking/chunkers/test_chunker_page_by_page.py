import pytest
from unittest.mock import patch, MagicMock

from redbox.loader.chunking.chunkers.page_by_page import PageByPageDocumentChunker
from redbox.models.file import ChunkResolution


def make_chunker(**kwargs):
    defaults = dict(
        chunk_resolution=ChunkResolution.normal,
        min_chunk_size=10,
        max_chunk_size=50,
        overlap_chars=5,
    )
    return PageByPageDocumentChunker(**{**defaults, **kwargs})


def make_generated_metadata(name="doc.pdf", description="A doc", keywords=None):
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

    def test_max_chunk_size_must_be_ge_min(self):
        with pytest.raises(ValueError):
            make_chunker(min_chunk_size=100, max_chunk_size=99)

    @pytest.mark.parametrize("value", [-1, -100])
    def test_negative_overlap_rejected(self, value):
        with pytest.raises(ValueError):
            make_chunker(overlap_chars=value)


class TestChunkingBehaviour:
    def test_empty_pages_produce_no_chunks(self):
        chunker = make_chunker()

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [],
                make_generated_metadata(),
            )
        )

        assert docs == []

    @pytest.mark.parametrize(
        "pages",
        [
            [""],
            [" "],
            ["\n"],
            ["\t"],
            ["   \n\n   "],
        ],
    )
    def test_blank_pages_produce_no_docs(self, pages):
        chunker = make_chunker()

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                pages,
                make_generated_metadata(),
            )
        )

        assert docs == []

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_page_numbers_are_correct(self, _):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=0,
        )

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                ["a" * 25, "b" * 25],
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

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_indices_are_global(self, _):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=0,
        )

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                ["a" * 25, "b" * 25],
                make_generated_metadata(),
            )
        )

        assert [d.metadata["index"] for d in docs] == list(range(len(docs)))


class TestChunkSizing:
    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_chunk_count_is_deterministic(self, _):
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

        # 25 chars / 10 chunk size => 3 chunks (10,10,5)
        assert len(docs) == 3

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_overlap_affects_chunk_density(self, _):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=2,
        )

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                ["01234567890123456789"],
                make_generated_metadata(),
            )
        )

        # overlap increases number of chunks
        assert len(docs) > 1

    @pytest.mark.parametrize(
        "text,max_chunk_size,expected_lengths",
        [
            ("a" * 12, 10, [10, 2]),
            ("a" * 10, 10, [10]),
            ("a" * 11, 10, [10, 1]),
        ],
    )
    def test_chunk_boundaries(self, text, max_chunk_size, expected_lengths):
        chunker = make_chunker(
            max_chunk_size=max_chunk_size,
            overlap_chars=0,
            min_chunk_size=1,
        )

        chunks = list(chunker._chunk_text(text, 1))
        assert [len(c[0]) for c in chunks] == expected_lengths

    def test_all_original_characters_appear_in_chunks(self):
        chunker = make_chunker(
            max_chunk_size=10,
            overlap_chars=2,
            min_chunk_size=1,
        )

        text = "0123456789AB"

        chunks = [c[0] for c in chunker._chunk_text(text, 1)]

        covered = set("".join(chunks))

        for ch in text:
            assert ch in covered


class TestOverlapBehaviour:
    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_chunks_share_overlap_prefix(self, _):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=3,
        )

        text = "0123456789ABCDEFGHIJ"

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [text],
                make_generated_metadata(),
            )
        )

        contents = [d.page_content for d in docs]

        for a, b in zip(contents, contents[1:]):
            assert b.startswith(a[-3:])

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_invalid_overlap_raises_at_runtime(self, _):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=10,  # invalid: equals max
        )

        with pytest.raises(ValueError):
            list(
                chunker.chunks(
                    "s3://x.pdf",
                    ["a" * 100],
                    make_generated_metadata(),
                )
            )


class TestMetadata:
    @patch("redbox.loader.chunking.base.tokeniser", return_value=5)
    def test_metadata_fields(self, _):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=0,
        )

        docs = list(
            chunker.chunks(
                "s3://my/file.pdf",
                ["hello world"],
                make_generated_metadata(name="file.pdf"),
            )
        )

        m = docs[0].metadata

        assert m["uri"] == "s3://my/file.pdf"
        assert m["name"] == "file.pdf"
        assert m["token_count"] == 5
        assert m["index"] == 0
        assert m["page_number"] == 1
        assert m["chunk_resolution"] == ChunkResolution.normal

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_created_datetime_is_shared(self, _):
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
