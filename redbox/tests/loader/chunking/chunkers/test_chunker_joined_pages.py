import pytest
from unittest.mock import MagicMock, patch

from redbox.loader.chunking.chunkers.joined_pages import JoinedPagesDocumentChunker
from redbox.models.file import ChunkResolution


def make_chunker(**kwargs):
    defaults = dict(
        chunk_resolution=ChunkResolution.normal,
        min_chunk_size=10,
        max_chunk_size=50,
        overlap_chars=5,
    )
    return JoinedPagesDocumentChunker(**{**defaults, **kwargs})


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


class TestJoinPages:
    def test_join_pages_returns_expected_text_and_spans(self):
        chunker = make_chunker()

        text, spans = chunker._join_pages(
            [
                "abc",
                "de",
            ]
        )

        assert text == "abc\nde\n"
        assert spans == [
            (0, 3, 1),
            (4, 6, 2),
        ]

    def test_joined_pages_include_separator(self):
        chunker = make_chunker()

        text, _ = chunker._join_pages(
            [
                "hello",
                "world",
            ]
        )

        assert text == "hello\nworld\n"


class TestPageMapping:
    @pytest.mark.parametrize(
        "start,end,expected",
        [
            (0, 2, [1]),
            (2, 5, [1, 2]),
            (4, 6, [2]),
        ],
    )
    def test_get_page_for_chunk(self, start, end, expected):
        chunker = make_chunker()

        spans = [
            (0, 3, 1),
            (4, 6, 2),
        ]

        assert chunker._get_page_for_chunk(start, end, spans) == expected

    def test_chunk_page_mapping_when_chunk_contains_only_separator(self):
        chunker = make_chunker()

        _, spans = chunker._join_pages(
            [
                "abc",
                "def",
            ]
        )

        assert chunker._get_page_for_chunk(3, 4, spans) == [1]


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
    def test_indices_are_global(self, _):
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

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_chunk_can_span_page_boundary(self, _):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=0,
        )

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [
                    "AAAAA",
                    "BBBBB",
                ],
                make_generated_metadata(),
            )
        )

        assert docs[0].page_content == "AAAAA\nBBBB"

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_page_numbers_are_first_page_covered(self, _):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=8,
            overlap_chars=0,
        )

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [
                    "AAAAA",
                    "BBBBB",
                ],
                make_generated_metadata(),
            )
        )

        assert [d.metadata["page_number"] for d in docs] == [
            1,
            2,
        ]


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

        assert len(docs) > 1

    def test_joined_text_preserves_all_page_content(self):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=2,
        )

        docs = list(
            chunker.chunks(
                "s3://x.pdf",
                [
                    "0123",
                    "4567",
                ],
                make_generated_metadata(),
            )
        )

        combined = "".join(d.page_content for d in docs)

        for ch in "01234567":
            assert ch in combined

    @patch("redbox.loader.chunking.base.tokeniser", return_value=1)
    def test_invalid_overlap_raises_at_runtime(self, _):
        chunker = make_chunker(
            min_chunk_size=1,
            max_chunk_size=10,
            overlap_chars=10,
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
