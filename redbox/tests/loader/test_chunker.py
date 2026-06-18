import pytest
from unittest.mock import patch, MagicMock
from redbox.loader.chunker import DocumentChunker
from redbox.models.file import ChunkResolution


def make_chunker(**kwargs) -> DocumentChunker:
    defaults = dict(
        chunk_resolution=ChunkResolution.normal,
        min_chunk_size=10,
        max_chunk_size=50,
        overlap_chars=5,
    )
    return DocumentChunker(**{**defaults, **kwargs})


def make_generated_metadata(name="doc.pdf", description="A doc", keywords=None):
    meta = MagicMock()
    meta.name = name
    meta.description = description
    meta.keywords = keywords or ["kw1"]
    return meta


class TestPageForOffset:
    @pytest.mark.parametrize(
        "offset, page_offsets, expected_page",
        [
            (0, [0, 100, 200], 1),  # start of first page
            (50, [0, 100, 200], 1),  # mid first page
            (100, [0, 100, 200], 2),  # exact start of second page
            (150, [0, 100, 200], 2),  # mid second page
            (200, [0, 100, 200], 3),  # exact start of third page
            (999, [0, 100, 200], 3),  # past last boundary -> still page 3
            (0, [0], 1),  # single page document
        ],
    )
    def test_returns_correct_page(self, offset, page_offsets, expected_page):
        assert make_chunker()._page_for_offset(offset, page_offsets) == expected_page


class TestParsePages:
    @pytest.mark.parametrize(
        "pages, expected_text, expected_offsets",
        [
            ([], "", []),
            (["hello"], "hello", [0]),
            (["abc", "def"], "abc\n\ndef", [0, 5]),
            (["aa", "bbb", "cccc"], "aa\n\nbbb\n\ncccc", [0, 4, 9]),
        ],
    )
    def test_parse_pages(self, pages, expected_text, expected_offsets):
        text, offsets = make_chunker()._parse_pages(pages)
        assert text == expected_text
        assert offsets == expected_offsets


class TestChunkText:
    @pytest.mark.parametrize(
        "text, min_size, max_size, overlap, expected_count",
        [
            ("", 1, 50, 0, 0),  # empty -> no chunks
            ("hello", 1, 100, 0, 1),  # shorter than max -> one chunk
            ("abcde", 1, 5, 0, 1),  # exactly max -> one chunk
            ("a" * 100, 1, 50, 0, 2),  # 100 / 50 = 2 exact chunks
            ("a" * 100, 1, 50, 10, 3),  # overlap causes extra chunk
            ("a" * 10, 1, 50, 0, 1),  # much shorter than max -> one chunk
        ],
    )
    def test_chunk_count(self, text, min_size, max_size, overlap, expected_count):
        chunker = make_chunker(min_chunk_size=min_size, max_chunk_size=max_size, overlap_chars=overlap)
        assert len(chunker._chunk_text(text)) == expected_count

    @pytest.mark.parametrize(
        "text, max_size, overlap, expected_offsets",
        [
            ("a" * 20, 10, 0, [0, 10]),  # no overlap: clean stride
            ("a" * 30, 10, 0, [0, 10, 20]),  # three clean strides
            ("a" * 18, 10, 2, [0, 8, 16]),  # overlap=2 -> stride=8
        ],
    )
    def test_start_offsets(self, text, max_size, overlap, expected_offsets):
        chunker = make_chunker(min_chunk_size=1, max_chunk_size=max_size, overlap_chars=overlap)
        offsets = [start for _, start in chunker._chunk_text(text)]
        assert offsets == expected_offsets

    @pytest.mark.parametrize("overlap", [2, 3, 5])
    def test_overlap_produces_shared_content(self, overlap):
        # tail of chunk[n] must equal head of chunk[n+1]
        chunker = make_chunker(min_chunk_size=1, max_chunk_size=10, overlap_chars=overlap)
        text = "0123456789ABCDEFGHIJ"  # 20 chars, guaranteed multiple chunks
        result = chunker._chunk_text(text)
        assert len(result) >= 2
        for (chunk_a, _), (chunk_b, _) in zip(result, result[1:]):
            assert chunk_b.startswith(chunk_a[-overlap:])

    def test_trailing_chunk_below_min_is_dropped(self):
        # 12 chars, max=10, min=5: second chunk is 2 chars -> dropped (unless first)
        chunker = make_chunker(min_chunk_size=5, max_chunk_size=10, overlap_chars=0)
        result = chunker._chunk_text("a" * 12)
        sizes = [len(c) for c, _ in result]
        assert all(s >= 5 or i == 0 for i, s in enumerate(sizes))


class TestChunks:
    @pytest.mark.parametrize(
        "chunk_config, pages, expected_page_nums",
        [
            # each 98-char page fills one 100-char chunk (98 content + 2 separator)
            ((1, 100, 0), ["a" * 98, "b" * 98, "c" * 98], [1, 2, 3]),
            # all short pages collapse into a single chunk
            ((1, 500, 0), ["page one", "page two", "page three"], [1]),
        ],
    )
    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_page_numbers(self, _mock_tok, chunk_config, pages, expected_page_nums):
        min_c, max_c, overlap = chunk_config
        chunker = make_chunker(min_chunk_size=min_c, max_chunk_size=max_c, overlap_chars=overlap)
        docs = list(chunker.chunks("s3://x.pdf", pages, make_generated_metadata()))
        assert [d.metadata["page_number"] for d in docs] == expected_page_nums

    @pytest.mark.parametrize(
        "pages, expected_min_docs",
        [
            ([], 0),
            (["short"], 1),
            (["a" * 200], 2),  # 200 chars / max_chunk=100 = 2
            (["a" * 200, "b" * 200], 4),  # two 200-char pages -> 4 chunks
        ],
    )
    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_doc_count_scales_with_content(self, _mock_tok, pages, expected_min_docs):
        chunker = make_chunker(min_chunk_size=1, max_chunk_size=100, overlap_chars=0)
        docs = list(chunker.chunks("s3://x.pdf", pages, make_generated_metadata()))
        assert len(docs) >= expected_min_docs

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
        docs = list(chunker.chunks("s3://x.pdf", ["some content"], make_generated_metadata()))
        assert docs[0].metadata["chunk_resolution"] == resolution

    @patch("redbox.loader.chunker.tokeniser", return_value=5)
    def test_metadata_fields(self, _mock_tok):
        chunker = make_chunker(min_chunk_size=1, max_chunk_size=100, overlap_chars=0)
        docs = list(chunker.chunks("s3://my/file.pdf", ["hello world"], make_generated_metadata(name="file.pdf")))
        meta = docs[0].metadata
        assert meta["uri"] == "s3://my/file.pdf"
        assert meta["name"] == "file.pdf"
        assert meta["token_count"] == 5
        assert meta["index"] == 0
        assert meta["page_number"] == 1


class TestTabularChunks:
    @pytest.mark.parametrize(
        "elements, expected_count",
        [
            (None, 0),  # None guard
            ([], 0),  # empty list
            ([{"text": "row 1"}], 1),  # single row
            ([{"text": "row 1"}, {"text": "row 2"}], 2),  # multiple rows
        ],
    )
    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_doc_count(self, _mock_tok, elements, expected_count):
        docs = list(make_chunker().tabular_chunks("s3://x.csv", elements, make_generated_metadata()))
        assert len(docs) == expected_count

    @pytest.mark.parametrize(
        "include_schema, extra_meta, expect_merged",
        [
            (True, {"col_name": "revenue"}, True),
            (False, {"col_name": "revenue"}, False),
        ],
    )
    @patch("redbox.loader.chunker.tokeniser", return_value=1)
    def test_schema_metadata_merging(self, _mock_tok, include_schema, extra_meta, expect_merged):
        chunker = make_chunker(include_schema_metadata=include_schema)
        elements = [{"text": "data", "metadata": extra_meta}]
        docs = list(chunker.tabular_chunks("s3://x.csv", elements, make_generated_metadata()))
        for key in extra_meta:
            assert (key in docs[0].metadata) == expect_merged

    @patch("redbox.loader.chunker.tokeniser", return_value=2)
    def test_metadata_fields(self, _mock_tok):
        elements = [{"text": "cell content"}]
        docs = list(
            make_chunker().tabular_chunks("s3://my/table.csv", elements, make_generated_metadata(name="table.csv"))
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
        elements = [{"text": f"row {i}"} for i in range(5)]
        docs = list(make_chunker().tabular_chunks("s3://x.csv", elements, make_generated_metadata()))
        assert [d.metadata["index"] for d in docs] == list(range(5))
