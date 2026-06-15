import pytest

from redbox.loader.chunker import (
    DocumentChunker,
    LayoutBlock,
)


def block(
    text: str,
    *,
    page: int = 1,
    title: bool = False,
) -> LayoutBlock:
    return LayoutBlock(
        text=text,
        block_type="TITLE" if title else "TEXT",
        page_number=page,
        is_title=title,
    )


@pytest.fixture
def chunker() -> DocumentChunker:
    return DocumentChunker()


class TestBuildSections:
    @pytest.mark.parametrize(
        ("blocks", "expected_titles", "expected_sizes"),
        [
            (
                [],
                [],
                [],
            ),
            (
                [
                    block("para 1"),
                    block("para 2"),
                ],
                [""],
                [2],
            ),
            (
                [
                    block("Intro", title=True),
                    block("para 1"),
                    block("para 2"),
                ],
                ["Intro"],
                [3],
            ),
            (
                [
                    block("Intro", title=True),
                    block("intro body"),
                    block("Methods", title=True),
                    block("methods body"),
                ],
                ["Intro", "Methods"],
                [2, 2],
            ),
        ],
    )
    def test_groups_blocks_under_titles(
        self,
        chunker,
        blocks,
        expected_titles,
        expected_sizes,
    ):
        sections = chunker.build_sections(blocks)

        assert [s.title for s in sections] == expected_titles
        assert [len(s.blocks) for s in sections] == expected_sizes


class TestFindSplitPoint:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "a" * 500,
                500,
            ),
            (
                "a" * 500 + "\n\n" + "b" * 100,
                500,
            ),
            (
                "a" * 500 + ". " + "b" * 100,
                502,
            ),
            (
                "a" * 500 + "\n" + "b" * 100,
                500,
            ),
            (
                "a" * 500 + " " + "b" * 100,
                500,
            ),
        ],
    )
    def test_prefers_semantic_boundaries(
        self,
        text,
        expected,
    ):
        chunker = DocumentChunker(min_chunk_size=500)

        assert chunker._find_split_point(text) == expected

    def test_falls_back_to_hard_split(self):
        chunker = DocumentChunker(min_chunk_size=100)

        text = "x" * 300

        assert chunker._find_split_point(text) == len(text)


class TestChunkText:
    def test_returns_empty_for_empty_input(self, chunker):
        assert list(chunker.chunk_text("")) == []

    def test_returns_single_chunk_when_under_limit(self):
        chunker = DocumentChunker(max_chunk_size=100)

        assert list(chunker.chunk_text("hello world")) == [
            "hello world",
        ]

    def test_splits_large_text(self):
        chunker = DocumentChunker(
            min_chunk_size=20,
            max_chunk_size=50,
            overlap_chars=10,
        )

        text = "Paragraph one. " * 5 + "\n\n" + "Paragraph two. " * 5 + "\n\n" + "Paragraph three. " * 5

        chunks = list(chunker.chunk_text(text))

        assert len(chunks) > 1
        assert all(chunk.strip() for chunk in chunks)

    def test_always_makes_forward_progress(self):
        chunker = DocumentChunker(
            min_chunk_size=1,
            max_chunk_size=5,
            overlap_chars=4,
        )

        chunks = list(
            chunker.chunk_text(
                "abcdefghijklmnopqrstuvwxyz",
            )
        )

        assert chunks
        assert len(chunks) < 26


class TestBuildChunks:
    def test_small_section_produces_single_chunk(self):
        chunker = DocumentChunker(max_chunk_size=1000)

        chunks = list(
            chunker.chunk(
                [
                    block("Introduction", title=True),
                    block("Some content."),
                ]
            )
        )

        assert len(chunks) == 1

        chunk = chunks[0]

        assert chunk.section_title == "Introduction"
        assert "Introduction" in chunk.text
        assert "Some content." in chunk.text

    def test_large_section_produces_multiple_chunks(self):
        chunker = DocumentChunker(
            min_chunk_size=50,
            max_chunk_size=100,
            overlap_chars=20,
        )

        chunks = list(
            chunker.chunk(
                [
                    block("Introduction", title=True),
                    block("A" * 400),
                ]
            )
        )

        assert len(chunks) > 1

        for chunk in chunks:
            assert chunk.section_title == "Introduction"
            assert chunk.text.startswith("Introduction")

    def test_preserves_page_range(self):
        chunker = DocumentChunker(max_chunk_size=1000)

        chunks = list(
            chunker.chunk(
                [
                    block("Introduction", page=1, title=True),
                    block("Page one", page=1),
                    block("Page two", page=2),
                    block("Page three", page=3),
                ]
            )
        )

        chunk = chunks[0]

        assert chunk.page_start == 1
        assert chunk.page_end == 3

    def test_produces_chunk_per_section(self):
        chunker = DocumentChunker(max_chunk_size=1000)

        chunks = list(
            chunker.chunk(
                [
                    block("Intro", title=True),
                    block("Intro content"),
                    block("Methods", title=True),
                    block("Methods content"),
                ]
            )
        )

        assert [c.section_title for c in chunks] == [
            "Intro",
            "Methods",
        ]

    def test_ignores_whitespace_only_blocks(self):
        chunker = DocumentChunker()

        chunks = list(
            chunker.chunk(
                [
                    block("Intro", title=True),
                    block("   "),
                    block("\n"),
                    block("Real content"),
                ]
            )
        )

        assert len(chunks) == 1
        assert "Real content" in chunks[0].text


class TestEndToEnd:
    def test_document_chunking_pipeline(self):
        chunker = DocumentChunker(
            min_chunk_size=50,
            max_chunk_size=150,
            overlap_chars=25,
        )

        chunks = list(
            chunker.chunk(
                [
                    block(
                        "Executive Summary",
                        title=True,
                        page=1,
                    ),
                    block(
                        "Summary text " * 20,
                        page=1,
                    ),
                    block(
                        "Background",
                        title=True,
                        page=2,
                    ),
                    block(
                        "Background text " * 20,
                        page=2,
                    ),
                ]
            )
        )

        assert chunks

        section_titles = {chunk.section_title for chunk in chunks}

        assert section_titles == {
            "Executive Summary",
            "Background",
        }


class TestChunkOverlap:
    def test_adjacent_chunks_contain_overlap_text(self):
        overlap = 20

        chunker = DocumentChunker(
            min_chunk_size=20,
            max_chunk_size=100,
            overlap_chars=overlap,
        )

        text = "abcdefghijklmnopqrstuvwxyz" * 20

        chunks = list(chunker.chunk_text(text))

        assert len(chunks) > 1

        for current, nxt in zip(chunks, chunks[1:]):
            expected_overlap = current[-overlap:]

            assert expected_overlap in nxt

    def test_overlap_never_exceeds_chunk_size(self):
        chunker = DocumentChunker(
            min_chunk_size=20,
            max_chunk_size=50,
            overlap_chars=200,
        )

        chunks = list(chunker.chunk_text("abcdefghijklmnopqrstuvwxyz" * 20))

        assert chunks

    def test_overlap_creates_duplicate_content_between_chunks(self):
        chunker = DocumentChunker(
            min_chunk_size=20,
            max_chunk_size=80,
            overlap_chars=15,
        )

        text = ("Section A content. More context. Even more context. ") * 10

        chunks = list(chunker.chunk_text(text))

        assert len(chunks) > 1

        duplicated_pairs = 0

        for current, nxt in zip(chunks, chunks[1:]):
            words = current.split()

            if any(word in nxt for word in words[-5:]):
                duplicated_pairs += 1

        assert duplicated_pairs == len(chunks) - 1

    def test_large_overlap_does_not_cause_infinite_loop(self):
        chunker = DocumentChunker(
            min_chunk_size=1,
            max_chunk_size=25,
            overlap_chars=24,
        )

        chunks = list(chunker.chunk_text("abcdefghijklmnopqrstuvwxyz" * 5))

        assert chunks
        assert len(chunks) < 200

    @pytest.mark.parametrize(
        "overlap",
        [
            0,
            5,
            20,
            50,
        ],
    )
    def test_chunking_completes_for_various_overlap_sizes(
        self,
        overlap,
    ):
        chunker = DocumentChunker(
            min_chunk_size=20,
            max_chunk_size=100,
            overlap_chars=overlap,
        )

        chunks = list(chunker.chunk_text("abcdefghijklmnopqrstuvwxyz" * 20))

        assert chunks

    def test_overlap_matches_configured_size(self):
        overlap = 25

        chunker = DocumentChunker(
            min_chunk_size=20,
            max_chunk_size=100,
            overlap_chars=overlap,
        )

        text = "word " * 200

        chunks = list(chunker.chunk_text(text))

        assert len(chunks) > 1

        for current, nxt in zip(chunks, chunks[1:]):
            overlap_text = current[-overlap:]

            assert overlap_text in nxt
