import pytest

from redbox.loader.parsers.markdown import _MarkdownLayoutParser


class TestCleanMarkdown:
    @pytest.mark.parametrize(
        ("input_text", "expected"),
        [
            (
                "**bold** text",
                "bold text",
            ),
            (
                "_italic_ text",
                "italic text",
            ),
            (
                "`code` example",
                "code example",
            ),
            (
                "~strike~ text",
                "strike text",
            ),
            (
                "[OpenAI](https://openai.com)",
                "OpenAI",
            ),
            (
                "- item one",
                "item one",
            ),
            (
                "* item two",
                "item two",
            ),
            (
                "+ item three",
                "item three",
            ),
            (
                "> quoted text",
                "quoted text",
            ),
        ],
    )
    def test_removes_markdown_syntax(
        self,
        input_text,
        expected,
    ):
        parser = _MarkdownLayoutParser()

        assert parser._clean_markdown(input_text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "\n\n",
        ],
    )
    def test_handles_empty_input(
        self,
        text,
    ):
        parser = _MarkdownLayoutParser()

        assert parser._clean_markdown(text) == ""

    def test_preserves_plain_text(self):
        parser = _MarkdownLayoutParser()

        text = "This is normal text."

        assert parser._clean_markdown(text) == text


class TestParseHeadings:
    @pytest.mark.parametrize(
        ("markdown", "expected"),
        [
            ("# Heading", "Heading"),
            ("## Heading", "Heading"),
            ("### Heading", "Heading"),
            ("#### Heading", "Heading"),
            ("##### Heading", "Heading"),
            ("###### Heading", "Heading"),
        ],
    )
    def test_creates_title_blocks(
        self,
        markdown,
        expected,
    ):
        parser = _MarkdownLayoutParser()

        blocks = parser.parse(markdown)

        assert len(blocks) == 1

        block = blocks[0]

        assert block.text == expected
        assert block.is_title is True
        assert block.block_type == "LAYOUT_TITLE"

    def test_strips_heading_whitespace(self):
        parser = _MarkdownLayoutParser()

        blocks = parser.parse("#     Heading     ")

        assert blocks[0].text == "Heading"


class TestParseParagraphs:
    @pytest.mark.parametrize(
        "markdown",
        [
            "Simple paragraph",
            "Paragraph with **bold** text",
            "Paragraph with [link](https://example.com)",
            "> quoted paragraph",
            "- list item",
        ],
    )
    def test_creates_text_blocks(
        self,
        markdown,
    ):
        parser = _MarkdownLayoutParser()

        blocks = parser.parse(markdown)

        assert len(blocks) == 1

        block = blocks[0]

        assert block.is_title is False
        assert block.block_type == "LAYOUT_TEXT"

    def test_paragraph_text_is_cleaned(self):
        parser = _MarkdownLayoutParser()

        blocks = parser.parse("Paragraph with **bold** and [link](https://example.com)")

        assert blocks[0].text == "Paragraph with bold and link"


class TestParseBlocks:
    def test_empty_document_returns_no_blocks(self):
        parser = _MarkdownLayoutParser()

        blocks = parser.parse("")

        assert blocks == []

    def test_multiple_paragraphs_create_multiple_blocks(self):
        parser = _MarkdownLayoutParser()

        blocks = parser.parse(
            """
            First paragraph.

            Second paragraph.

            Third paragraph.
            """
        )

        assert len(blocks) == 1

    def test_heading_and_paragraphs_create_separate_blocks(self):
        parser = _MarkdownLayoutParser()

        blocks = parser.parse(
            """
            # Introduction

            Intro text

            ## Background

            Background text
            """
        )

        assert [(b.is_title, b.text) for b in blocks] == [
            (True, "Introduction"),
            (False, "Intro text"),
            (True, "Background"),
            (False, "Background text"),
        ]

    def test_all_blocks_have_page_number_one(self):
        parser = _MarkdownLayoutParser()

        blocks = parser.parse(
            """
            # Title

            Paragraph

            Another paragraph
            """
        )

        assert all(block.page_number == 1 for block in blocks)


class TestEndToEnd:
    def test_realistic_markdown_document(self):
        parser = _MarkdownLayoutParser()

        blocks = parser.parse(
            """
            # Executive Summary

            This document explains the architecture.

            ## Background

            Background information with **important** details.

            ## References

            - First item
            - Second item

            > Additional notes

            See [documentation](https://example.com).
            """
        )

        assert [(b.is_title, b.text) for b in blocks] == [
            (True, "Executive Summary"),
            (False, "This document explains the architecture."),
            (True, "Background"),
            (False, "Background information with important details."),
            (True, "References"),
            (
                False,
                "First item\nSecond item\n\nAdditional notes\n\nSee documentation.",
            ),
        ]

    def test_parse_is_idempotent_for_new_instance(self):
        markdown = """
        # Title

        Paragraph
        """

        first = _MarkdownLayoutParser().parse(markdown)
        second = _MarkdownLayoutParser().parse(markdown)

        assert first == second


class TestParserState:
    def test_parse_does_not_accumulate_blocks_between_calls(self):
        parser = _MarkdownLayoutParser()

        parser.parse("# First")

        blocks = parser.parse("# Second")

        assert len(blocks) == 1
        assert blocks[0].is_title is True
        assert blocks[0].text == "Second"
