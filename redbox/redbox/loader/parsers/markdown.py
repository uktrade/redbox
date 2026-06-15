import re

from redbox.loader.chunker import LayoutBlock


_MARKDOWN_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s+(.*)$",
    re.MULTILINE,
)


class _MarkdownLayoutParser:
    """
    Lightweight Markdown parser that emits LayoutBlocks.

    Supported:
      - ATX headings (# ## ### ...)
      - paragraphs
      - lists
      - blockquotes
      - inline markdown cleanup

    Not intended to be a full markdown renderer.
    """

    def __init__(self) -> None:
        self._page = 1

    def parse(
        self,
        markdown: str,
    ) -> list[LayoutBlock]:

        blocks: list[LayoutBlock] = []

        paragraphs = re.split(
            r"\n{2,}",
            markdown.strip(),
        )

        current_content: list[str] = []

        def flush_content() -> None:
            if not current_content:
                return

            text = "\n\n".join(current_content).strip()

            if text:
                blocks.append(
                    LayoutBlock(
                        text=text,
                        block_type="LAYOUT_TEXT",
                        page_number=self._page,
                        is_title=False,
                    )
                )

            current_content.clear()

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            heading_match = _MARKDOWN_HEADING_RE.match(paragraph)

            if heading_match:
                flush_content()

                blocks.append(
                    LayoutBlock(
                        text=heading_match.group(1).strip(),
                        block_type="LAYOUT_TITLE",
                        page_number=self._page,
                        is_title=True,
                    )
                )
                continue

            text = self._clean_markdown(paragraph)

            if text:
                current_content.append(text)

        flush_content()

        return blocks

    def _clean_markdown(
        self,
        text: str,
    ) -> str:
        """
        Remove common markdown syntax while preserving content.
        """

        # [text](url) -> text
        text = re.sub(
            r"\[([^\]]+)\]\([^)]+\)",
            r"\1",
            text,
        )

        # emphasis/code markers
        text = re.sub(
            r"[*_`~]",
            "",
            text,
        )

        # list markers
        text = re.sub(
            r"^\s*[-*+]\s+",
            "",
            text,
            flags=re.MULTILINE,
        )

        # blockquotes
        text = re.sub(
            r"^\s*>\s+",
            "",
            text,
            flags=re.MULTILINE,
        )

        return text.strip()
