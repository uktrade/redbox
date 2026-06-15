from dataclasses import dataclass
from typing import Iterator


@dataclass
class LayoutBlock:
    text: str
    block_type: str
    page_number: int
    is_title: bool


@dataclass
class Section:
    title: str
    blocks: list[LayoutBlock]


@dataclass
class Chunk:
    text: str
    section_title: str
    page_start: int
    page_end: int


class DocumentChunker:
    """
    Converts: LayoutBlock[] -> Section[] -> Chunk[]
    """

    def __init__(
        self,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap_chars: int = 200,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_chars = overlap_chars

    def chunk(
        self,
        blocks: list[LayoutBlock],
    ) -> Iterator[Chunk]:
        """
        LayoutBlocks -> Sections -> Chunks
        """
        sections = self.build_sections(blocks=blocks)
        yield from self.build_chunks(sections=sections)

    def build_sections(
        self,
        blocks: list[LayoutBlock],
    ) -> list[Section]:
        """
        Group blocks under their nearest title.

        Example:

            Title A
            paragraph
            paragraph

            Title B
            paragraph

        becomes:

            Section("Title A")
            Section("Title B")
        """
        if not blocks:
            return []

        sections: list[Section] = []

        current_title = ""
        current_blocks: list[LayoutBlock] = []

        for block in blocks:
            if block.is_title:
                if current_blocks:
                    sections.append(
                        Section(
                            title=current_title,
                            blocks=current_blocks,
                        )
                    )

                current_title = block.text
                current_blocks = [block]

            else:
                current_blocks.append(block)

        if current_blocks:
            sections.append(
                Section(
                    title=current_title,
                    blocks=current_blocks,
                )
            )

        return sections

    def build_chunks(
        self,
        sections: list[Section],
    ) -> Iterator[Chunk]:

        for section in sections:
            if not section.blocks:
                continue

            body = "\n\n".join(block.text for block in section.blocks if block.text.strip())

            if not body:
                continue

            start_page = section.blocks[0].page_number
            end_page = section.blocks[-1].page_number

            title_prefix = f"{section.title}\n\n" if section.title else ""

            available_body_size = max(
                self.min_chunk_size,
                self.max_chunk_size - len(title_prefix),
            )

            # Section fits into one chunk
            if len(body) <= available_body_size:
                yield Chunk(
                    text=f"{title_prefix}{body}",
                    section_title=section.title,
                    page_start=start_page,
                    page_end=end_page,
                )

                continue

            # Large section -> split body while repeating title
            for chunk_body in self.chunk_text(
                body,
                max_size=available_body_size,
            ):
                yield Chunk(
                    text=f"{title_prefix}{chunk_body}",
                    section_title=section.title,
                    page_start=start_page,
                    page_end=end_page,
                )

    def chunk_text(
        self,
        text: str,
        max_size: int | None = None,
    ) -> Iterator[str]:
        """
        Prefer: paragraph boundary -> sentence boundary -> line boundary -> word boundary -> hard split

        while preserving overlap.
        """

        if not text:
            return

        max_size = max_size or self.max_chunk_size

        start = 0
        length = len(text)

        while start < length:
            target_end = min(
                start + max_size,
                length,
            )

            # final chunk
            if target_end >= length:
                chunk = text[start:].strip()

                if chunk:
                    yield chunk

                break

            candidate = text[start:target_end]

            split_at = self._find_split_point(candidate)

            chunk = candidate[:split_at].strip()

            if chunk:
                yield chunk

            next_start = start + split_at

            # guarantee forward progress
            start = max(
                next_start - self.overlap_chars,
                start + 1,
            )

    def _find_split_point(
        self,
        text: str,
    ) -> int:
        """
        Find best semantic split location.

        Search only after min_chunk_size so we don't
        create tiny chunks.
        """

        if len(text) <= self.min_chunk_size:
            return len(text)

        min_pos = self.min_chunk_size

        # Paragraph break
        idx = text.rfind("\n\n", min_pos)

        if idx != -1:
            return idx

        # Sentence break
        for delimiter in (". ", "! ", "? "):
            idx = text.rfind(delimiter, min_pos)

            if idx != -1:
                return idx + len(delimiter)

        # Single newline
        idx = text.rfind("\n", min_pos)

        if idx != -1:
            return idx

        # Whitespace
        idx = text.rfind(" ", min_pos)

        if idx != -1:
            return idx

        # Hard split fallback
        return len(text)
