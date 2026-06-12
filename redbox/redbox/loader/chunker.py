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
    Converts:

        LayoutBlock[]
            ↓
         Section[]
            ↓
          Chunk[]

    regardless of whether blocks came from:
      - Textract Layout
      - Unstructured DOCX
      - Unstructured PPTX
      - Unstructured HTML
      - Unstructured Markdown
      - etc.
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
        sections = self.build_sections(blocks)
        yield from self.chunk_sections(sections)

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

        current_title = "Document Start"
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

    def chunk_sections(
        self,
        sections: list[Section],
    ) -> Iterator[Chunk]:

        for section in sections:
            if not section.blocks:
                continue

            content_blocks = [block.text for block in section.blocks if not block.is_title]

            body = "\n\n".join(content_blocks)

            if not body.strip():
                continue

            section_text = f"{section.title}\n\n{body}"

            start_page = section.blocks[0].page_number
            end_page = section.blocks[-1].page_number

            if len(section_text) <= self.max_chunk_size:
                yield Chunk(
                    text=section_text,
                    section_title=section.title,
                    page_start=start_page,
                    page_end=end_page,
                )

                continue

            for chunk_text in self.chunk_text(section_text):
                yield Chunk(
                    text=chunk_text,
                    section_title=section.title,
                    page_start=start_page,
                    page_end=end_page,
                )

    def chunk_text(
        self,
        text: str,
    ) -> list[str]:
        """
        Character chunking with overlap.

        Used only when a section exceeds max_chunk_size.
        """
        if not text:
            return []

        chunks: list[str] = []

        start = 0
        length = len(text)

        while start < length:
            end = min(
                start + self.max_chunk_size,
                length,
            )
            chunk = text[start:end]

            if len(chunk) >= self.min_chunk_size or not chunks:
                chunks.append(chunk)

            if end >= length:
                break

            start = max(
                end - self.overlap_chars,
                start + 1,
            )

        return chunks
