from dataclasses import dataclass
from typing import Iterator, Callable


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
    All size comparisons are in tokens, not characters.
    """

    def __init__(
        self,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap_chars: int = 200,
        tokeniser: Callable[[str], int] = len,  # fallback to char count if not provided
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_chars = overlap_chars
        self.tokeniser = tokeniser

    def _token_count(self, text: str) -> int:
        return self.tokeniser(text)

    def chunk(self, blocks: list[LayoutBlock]) -> Iterator[Chunk]:
        sections = self.build_sections(blocks=blocks)
        yield from self.build_chunks(sections=sections)

    def build_sections(self, blocks: list[LayoutBlock]) -> list[Section]:
        if not blocks:
            return []

        sections: list[Section] = []
        current_title = ""
        current_blocks: list[LayoutBlock] = []

        for block in blocks:
            if block.is_title:
                if current_blocks:
                    sections.append(Section(title=current_title, blocks=current_blocks))
                current_title = block.text
                current_blocks = []  # fix: don't include title block in body

            else:
                current_blocks.append(block)

        if current_blocks:
            sections.append(Section(title=current_title, blocks=current_blocks))

        return sections

    def build_chunks(self, sections: list[Section]) -> Iterator[Chunk]:
        for section in sections:
            if not section.blocks:
                continue

            body = "\n\n".join(block.text for block in section.blocks if block.text.strip())

            if not body:
                continue

            start_page = section.blocks[0].page_number
            end_page = section.blocks[-1].page_number

            title_prefix = f"{section.title}\n\n" if section.title else ""
            title_tokens = self._token_count(title_prefix)

            available_body_tokens = max(
                self.min_chunk_size,
                self.max_chunk_size - title_tokens,  # token-aware budget
            )

            # Section fits into one chunk
            if self._token_count(body) <= available_body_tokens:
                yield Chunk(
                    text=f"{title_prefix}{body}",
                    section_title=section.title,
                    page_start=start_page,
                    page_end=end_page,
                )
                continue

            # Large section -> split body while repeating title
            for chunk_body in self.chunk_text(body, max_token_size=available_body_tokens):
                yield Chunk(
                    text=f"{title_prefix}{chunk_body}",
                    section_title=section.title,
                    page_start=start_page,
                    page_end=end_page,
                )

    def chunk_text(
        self,
        text: str,
        max_token_size: int | None = None,
    ) -> Iterator[str]:
        """
        Split text respecting token budget.
        Prefer: paragraph -> sentence -> line -> word -> hard split.
        """
        if not text:
            return

        max_token_size = max_token_size or self.max_chunk_size

        start = 0
        length = len(text)

        while start < length:
            remaining = text[start:]
            remaining_tokens = self._token_count(remaining)

            # Everything remaining fits
            if remaining_tokens <= max_token_size:
                chunk = remaining.strip()
                if chunk:
                    yield chunk
                break

            # Binary search for the char index where token count ~ max_token_size.
            # Tokens are sub-linear in chars so this converges quickly.
            lo, hi = 0, len(remaining)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self._token_count(remaining[:mid]) <= max_token_size:
                    lo = mid
                else:
                    hi = mid - 1

            candidate = remaining[:lo]

            # lookahead: if what's left after this split is too small, absorb it
            leftover = remaining[lo:]
            if self._token_count(leftover) < self.min_chunk_size:
                chunk = remaining.strip()
                if chunk:
                    yield chunk
                break

            split_at = self._find_split_point(candidate)
            chunk = candidate[:split_at].strip()

            if chunk:
                yield chunk

            next_start = start + split_at
            start = max(next_start - self.overlap_chars, start + 1)

    def _find_split_point(self, text: str) -> int:
        """
        Find best semantic split within a char-bounded candidate.
        Min position guard uses char length as a proxy — the candidate
        is already token-bounded so this is just for semantic quality.
        """
        if not text:
            return 0

        # Use char-based min_pos as a rough guard to avoid tiny leading chunks.
        # Exact token enforcement is handled by the caller.
        min_pos = len(text) // 2  # search only in the second half

        idx = text.rfind("\n\n", min_pos)
        if idx != -1:
            return idx

        for delimiter in (". ", "! ", "? "):
            idx = text.rfind(delimiter, min_pos)
            if idx != -1:
                return idx + len(delimiter)

        idx = text.rfind("\n", min_pos)
        if idx != -1:
            return idx

        idx = text.rfind(" ", min_pos)
        if idx != -1:
            return idx

        return len(text)
