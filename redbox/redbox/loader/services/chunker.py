from typing import Iterator


class TextChunker:
    def __init__(
        self,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap_chars: int = 200,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_chars = overlap_chars

    def chunk(self, text: str) -> Iterator[str]:

        if not text:
            return

        start = 0
        length = len(text)

        while start < length:
            end = min(
                start + self.max_chunk_size,
                length,
            )

            chunk = text[start:end]

            if len(chunk) >= self.min_chunk_size:
                yield chunk

            start = end - self.overlap_chars
