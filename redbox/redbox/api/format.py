import logging
import re
from typing import Any

from langchain_core.documents.base import Document

from redbox.transform import combine_documents

log = logging.getLogger(__name__)


def format_documents(documents: list[Document]) -> str:
    log.warning("[format_documents] Received %s documents for formatting", len(documents))
    formatted: list[str] = []
    for d in documents:
        doc_xml = (
            f"<Document>\n"
            f"\t<SourceType>{d.metadata.get('creator_type', 'Unknown')}</SourceType>\n"
            f"\t<Source>{d.metadata.get('uri', '')}</Source>\n"
            f"\t<page_number>{d.metadata.get('page_number', '')}</page_number>\n"
            "\t<Content>\n"
            f"{d.page_content}\n"
            "\t</Content>\n"
            f"</Document>"
        )
        formatted.append(doc_xml)

    return "\n\n".join(formatted)


def reduce_chunks_by_tokens(chunks: list[Document] | None, chunk: Document, max_tokens: int) -> list[Document]:
    if not chunks:
        return [chunk]

    last_chunk = chunks[-1]

    chunk_tokens = chunk.metadata["token_count"]
    last_chunk_tokens = last_chunk.metadata["token_count"]
    if chunk_tokens + last_chunk_tokens <= max_tokens:
        chunks[-1] = combine_documents(last_chunk, chunk)
    else:
        chunks.append(chunk)
    return chunks


def find_first_link_field(data) -> str | None:
    """Recursively find the first "url" field in a nested structure."""

    if isinstance(data, dict):
        # Check current level first
        if "url" in data.keys() and data.get("url") is not None:
            return str(data.get("url"))
        # Then recurse into values
        for value in data.values():
            result = find_first_link_field(value)
            if result:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_first_link_field(item)
            if result:
                return result

    return None


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def extract_links(data: dict | None) -> list[tuple[str, Any]]:
    """
    Handles two shapes:
      - Single object:  { ...fields... }
      - Paged result:   { "total": N, "<key>": [ {...}, {...} ] }

    Returns all found url values.
    """
    if data is None:
        return []

    # Detect paged result: has "total" field + at least one list field
    if "total" in data:
        for key, value in data.items():
            if key == "total":
                continue
            if isinstance(value, list):
                # Extract first _link from each item in the list
                return [(link, item) for item in value if (link := find_first_link_field(item))]

    # Single object — extract first _link from root
    link = find_first_link_field(data)
    return [(link, data)] if link else []
