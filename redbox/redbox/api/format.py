import json
import logging
from typing import Any

from langchain_core.documents.base import Document

from redbox.models.file import ChunkCreatorType
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


def find_first_link_field(data, _depth: int = 0, max_depth: int = 0) -> str | None:
    """Recursively find the first "url" field in a nested structure."""
    if _depth > max_depth:
        return None

    if isinstance(data, dict):
        # Check current level first
        if "url" in data.keys() and data.get("url", None) is not None:
            return str(data.get("url"))
        # Then recurse into values
        for value in data.values():
            result = find_first_link_field(value, _depth + 1, max_depth)
            if result:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_first_link_field(item, _depth + 1, max_depth)
            if result:
                return result

    return None


# def extract_links(data: dict | None) -> list[tuple[str, Any]]:
#     """
#     Handles two shapes:
#       - Single object:  { ...fields... }
#       - Paged result:   { "total": N, "<key>": [ {...}, {...} ] }

#     Returns all found url values.
#     """
#     if data is None:
#         return []

#     # Detect paged result: has "total" field + at least one list field
#     if "total" in data:
#         for key, value in data.items():
#             if key == "total":
#                 continue
#             if isinstance(value, list):
#                 # Extract first _link from each item in the list
#                 return [(link, item) for item in value if (link := find_first_link_field(item))]

#     # Single object — extract first _link from root
#     link = find_first_link_field(data)
#     return [(link, data)] if link else []


def _find_paged_list(data: dict) -> list | None:
    """
    Returns the list field from a paged result, or None if not a paged shape.
    A paged result must have 'total' (int) + exactly one list field (ignoring 'total').
    """
    if "total" not in data or not isinstance(data["total"], int):
        return None

    list_fields = [v for k, v in data.items() if k != "total" and isinstance(v, list)]
    return list_fields[0] if len(list_fields) == 1 else None


def extract_links(data: dict | None) -> list[tuple[str, Any]]:
    """
    Handles two shapes:
      - Paged result:  { "total": N, "<key>": [ {...}, {...} ] }
      - Single object: { ...fields... }

    Returns list of (url, object) tuples.
    """
    if data is None:
        return []

    paged_list = _find_paged_list(data)
    if paged_list is not None:
        return [(link, item) for item in paged_list if (link := find_first_link_field(item))]

    link = find_first_link_field(data)
    return [(link, data)] if link else []


def format_mcp_tool_response(tool_response, creator_type: ChunkCreatorType) -> str:
    data = json.loads(tool_response)
    result_type = data.get("result_type")
    result = data.get("result")

    if result_type is None or result is None:
        return tool_response if isinstance(tool_response, str) else str(tool_response)

    deep_links = []
    match result_type:
        case "nullable":
            deep_links = [(result.get("url"), result)]
        case "paged":
            deep_links = [(p.get("url"), p) for p in result.get("items", [])]
        case "multipaged":
            paged_results = [v for v in result.values() if v is not None]
            for page_result in paged_results:
                deep_links += [(p.get("url"), p) for p in page_result.get("result", {}).get("items", [])]
        case "composite":
            parent, paged_data = result
            deep_links = [(parent.get("url"), parent)]
            paged_results = [v for v in paged_data.values() if v is not None]
            for page in paged_results:
                deep_links += [(item.get("url"), item) for item in page.get("result", {}).get("items", [])]

    citations = [
        Document(
            page_content=json.dumps(item),
            metadata={"creator_type": creator_type, "uri": link or "", "page_number": ""},
        )
        for link, item in deep_links
    ]
    return format_documents(documents=citations)
