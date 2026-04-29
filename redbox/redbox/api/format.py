import json
import logging
from pydantic import BaseModel, ValidationError
from typing import Optional

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


class MCPResponseMetadata(BaseModel):
    class UserFeedback(BaseModel):
        required: bool = False
        reason: Optional[str] = None

    user_feedback: UserFeedback = UserFeedback()


def format_mcp_tool_response(tool_response, creator_type: ChunkCreatorType) -> tuple[str, MCPResponseMetadata]:
    data = json.loads(tool_response)
    result_type = data.get("result_type")
    result = data.get("result")

    try:
        metadata = MCPResponseMetadata.model_validate(data.get("metadata", {}))
    except ValidationError:
        metadata = MCPResponseMetadata()

    if result_type is None or result is None:
        return (tool_response if isinstance(tool_response, str) else str(tool_response), metadata)

    deep_links = []
    match result_type:
        case "nullable":
            if isinstance(result, str):
                return result, metadata

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

    response = []
    for link, item in deep_links:
        if link:
            response.append(
                format_documents(
                    [
                        Document(
                            page_content=json.dumps(item),
                            metadata={"creator_type": creator_type, "uri": link, "page_number": ""},
                        )
                    ]
                )
            )
        else:
            response.append(json.dumps(item))

    if not response:
        return ("No results found.", metadata)

    return ("\n\n".join(response), metadata)
