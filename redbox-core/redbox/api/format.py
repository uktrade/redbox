from langchain_core.documents.base import Document

from redbox.transform import combine_documents


def format_documents(documents: list[Document]) -> str:
    """
    Format a list of Document objects into a plain text string for display.
    Args:
        documents: List of Document objects.
    Returns:
        A human-readable string summarizing the document contents.
    """
    try:
        if not documents:
            return "No content available."
        formatted = []
        for doc in documents:
            if not hasattr(doc, "page_content") or not doc.page_content:
                continue
            source = doc.metadata.get("uri", "Unknown source")
            creator_type = doc.metadata.get("creator_type", "Unknown")
            page_number = doc.metadata.get("page_number", "N/A")
            # Truncate content to avoid overwhelming the output
            content_snippet = doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else "")
            formatted.append(
                f"Source: {source}\n"
                f"Type: {creator_type}\n"
                f"Page: {page_number}\n"
                f"Content: {content_snippet}"
            )
        result = "\n\n".join(formatted) or "No valid content found."
        return result
    except Exception as e:
        return f"Error formatting documents: {str(e)}"


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
