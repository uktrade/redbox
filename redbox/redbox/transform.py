import itertools
import json
import logging
import math
import os
import re
from functools import wraps
from typing import Dict, Iterable
from uuid import NAMESPACE_DNS, UUID, uuid5

import boto3
from ddtrace import tracer
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableLambda

from redbox.models.chain import DocumentMapping, DocumentState, LLMCallMetadata, RedboxState, RequestMetadata
from redbox.models.graph import RedboxEventType

log = logging.getLogger(__name__)

_BEDROCK_CONTEXT_KEY = "_redbox_bedrock_api_params"
_BEDROCK_CLIENT_PATCHED = False
_BEDROCK_DIAGNOSTICS_ENV = "REDBOX_BEDROCK_DIAGNOSTICS"


def _serialize_text(text: bytes | bytearray | str | None) -> str:
    if text is None:
        return ""
    if isinstance(text, (bytes, bytearray)):
        try:
            return text.decode("utf-8")
        except Exception:
            return text.decode("utf-8", errors="ignore")
    return str(text)


def _diagnostics_enabled() -> bool:
    return os.getenv(_BEDROCK_DIAGNOSTICS_ENV, "false").lower() in {"1", "true", "yes", "on"}


def _is_sensitive_key(key: str) -> bool:
    key = (key or "").lower()
    sensitive_terms = [
        "authorization",
        "token",
        "credential",
        "signature",
        "secret",
        "cookie",
        "x-api-key",
        "api_key",
        "passwd",
        "password",
    ]
    return any(term in key for term in sensitive_terms)


def _log_bedrock_diagnostics(*, api_params: dict, span):
    if not _diagnostics_enabled():
        return
    try:
        model_id = _bedrock_model_from_params(api_params)
        span_name = getattr(span, "name", "unknown") if span else "unknown"
        span_id = getattr(span, "span_id", "unknown") if span else "unknown"
        trace_id = getattr(span, "trace_id", "unknown") if span else "unknown"
        log.warning(
            "BEDROCK_DIAGNOSTICS modelId=%s span_name=%s span_id=%s trace_id=%s",
            model_id,
            span_name,
            span_id,
            trace_id,
        )
    except Exception as e:
        log.warning("BEDROCK_DIAGNOSTICS_ERROR %s", str(e))


def _bedrock_request_text(body: str) -> str:
    if not body:
        return ""

    try:
        payload = json.loads(body)
    except Exception:
        return body

    if isinstance(payload, dict):
        if "messages" in payload and isinstance(payload["messages"], list):
            return " ".join(
                str(message.get("content", "")) for message in payload["messages"] if isinstance(message, dict)
            )
        if "input" in payload:
            return str(payload["input"])
        if "content" in payload:
            return str(payload["content"])
    return body


def _bedrock_response_text(parsed: dict | None, http_response: object | None) -> str:
    if isinstance(parsed, dict):
        content = parsed.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                return str(first.get("text", ""))
            if isinstance(first, str):
                return first
        if isinstance(content, str):
            return content

    if http_response is not None:
        content_attr = getattr(http_response, "content", None)
        if isinstance(content_attr, (bytes, bytearray)):
            return _serialize_text(content_attr)

    return ""


def annotate_span_with_token_metrics(model: str, input_tokens: int, output_tokens: int, provider: str = "bedrock"):
    span = tracer.current_span()
    if span is None:
        return

    total_tokens = input_tokens + output_tokens

    span.set_tag("input_tokens", input_tokens)
    span.set_tag("output_tokens", output_tokens)
    span.set_tag("total_tokens", total_tokens)
    span.set_tag("model", model)
    span.set_tag("llm.model", model)
    span.set_tag("provider", provider)
    span.set_tag("llm.provider", provider)

    span.set_metric("input_tokens", input_tokens)
    span.set_metric("output_tokens", output_tokens)
    span.set_metric("total_tokens", total_tokens)

    span.set_metric("gen_ai.usage.input_tokens", input_tokens)
    span.set_metric("gen_ai.usage.output_tokens", output_tokens)
    span.set_metric("gen_ai.usage.total_tokens", total_tokens)


def _extract_text_from_content_blocks(content: list | str | None) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if text:
                parts.append(str(text))
    return " ".join(parts)


def _bedrock_request_text_from_params(api_params: dict | None) -> str:
    if not isinstance(api_params, dict):
        return ""

    if body := api_params.get("body"):
        return _bedrock_request_text(_serialize_text(body))

    if messages := api_params.get("messages"):
        return " ".join(
            _extract_text_from_content_blocks(message.get("content"))
            for message in messages
            if isinstance(message, dict)
        )

    if system := api_params.get("system"):
        return _extract_text_from_content_blocks(system)

    if input_text := api_params.get("inputText"):
        return str(input_text)

    if input_body := api_params.get("input"):
        return str(input_body)

    try:
        return json.dumps(api_params)
    except TypeError:
        return str(api_params)


def _bedrock_response_text_from_parsed(parsed: dict | None) -> str:
    if not isinstance(parsed, dict):
        return ""

    if output := parsed.get("output"):
        if isinstance(output, dict):
            if message := output.get("message"):
                if isinstance(message, dict):
                    return _extract_text_from_content_blocks(message.get("content"))
            if text := output.get("text"):
                return str(text)

    return _bedrock_response_text(parsed=parsed, http_response=None)


def _usage_value(usage: dict | None, *keys: str) -> int | None:
    if not isinstance(usage, dict):
        return None

    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _bedrock_model_from_params(api_params: dict | None) -> str:
    if not isinstance(api_params, dict):
        return "unknown"

    return str(api_params.get("modelId") or api_params.get("model_id") or "unknown")


def _capture_bedrock_request_for_metrics(model=None, params=None, context=None, **kwargs):
    if isinstance(context, dict):
        context[_BEDROCK_CONTEXT_KEY] = params if isinstance(params, dict) else {}

    if _diagnostics_enabled():
        try:
            model_id = _bedrock_model_from_params(params if isinstance(params, dict) else {})
            log.warning("BEDROCK_DIAGNOSTICS_REQUEST modelId=%s", model_id)
        except Exception as e:
            log.warning("BEDROCK_DIAGNOSTICS_REQUEST_ERROR %s", str(e))


def _annotate_bedrock_response_metrics(model=None, http_response=None, parsed=None, context=None, **kwargs):
    api_params = context.get(_BEDROCK_CONTEXT_KEY, {}) if isinstance(context, dict) else {}
    usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}

    input_tokens = _usage_value(usage, "input_tokens", "inputTokens")
    output_tokens = _usage_value(usage, "output_tokens", "outputTokens")

    if input_tokens is None:
        request_text = _bedrock_request_text_from_params(api_params)
        if request_text:
            input_tokens = bedrock_tokeniser(request_text)

    if output_tokens is None:
        response_text = _bedrock_response_text_from_parsed(parsed)
        if response_text:
            output_tokens = bedrock_tokeniser(response_text)

    if input_tokens is None or output_tokens is None:
        span = tracer.current_span()
        _log_bedrock_diagnostics(api_params=api_params, span=span)
        return

    annotate_span_with_token_metrics(
        model=_bedrock_model_from_params(api_params),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider="bedrock",
    )

    span = tracer.current_span()
    _log_bedrock_diagnostics(api_params=api_params, span=span)


def _register_bedrock_client_token_handlers(client):
    meta = getattr(client, "meta", None)
    if meta is None or getattr(meta, "service_model", None) is None:
        return client

    if meta.service_model.service_name != "bedrock-runtime":
        return client

    if getattr(meta, "_redbox_token_metrics_registered", False):
        return client

    meta.events.register("before-call.bedrock-runtime", _capture_bedrock_request_for_metrics)
    meta.events.register("after-call.bedrock-runtime", _annotate_bedrock_response_metrics)
    meta._redbox_token_metrics_registered = True
    return client


def ensure_bedrock_client_token_metrics():
    global _BEDROCK_CLIENT_PATCHED

    if _BEDROCK_CLIENT_PATCHED:
        return

    original_boto3_client = boto3.client
    original_session_client = boto3.session.Session.client

    @wraps(original_boto3_client)
    def _patched_boto3_client(*args, **kwargs):
        client = original_boto3_client(*args, **kwargs)
        return _register_bedrock_client_token_handlers(client)

    @wraps(original_session_client)
    def _patched_session_client(self, *args, **kwargs):
        client = original_session_client(self, *args, **kwargs)
        return _register_bedrock_client_token_handlers(client)

    boto3.client = _patched_boto3_client
    boto3.session.Session.client = _patched_session_client
    _BEDROCK_CLIENT_PATCHED = True


def bedrock_tokeniser_tokens(text: str) -> list[str]:
    # Simple tokeniser that counts the number of words in the text
    tokens = re.findall(r"\w+|[^\w\s]", text)

    # Check if there's a trailing space and add 1 token if needed
    if text.endswith(" "):
        tokens.append("<space>")  # Just a placeholder, not an actual token

    return tokens


def bedrock_tokeniser(text: str) -> int:
    tokens = bedrock_tokeniser_tokens(text=text)
    return len(tokens)


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    # Use the same tokenization logic as bedrock_tokeniser
    tokens = bedrock_tokeniser_tokens(text=text)
    token_count = len(tokens)

    # If it's already small enough, return unchanged
    if token_count <= max_tokens:
        return text, token_count

    # Otherwise cut to max_tokens
    truncated_tokens = tokens[:max_tokens]

    # Reconstruct text – simple join rules
    result = ""
    for t in truncated_tokens:
        if t == "<space>":
            result += " "
        elif re.match(r"\w+", t):
            # Token is a word → add space before it unless at start
            if result and not result.endswith(" "):
                result += " "
            result += t
        else:
            # punctuation — attach directly
            result += t

    return result.strip(), max_tokens


def join_result_with_token_limit(result: list, max_tokens: int, log_stub: str) -> str:
    result_content = []
    current_token_counts = 0

    for res in result:
        content = res if isinstance(res, str) else res.content
        token_count = bedrock_tokeniser(content)
        log.warning(f"{log_stub} Tool response token count: {token_count}")

        # If adding this whole piece still fits, append normally
        if current_token_counts + token_count <= max_tokens:
            result_content.append(content)
            current_token_counts += token_count
        else:
            # If no room, add only what fits
            remaining_tokens = max_tokens - current_token_counts
            if remaining_tokens > 0:
                log.warning(f"{log_stub} Truncating tool output to fit remaining token budget ({remaining_tokens}).")
                truncated, truncated_token_count = truncate_to_tokens(content, remaining_tokens)
                result_content.append(truncated)
                current_token_counts += truncated_token_count
            else:
                log.warning(f"{log_stub} No remaining token budget ({max_tokens}). Skipping.")
            break  # Max reached — stop processing further results
    return " ".join(result_content)


# This should be unnecessary and indicates we're not chunking correctly
def combine_documents(a: Document, b: Document):
    def listify(metadata: dict, field_name: str) -> list:
        field_value = metadata.get(field_name)
        if isinstance(field_value, list):
            return field_value
        if field_value is None:
            return []
        return [field_value]

    def sorted_list_or_none(obj: list):
        return sorted(set(obj)) or None

    def combine_values(field_name: str):
        return sorted_list_or_none(listify(a.metadata, field_name) + listify(b.metadata, field_name))

    combined_content = a.page_content + b.page_content
    combined_metadata = a.metadata.copy()
    combined_metadata["token_count"] = a.metadata["token_count"] + b.metadata["token_count"]
    combined_metadata["page_number"] = combine_values("page_number")
    combined_metadata["languages"] = combine_values("languages")
    combined_metadata["link_texts"] = combine_values("link_texts")
    combined_metadata["link_urls"] = combine_values("link_urls")
    combined_metadata["links"] = combine_values("links")

    return Document(page_content=combined_content, metadata=combined_metadata)


def to_document_mapping(docs: list[Document]) -> DocumentMapping:
    return {doc.metadata["uuid"]: doc for doc in docs}


def group_documents(docs: Iterable[Document]) -> dict[str, list[Document]]:
    def get_uri(d):
        return d.metadata["uri"]

    grouped_docs = itertools.groupby(sorted(docs, key=get_uri), key=get_uri)
    return {key: list(values) for key, values in grouped_docs}


def structure_documents_by_file_name(docs: list[Document]) -> DocumentState:
    """Structures a list of documents by a group_uuid and document_uuid.

    The group_uuid is generated deterministically based on the file_name.

    The document_uuid is taken from the Document metadata directly.
    """
    result = DocumentState()

    grouped_docs = group_documents(docs)

    result.groups = {uuid5(NAMESPACE_DNS, uri): to_document_mapping(d) for uri, d in grouped_docs.items()}

    return result


def create_group_uuid(file_name: str, indices: list[int]) -> UUID:
    """Uses a file name and list of indices to generate a deterministic UUID."""
    unique_str = file_name + "-" + ",".join(map(str, sorted(indices)))
    return uuid5(NAMESPACE_DNS, unique_str)


def create_group_uuid_for_group(documents: list[Document]) -> UUID:
    """create a uuid for a DocumentGroup"""
    if not documents:
        raise ValueError("at least one document is required")

    file_name = documents[0].metadata["uri"]
    group_indices = [d.metadata["index"] for d in documents]
    return create_group_uuid(file_name, group_indices)


def documents_are_consecutive(first: Document, second: Document) -> bool:
    """are the two documents consecutive, i.e. do they appear next to each other in the original text?"""
    if first.metadata["uri"] is None:
        return True

    if first.metadata["uri"] != second.metadata["uri"]:
        return False

    return abs(first.metadata["index"] - second.metadata["index"]) <= 1


def group_and_sort_documents(group: list[Document]) -> list[list[Document]]:
    """Breaks a group into blocks of ordered consecutive indices.

    The group is intended to be a single file_name.
    """
    if not group:
        return []

    # Process consecutive blocks and sort them by index
    consecutive_blocks = []
    temp_block = [group[0]]

    for doc in group[1:]:
        if documents_are_consecutive(temp_block[-1], doc):
            temp_block.append(doc)
        else:
            # Append the current block
            consecutive_blocks.append(temp_block)
            temp_block = [doc]

    # Append the last block
    consecutive_blocks.append(temp_block)

    # Sort each block by index
    sorted_blocks = [sorted(block, key=lambda d: d.metadata["index"]) for block in consecutive_blocks]

    return sorted_blocks


def structure_documents_by_group_and_indices(docs: list[Document]) -> DocumentState:
    """Structures a list of documents by blocks of consecutive indices in group_uuids.

    Assumes a sorted list was passed where blocks of group_uuids with consecutive
    indices are already together, as per redbox.transform.sort_documents().

    The group_uuid is generated deterministically based on the file_name and group indices.

    The document_uuid is taken from the Document metadata directly.
    """
    result = DocumentState()

    groups = group_and_sort_documents(docs)

    result.groups = {
        create_group_uuid_for_group(group): {doc.metadata["uuid"]: doc for doc in group} for group in groups
    }

    return result


def flatten_document_state(documents: DocumentState | None) -> list[Document]:
    """Flattens a DocumentState into a list of Documents."""
    if not documents:
        return []
    return [document for group in documents.groups.values() for document in group.values()]


def get_document_token_count(state: RedboxState) -> int:
    """Calculates the total token count of all documents in a state."""
    return sum(d.metadata["token_count"] for d in flatten_document_state(state.documents))


def to_request_metadata(obj: dict) -> RequestMetadata:
    """Takes a dictionary with keys 'prompt', 'response' and 'model' and creates metadata.

    Will also emit events for metadata updates.
    """

    prompt = obj["prompt"]
    response = obj["text_and_tools"]["raw_response"].content
    model = obj["model"]

    tokeniser = bedrock_tokeniser
    input_tokens = tokeniser(prompt)
    try:
        output_tokens = tokeniser(response)
    except Exception:
        output_tokens = len(response[0].get("text", []))

    annotate_span_with_token_metrics(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider="bedrock",
    )

    metadata_event = RequestMetadata(
        llm_calls=[
            LLMCallMetadata(
                llm_model_name=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        ]
    )

    dispatch_custom_event(RedboxEventType.on_metadata_generation.value, metadata_event)

    return metadata_event


@RunnableLambda
def get_all_metadata(obj: dict):
    text_and_tools = obj["text_and_tools"]

    if parsed_response := text_and_tools.get("parsed_response"):
        try:
            text = getattr(parsed_response, "answer", parsed_response.model_dump_json())
        except Exception:
            text = getattr(parsed_response, "answer", parsed_response)
        citations = getattr(parsed_response, "citations", [])
    else:
        text = text_and_tools["raw_response"].content
        citations = []

    out = {
        "messages": [AIMessage(content=text, tool_calls=text_and_tools["raw_response"].tool_calls)],
        "metadata": to_request_metadata(obj),
        "citations": citations,
        "final_chain": obj["final_chain"],
    }
    return out


def merge_documents(initial: list[Document], adjacent: list[Document]) -> list[Document]:
    """Merges a list of adjacent documents with an initial list.

    Privileges the initial score.
    """
    # Keep initial scores
    merged_dict = to_document_mapping(adjacent) | to_document_mapping(initial)

    return sorted(list(merged_dict.values()), key=lambda d: -d.metadata["score"])[: len(initial)]


def sort_documents(documents: list[Document]) -> list[Document]:
    """Sorts a list of documents so chunks are both consecutive and ordered by score.

    More explicitly:

    * Blocks of documents from the same file with consecutive indices are presented together, in order of ascending index
    * Blocks of documents are presented in order of their highest-scoring member

    For example, in this list of (score, file, index):

    5, foo.txt, 3
    4.9, foo.txt, 2
    4.8, bar.txt, 9
    4.1, foo.txt, 1
    3.8, foo.txt, 24

    We will get:

    4.1, foo.txt, 1
    4.9, foo.txt, 2
    5, foo.txt, 3
    4.8, bar.txt, 9
    3.8, foo.txt, 24
    """

    def max_score(group: list[Document]) -> float:
        """Returns the maximum score in a group of documents."""
        return max(d.metadata["score"] for d in group)

    # Group and sort docs by file_name and handle consecutive indices
    grouped_by_file = group_documents(documents)

    # Step 1: group & sort each group
    document_blocks = [group_and_sort_documents(docs) for docs in grouped_by_file.values()]

    # Step 2: flatten blocks into a single list of docs
    all_sorted_blocks = itertools.chain(*document_blocks)

    # Step 3: Sort the blocks by the maximum score within each block
    all_sorted_blocks_by_max_score = sorted(all_sorted_blocks, key=max_score, reverse=True)

    # Step 4: Flatten the list of blocks back into a single list
    return list(itertools.chain.from_iterable(all_sorted_blocks_by_max_score))


TRUNCATION_MARKER = " ...[truncated]"


def combine_agents_state(agents_results: Dict[str, AnyMessage], max_tokens: int) -> dict:
    """
    Combine a list of agent results into a string.
    OR if it is over the max amount, truncating proportially
    A truncation marker is appended to any block, so the downstream LLM knows that the content is incomplete
    """
    if not agents_results:
        return {}
    sizes = {aid: bedrock_tokeniser(msg.content) for aid, msg in agents_results.items()}
    total_tokens = sum(sizes.values())

    if max_tokens <= 0:
        log.warning("combine_agents_state: non-positive token budget, returning empty result!")
        return {}

    if total_tokens <= max_tokens:
        flatten_agent_results = "\n\n".join([msg.content for msg in agents_results.values()])
        return {"all_result": flatten_agent_results}

    marker_cost = bedrock_tokeniser(TRUNCATION_MARKER)
    parts: list[str] = []
    for agent_id, msg in agents_results.items():
        original_tokens = sizes[agent_id]
        share = max(1, math.floor(max_tokens * original_tokens / total_tokens))
        if original_tokens <= share:
            parts.append(msg.content)
        else:
            budget = max(1, share - marker_cost)
            truncated, _ = truncate_to_tokens(msg.content, budget)
            parts.append(truncated + TRUNCATION_MARKER)
            log.warning(
                "combine_agents_state: truncated agent %s from %d to ~%d tokens.",
                agent_id,
                original_tokens,
                share,
            )
    joined = "\n\n".join(parts)
    if bedrock_tokeniser(joined) > max_tokens:
        joined, _ = truncate_to_tokens(joined, max_tokens)
    return {"all_result": joined}
