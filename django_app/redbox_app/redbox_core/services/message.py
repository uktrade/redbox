import logging
import re
from collections.abc import Sequence

import markdown
from django.template.loader import render_to_string

from redbox_app.redbox_core.models import ChatMessage, Citation, File
from redbox_app.redbox_core.types import STREAM_REF_RE, CitationMap

logger = logging.getLogger(__name__)


def render_citation(citation: Citation, footnote_counter: int) -> str:
    return render_to_string(
        "chat/message/citations/popover.html",
        {
            "citation": citation,
            "footnote_counter": footnote_counter,
        },
    )


def render_citation_placeholder(footnote_counter: int) -> str:
    return render_to_string(
        "chat/message/citations/popover-placeholder.html",
        {
            "footnote_counter": footnote_counter,
        },
    )


def render_resources(message: ChatMessage) -> str:
    return render_to_string(
        "chat/message/citations/resources.html",
        {
            "resources": message.resources,
            "message_id": message.id,
        },
    )


def render_message(message: ChatMessage) -> str:
    return render_to_string(
        "chat/message/message-box.html",
        {
            "message": message,
        },
    )


def render_message_content(message: ChatMessage, text: str | None = None) -> str:
    return render_to_string(
        "chat/message/message-content.html",
        {
            "role": message.role,
            "text": text or message.text,
            "message_id": message.id,
            "selected_files": message.unique_selected_files(),
            "resources": message.resources,
            "route": message.route,
        },
    )


def streaming_replace_refs(text: str, citation_map: CitationMap) -> str:
    def repl(match):
        ref_num = match.group(0) or match.group(1) or match.group(2)
        key = f"ref_{ref_num}"
        num = citation_map.resolve(key)

        return render_citation_placeholder(footnote_counter=num)

    return STREAM_REF_RE.sub(repl, str(text))


def replace_ref(message_text: str, citation: Citation, footnote_counter: int) -> str:
    citation_name = citation.citation_name
    pattern = rf"[\[\(\{{<]{citation_name}[\]\)\}}>]|\b{citation_name}\b"
    rendered_citation = render_citation(citation, footnote_counter)

    message_text = re.sub(
        pattern,
        rendered_citation,
        message_text,
        # count=1,
    )
    return re.sub(pattern, "", message_text)


def replace_text_in_answer(message_text: str, citation: Citation, footnote_counter: int) -> str:
    return message_text.replace(
        citation.text_in_answer,
        f"{citation.text_in_answer}{render_citation(citation, footnote_counter)}",
    )


def remove_dangling_citation(message_text: str) -> str:
    pattern = r"[\[\(\{<]ref_\d+[\]\)\}>]|\bref_\d+\b"  # Hallucinated citations
    empty_pattern = r"[\[\(\{<]\s*,?\s*[\]\)\}>]"  # Brackets with only commas and and spaces
    left_pattern = r"\(\s*,\s*([^()]+)\)"  # remove (,text)
    right_pattern = r"\(\s*([^()]+),\s*\)"  #  remove (text,)
    text = re.sub(pattern, "", message_text, flags=re.IGNORECASE)
    text = re.sub(empty_pattern, "", text)
    text = re.sub(left_pattern, r"\1", text)
    return re.sub(right_pattern, r"\1", text)


def citation_not_inserted(message_text: str, citation: Citation, footnote_counter: int) -> bool:
    return render_citation(citation, footnote_counter) not in message_text


def check_ref_ids_unique(message: ChatMessage) -> bool:
    ref_names = [citation.citation_name for citation in message.get_citations()]
    return len(ref_names) == len(set(ref_names))


def decorate_selected_files(all_files: Sequence[File], messages: Sequence[ChatMessage]) -> Sequence[File]:
    if messages:
        user_message_history = [m for m in messages if m.role == ChatMessage.Role.user]
        last_user_message = user_message_history[-1] if user_message_history else None
        selected_files: Sequence[File] = last_user_message.selected_files.all() if last_user_message else []
    else:
        selected_files = []

    for file in all_files:
        file.selected = file in selected_files
    return all_files


class MarkdownConverter:
    def __init__(self):
        self.md = markdown.Markdown(
            extensions=[
                "fenced_code",
                "tables",
                "sane_lists",
                "nl2br",
                "mdx_headdown",
            ],
            extension_configs={
                "mdx_headdown": {
                    "offset": 2,
                },
            },
        )
        self.convert = self.md.convert
        self.reset()

    def reset(self):
        self.md.reset()


def decorate_messages(
    messages: Sequence[ChatMessage] | None = None, as_html: bool = True
) -> Sequence[ChatMessage] | None:
    markdown_converter = MarkdownConverter()
    decorated_messages: Sequence[ChatMessage] | None = []

    # Add citatition links and footnotes to messages
    for message in messages:
        if as_html:
            markdown_converter.reset()
            message.text = markdown_converter.convert(message.text)

        decorated_message = decorate_message(message=message, as_html=False)
        decorated_messages.append(decorated_message)

    return decorated_messages


def decorate_message(message: ChatMessage, as_html: bool = True):
    if as_html:
        message.text = MarkdownConverter().convert(message.text)

    footnote_counter = 1

    for citation in message.get_citations():
        citation_names_unique = check_ref_ids_unique(message)

        if citation.citation_name and citation_names_unique:
            message.text = replace_ref(
                message_text=message.text,
                citation=citation,
                footnote_counter=footnote_counter,
            )

            if citation_not_inserted(
                message_text=message.text,
                citation=citation,
                footnote_counter=footnote_counter,
            ):
                logger.info("Citation Numbering Missed")
            else:
                footnote_counter += 1

        elif citation.text_in_answer:
            message.text = replace_text_in_answer(
                message_text=message.text,
                citation=citation,
                footnote_counter=footnote_counter,
            )
            footnote_counter += 1

            if citation_not_inserted(
                message_text=message.text,
                citation=citation,
                footnote_counter=footnote_counter,
            ):
                logger.info("Citation Numbering Missed")

    message.text = remove_dangling_citation(message_text=message.text)
    return message
