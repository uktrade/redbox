import logging
import re
from collections.abc import Sequence

from redbox_app.redbox_core.models import ChatMessage, Citation, File

logger = logging.getLogger(__name__)


def replace_ref(
    message_text: str,
    ref_name: str,
    cit_id: str,
    footnote_counter: int,
) -> str:
    pattern = rf"[\[\(\{{<]{ref_name}[\]\)\}}>]|\b{ref_name}\b"
    citation = Citation.objects.get(id=cit_id)
    message_text = re.sub(
        pattern,
        f'<a class="rb-footnote-link" href="{citation.internal_url}">{footnote_counter}</a>',
        message_text,
        # count=1,
    )
    return re.sub(pattern, "", message_text)


def replace_text_in_answer(
    message_text: str,
    text_in_answer: str,
    cit_id: str,
    footnote_counter: int,
) -> str:
    citation = Citation.objects.get(id=cit_id)
    return message_text.replace(
        text_in_answer,
        f'{text_in_answer}<a class="rb-footnote-link" href="{citation.internal_url}">{footnote_counter}</a>',
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


def citation_not_inserted(message_text, cit_id, footnote_counter) -> bool:
    citation = Citation.objects.get(id=cit_id)
    return f'<a class="rb-footnote-link" href="{citation.internal_url}">{footnote_counter}</a>' not in message_text


def check_ref_ids_unique(message) -> bool:
    ref_names = [citation_tup[-1] for citation_tup in message.unique_citation_uris()]
    return len(ref_names) == len(set(ref_names))


def decorate_selected_files(all_files: Sequence[File], messages: Sequence[ChatMessage]) -> Sequence[File]:
    if messages:
        last_user_message = [m for m in messages if m.role == ChatMessage.Role.user][-1]
        selected_files: Sequence[File] = last_user_message.selected_files.all() or []
    else:
        selected_files = []

    for file in all_files:
        file.selected = file in selected_files
    return all_files
