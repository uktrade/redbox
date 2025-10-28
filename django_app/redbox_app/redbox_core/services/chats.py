import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from django.conf import settings
from django.http import HttpRequest
from django.shortcuts import get_object_or_404
from django.template.response import TemplateResponse
from waffle import flag_is_active
from yarl import URL

from redbox_app.redbox_core.models import Chat, ChatLLMBackend, ChatMessage
from redbox_app.redbox_core.services import documents as documents_service
from redbox_app.redbox_core.services import message as message_service

logger = logging.getLogger(__name__)


def get_context(request: HttpRequest, chat_id: uuid.UUID | None = None) -> dict:
    chat = Chat.get_ordered_by_last_message_date(request.user)

    file_context = documents_service.get_file_context(request)

    messages: Sequence[ChatMessage] = []
    current_chat = None
    if chat_id:
        current_chat = get_object_or_404(Chat, id=chat_id)
        if current_chat.user != request.user:
            chat_id = None
        else:
            messages = ChatMessage.get_messages_ordered_by_citation_priority(chat_id)

    endpoint = URL.build(
        scheme=settings.WEBSOCKET_SCHEME,
        host=("localhost" if settings.ENVIRONMENT.is_test else settings.ENVIRONMENT.hosts[0]),
        port=(int(request.META["SERVER_PORT"]) if settings.ENVIRONMENT.is_test else None),
        path=r"/ws/chat/",
    )

    completed_files = message_service.decorate_selected_files(file_context["completed_files"], messages)

    file_context["completed_files"] = completed_files

    chat_backend = current_chat.chat_backend if current_chat else ChatLLMBackend.objects.get(is_default=True)

    # Add footnotes to messages
    for message in messages:
        footnote_counter = 1
        for (
            _display,
            _href,
            cit_id,
            text_in_answer,
            citation_name,
        ) in message.unique_citation_uris():
            citation_names_unique = message_service.check_ref_ids_unique(message)
            if citation_name and citation_names_unique:
                message.text = message_service.replace_ref(
                    message_text=message.text,
                    ref_name=citation_name,
                    message_id=message.id,
                    cit_id=cit_id,
                    footnote_counter=footnote_counter,
                )

                if message_service.citation_not_inserted(
                    message_text=message.text,
                    message_id=message.id,
                    cit_id=cit_id,
                    footnote_counter=footnote_counter,
                ):
                    logger.info("Citation Numbering Missed")
                else:
                    footnote_counter = footnote_counter + 1
            elif text_in_answer:
                message.text = message_service.replace_text_in_answer(
                    message_text=message.text,
                    text_in_answer=text_in_answer,
                    message_id=message.id,
                    cit_id=cit_id,
                    footnote_counter=footnote_counter,
                )
                footnote_counter = footnote_counter + 1
                if message_service.citation_not_inserted(
                    message_text=message.text,
                    message_id=message.id,
                    cit_id=cit_id,
                    footnote_counter=footnote_counter,
                ):
                    logger.info("Citation Numbering Missed")
        message.text = message_service.remove_dangling_citation(message_text=message.text)

    return {
        "chat_id": chat_id,
        "messages": messages,
        "chats": chat,
        "current_chat": current_chat,
        "streaming": {"endpoint": str(endpoint)},
        "contact_email": settings.CONTACT_EMAIL,
        "chat_title_length": settings.CHAT_TITLE_LENGTH,
        "llm_options": [
            {
                "name": str(chat_llm_backend),
                "default": chat_llm_backend.is_default,
                "selected": chat_llm_backend == chat_backend,
                "id": chat_llm_backend.id,
            }
            for chat_llm_backend in ChatLLMBackend.objects.filter(enabled=True)
        ],
        "redbox_api_key": settings.REDBOX_API_KEY,
        "enable_dictation_flag_is_active": flag_is_active(request, "enable_dictation"),
        **file_context,
    }


def render_recent_chats(request, active_chat_id) -> TemplateResponse:
    context = get_context(request, active_chat_id)

    return TemplateResponse(
        request,
        "side_panel/recent_chats_list.html",
        context,
    )


def render_chat_window(request, active_chat_id) -> TemplateResponse:
    context = get_context(request, active_chat_id)

    return TemplateResponse(
        request,
        "chat/chat_window.html",
        context,
    )
