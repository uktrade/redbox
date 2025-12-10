import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.template.response import TemplateResponse
from waffle import flag_is_active
from yarl import URL

from redbox_app.redbox_core import flags
from redbox_app.redbox_core.models import Chat, ChatLLMBackend, ChatMessage, Skill
from redbox_app.redbox_core.services import documents as documents_service
from redbox_app.redbox_core.services import message as message_service
from redbox_app.redbox_core.services import url as url_service

logger = logging.getLogger(__name__)


def get_context(request: HttpRequest, chat_id: uuid.UUID | None = None, skill_slug: str | None = None) -> dict:
    messages: Sequence[ChatMessage] = []
    current_chat = None
    skill = None

    if chat_id:
        current_chat = get_object_or_404(Chat, id=chat_id)
        skill = current_chat.skill
        if current_chat.user != request.user:
            chat_id = None
        else:
            messages = ChatMessage.get_messages_ordered_by_citation_priority(chat_id)

    if not skill and skill_slug:
        skill = get_object_or_404(Skill, slug=skill_slug)

    endpoint = URL.build(
        scheme=settings.WEBSOCKET_SCHEME,
        host=("localhost" if settings.ENVIRONMENT.is_test else settings.ENVIRONMENT.hosts[0]),
        port=(int(request.META["SERVER_PORT"]) if settings.ENVIRONMENT.is_test else None),
        path=r"/ws/chat/",
    )

    file_context = documents_service.get_file_context(request, skill)
    completed_files = message_service.decorate_selected_files(file_context["completed_files"], messages)
    file_context["completed_files"] = completed_files

    chat_backend = current_chat.chat_backend if current_chat else ChatLLMBackend.objects.get(is_default=True)

    # Add footnotes to messages
    for message in messages:
        footnote_counter = 1
        for citation in message.get_citations():
            citation_names_unique = message_service.check_ref_ids_unique(message)
            if citation.citation_name and citation_names_unique:
                message.text = message_service.replace_ref(
                    message_text=message.text,
                    citation=citation,
                    footnote_counter=footnote_counter,
                )

                if message_service.citation_not_inserted(
                    message_text=message.text,
                    citation=citation,
                    footnote_counter=footnote_counter,
                ):
                    logger.info("Citation Numbering Missed")
                else:
                    footnote_counter = footnote_counter + 1
            # elif citation.text_in_answer:
            #     message.text = message_service.replace_text_in_answer(
            #         message_text=message.text,
            #         citation=citation,
            #         footnote_counter=footnote_counter,
            #     )
            #     footnote_counter = footnote_counter + 1
            #     if message_service.citation_not_inserted(
            #         message_text=message.text,
            #         citation=citation,
            #         footnote_counter=footnote_counter,
            #     ):
            #         logger.info("Citation Numbering Missed")
        message.text = message_service.remove_dangling_citation(message_text=message.text)

    urls = {
        "chat_url": url_service.get_chat_url(chat_id, skill_slug),
        "new_chat_url": url_service.get_chat_url(None, skill_slug),
    }

    return {
        "skill": skill,
        "chat_id": chat_id,
        "messages": messages,
        "chats": Chat.get_ordered_by_last_message_date(request.user, skill),
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
        "enable_dictation_flag_is_active": flag_is_active(request, flags.ENABLE_DICTATION),
        **file_context,
        "urls": urls,
    }


def render_chats(request: HttpRequest, context: dict) -> HttpResponse:
    return render(
        request,
        template_name="chats.html",
        context=context,
    )


def render_recent_chats(
    request: HttpRequest, active_chat_id: uuid.UUID | None = None, skill_slug: str | None = None
) -> TemplateResponse:
    context = get_context(request, active_chat_id, skill_slug)

    return TemplateResponse(
        request,
        "side_panel/recent_chats_list.html",
        context,
    )


def render_chat_window(
    request: HttpRequest, active_chat_id: uuid.UUID | None = None, skill_slug: str | None = None
) -> TemplateResponse:
    context = get_context(request, active_chat_id, skill_slug)

    return TemplateResponse(
        request,
        "chat/chat_window.html",
        context,
    )
