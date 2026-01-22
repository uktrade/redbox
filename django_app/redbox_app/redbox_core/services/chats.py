import logging
from collections.abc import Sequence
from uuid import UUID

from django.conf import settings
from django.contrib.auth import get_user_model
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.template.response import TemplateResponse
from waffle import flag_is_active
from yarl import URL

from redbox_app.redbox_core import flags
from redbox_app.redbox_core.models import Chat, ChatLLMBackend, ChatMessage, Tool
from redbox_app.redbox_core.services import documents as documents_service
from redbox_app.redbox_core.services import message as message_service
from redbox_app.redbox_core.services import url as url_service
from redbox_app.redbox_core.utils import resolve_instance

logger = logging.getLogger(__name__)
User = get_user_model()


def get_context(request: HttpRequest, chat_id: UUID | None = None, slug: str | None = None) -> dict:
    current_chat = _get_valid_chat(request.user, chat_id)
    chat_id = current_chat.id if current_chat else None
    tool = (
        current_chat.tool if current_chat else resolve_instance(value=slug, model=Tool, lookup="slug", raise_404=True)
    )

    if tool and current_chat and tool.settings.deselect_documents_on_load:
        current_chat.clear_selected_files()

    messages = ChatMessage.get_messages_ordered_by_citation_priority(chat_id) if current_chat else []
    endpoint = _build_ws_endpoint(request)
    file_context = documents_service.decorate_file_context(request, tool, messages)
    chat_backend = current_chat.chat_backend if current_chat else ChatLLMBackend.objects.get(is_default=True)
    messages = _decorate_messages(messages)

    urls = {
        "chat_url": url_service.get_chat_url(chat_id=chat_id, slug=slug),
        "new_chat_url": url_service.get_chat_url(chat_id=None, slug=slug),
        "upload_url": url_service.get_upload_url(slug=slug),
    }

    sidepanel_collapsed = request.COOKIES.get("rbds-side-panel-collapsed", "false") == "true"

    return {
        "tool": tool,
        "chat_id": chat_id,
        "messages": messages,
        "chats": Chat.get_ordered_by_last_message_date(request.user, tool),
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
        "errors": {"upload_doc": []},
        "request": request,
        "promoted_tool": Tool.objects.get(slug="submissions-checker") or None,
        "sidepanel_collapsed": sidepanel_collapsed,
    }


def _get_valid_chat(user: User, chat_id: UUID | None):
    if not chat_id:
        return None
    chat = get_object_or_404(Chat, id=chat_id)
    return chat if chat.user == user else None


def _decorate_messages(messages: Sequence[ChatMessage] | None = None):
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
            elif citation.text_in_answer:
                message.text = message_service.replace_text_in_answer(
                    message_text=message.text,
                    citation=citation,
                    footnote_counter=footnote_counter,
                )
                footnote_counter = footnote_counter + 1
                if message_service.citation_not_inserted(
                    message_text=message.text,
                    citation=citation,
                    footnote_counter=footnote_counter,
                ):
                    logger.info("Citation Numbering Missed")
        message.text = message_service.remove_dangling_citation(message_text=message.text)
    return messages


def _build_ws_endpoint(request: HttpRequest):
    return URL.build(
        scheme=settings.WEBSOCKET_SCHEME,
        host=("localhost" if settings.ENVIRONMENT.is_test else settings.ENVIRONMENT.hosts[0]),
        port=(int(request.META["SERVER_PORT"]) if settings.ENVIRONMENT.is_test else None),
        path=r"/ws/chat/",
    )


def render_chats(request: HttpRequest, context: dict) -> HttpResponse:
    return render(
        request,
        template_name="chats.html",
        context=context,
    )


def render_recent_chats(
    request: HttpRequest, active_chat_id: UUID | None = None, slug: str | None = None
) -> TemplateResponse:
    context = get_context(request, active_chat_id, slug)

    return TemplateResponse(
        request,
        "side_panel/conversations.html",
        context,
    )


def render_chat_window(
    request: HttpRequest, active_chat_id: UUID | None = None, slug: str | None = None
) -> TemplateResponse:
    context = get_context(request, active_chat_id, slug)

    return TemplateResponse(
        request,
        "chat/chat_window.html",
        context,
    )
