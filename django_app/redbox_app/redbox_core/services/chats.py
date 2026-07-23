import logging
from collections.abc import Sequence
from datetime import date, datetime
from uuid import UUID
from zoneinfo import ZoneInfo

from django.conf import settings
from django.contrib.auth import get_user_model
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render
from waffle import flag_is_active
from yarl import URL

from redbox_app.redbox_core import flags
from redbox_app.redbox_core.models import Chat, ChatLLMBackend, ChatMessage, Tool, UserTeamMembership
from redbox_app.redbox_core.services import documents as documents_service
from redbox_app.redbox_core.services import message as message_service
from redbox_app.redbox_core.services import url as url_service
from redbox_app.redbox_core.types import FilterChat, GroupedChats
from redbox_app.redbox_core.utils import resolve_instance

logger = logging.getLogger(__name__)
User = get_user_model()


def get_context(request: HttpRequest, chat_id: UUID | None = None, slug: str | None = None, **kwargs) -> dict:
    if not request.user.is_authenticated:
        return {"request": request, "contact_email": settings.CONTACT_EMAIL}

    if kwargs:
        slug = kwargs.get("slug", slug)
        chat_id = kwargs.get("chat_id", chat_id)

    current_chat = _get_valid_chat(request.user, chat_id)
    chat_id = current_chat.id if current_chat else None
    tool = (
        current_chat.tool if current_chat else resolve_instance(value=slug, model=Tool, lookup="slug", raise_404=True)
    )

    if tool and current_chat and tool.settings.deselect_documents_on_load:
        current_chat.clear_selected_files()

    tools = Tool.objects.for_user(request.user)

    messages = ChatMessage.get_messages_ordered_by_citation_priority(chat_id) if current_chat else []
    endpoint = _build_ws_endpoint(request)
    file_context = documents_service.decorate_file_context(request, tool, messages)
    chat_backend = current_chat.chat_backend if current_chat else ChatLLMBackend.objects.get(is_default=True)
    messages = message_service.decorate_messages(messages)

    urls = {
        "chat_url": url_service.get_chat_url(chat_id=chat_id, slug=slug),
        "new_chat_url": url_service.get_chat_url(chat_id=None, slug=slug),
        "upload_url": url_service.get_upload_url(slug=slug),
    }

    sidepanel_collapsed = request.COOKIES.get("ids-side-panel-collapsed", "false") == "true"

    context = {
        "tool": tool,
        "tools": tools,
        "chat_id": chat_id,
        "messages": messages,
        "chats": Chat.get_ordered_by_last_message_date(request.user, tool),
        "current_chat": current_chat,
        "streaming": {"endpoint": str(endpoint)},
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
        "enable_chats_redesign": flag_is_active(request, flags.ENABLE_CHATS_REDESIGN),
        **file_context,
        "urls": urls,
        "errors": {"upload_doc": []},
        "request": request,
        "promoted_tool": Tool.objects.filter(slug="submissions-checker").first() or None,
        "sidepanel_collapsed": sidepanel_collapsed,
        "pageTitle": _get_page_title(current_chat, tool),
    }

    if flag_is_active(request.user, flags.ENABLE_TEAMS):
        context["user_teams"] = UserTeamMembership.objects.filter(user=request.user)

    return context


def _get_valid_chat(user: User, chat_id: UUID | None):
    if not chat_id:
        return None
    chat = get_object_or_404(Chat, id=chat_id)
    return chat if chat.user == user else None


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


def render_conversations(request: HttpRequest, context: dict | None = None) -> HttpResponse:
    return render(
        request,
        template_name="side_panel/conversations.html",
        context=context or get_context(request),
    )


def _get_page_title(current_chat: Chat, tool: Tool):
    tool_name = tool.name if tool else None
    chat_name = current_chat.name if current_chat else "New chat"

    parts = [chat_name, "Chats", tool_name]

    return " - ".join(part for part in parts if part)


def get_filtered_and_grouped_chats(
    user: User, tz: ZoneInfo, tool: Tool | None = None, chat_name_query: str | None = None
) -> GroupedChats:
    chats = Chat.filter_by_name_ordered_by_last_message_date(user, tool, chat_name_query)
    filter_chats = [FilterChat.from_chat(chat, tz) for chat in chats]

    localised_date = datetime.now(tz=tz).date()
    return _group_chats(filter_chats, localised_date)


def _group_chats(chats: list[FilterChat], localised_date: date) -> GroupedChats:
    td_today = 0
    td_yesterday = 1
    td_a_week_ago = 7
    td_a_month_ago = 30
    td_a_year_ago = 365

    grouped_chats = GroupedChats()
    for chat in chats:
        date = chat.local_last_message_datetime.date()
        delta = (localised_date - date).days

        if delta < td_today:
            logger.warning("Found chat with a latest message in the future when grouping all chats")
            grouped_chats.today.append(chat)
        elif delta <= td_today:
            grouped_chats.today.append(chat)
        elif delta == td_yesterday:
            grouped_chats.yesterday.append(chat)
        elif td_yesterday < delta <= td_a_week_ago:
            grouped_chats.previous_7_days.append(chat)
        elif td_a_week_ago < delta <= td_a_month_ago:
            grouped_chats.previous_30_days.append(chat)
        elif td_a_month_ago < delta <= td_a_year_ago:
            grouped_chats.previous_year.append(chat)
        else:
            grouped_chats.over_a_year.append(chat)

    return grouped_chats
