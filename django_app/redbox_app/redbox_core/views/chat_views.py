import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from itertools import groupby
from operator import attrgetter

from dataclasses_json import Undefined, dataclass_json
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from waffle import flag_is_active
from yarl import URL

from redbox_app.redbox_core.models import Chat, ChatLLMBackend, ChatMessage, File

logger = logging.getLogger(__name__)


def replace_ref(message_text: str, ref_name: str, message_id: str, cit_id: str, footnote_counter: int) -> str:
    pattern = rf"[\[\(\{{<]{ref_name}[\]\)\}}>]|\b{ref_name}\b"
    message_text = re.sub(
        pattern,
        f'<a class="rb-footnote-link" href="/citations/{message_id}/#{cit_id}">{footnote_counter}</a>',
        message_text,
        # count=1,
    )
    return re.sub(pattern, "", message_text)


def replace_text_in_answer(
    message_text: str, text_in_answer: str, message_id: str, cit_id: str, footnote_counter: int
) -> str:
    return message_text.replace(
        text_in_answer,
        f'{text_in_answer}<a class="rb-footnote-link" href="/citations/{message_id}/#{cit_id}">{footnote_counter}</a>',
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


def citation_not_inserted(message_text, message_id, cit_id, footnote_counter) -> bool:
    return (
        f'<a class="rb-footnote-link" href="/citations/{message_id}/#{cit_id}">{footnote_counter}</a>'
        not in message_text
    )


def check_ref_ids_unique(message) -> bool:
    ref_names = [citation_tup[-1] for citation_tup in message.unique_citation_uris()]
    return len(ref_names) == len(set(ref_names))


class ChatsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest, chat_id: uuid.UUID | None = None) -> HttpResponse:
        chat = Chat.get_ordered_by_last_message_date(request.user)

        messages: Sequence[ChatMessage] = []
        current_chat = None
        if chat_id:
            current_chat = get_object_or_404(Chat, id=chat_id)
            if current_chat.user != request.user:
                return redirect(reverse("chats"))
            messages = ChatMessage.get_messages_ordered_by_citation_priority(chat_id)

        endpoint = URL.build(
            scheme=settings.WEBSOCKET_SCHEME,
            host="localhost" if settings.ENVIRONMENT.is_test else settings.ENVIRONMENT.hosts[0],
            port=int(request.META["SERVER_PORT"]) if settings.ENVIRONMENT.is_test else None,
            path=r"/ws/chat/",
        )

        completed_files, processing_files = File.get_completed_and_processing_files(request.user)

        self.decorate_selected_files(completed_files, messages)
        chat_grouped_by_date_group = groupby(chat, attrgetter("date_group"))

        chat_backend = current_chat.chat_backend if current_chat else ChatLLMBackend.objects.get(is_default=True)

        # Add footnotes to messages
        for message in messages:
            footnote_counter = 1
            for display, href, cit_id, text_in_answer, citation_name in message.unique_citation_uris():  # noqa: B007
                citation_names_unique = check_ref_ids_unique(message)
                if citation_name and citation_names_unique:
                    message.text = replace_ref(
                        message_text=message.text,
                        ref_name=citation_name,
                        message_id=message.id,
                        cit_id=cit_id,
                        footnote_counter=footnote_counter,
                    )

                    if citation_not_inserted(
                        message_text=message.text,
                        message_id=message.id,
                        cit_id=cit_id,
                        footnote_counter=footnote_counter,
                    ):
                        logger.info("Citation Numbering Missed")
                    else:
                        footnote_counter = footnote_counter + 1
                elif text_in_answer:
                    message.text = replace_text_in_answer(
                        message_text=message.text,
                        text_in_answer=text_in_answer,
                        message_id=message.id,
                        cit_id=cit_id,
                        footnote_counter=footnote_counter,
                    )
                    footnote_counter = footnote_counter + 1
                    if citation_not_inserted(
                        message_text=message.text,
                        message_id=message.id,
                        cit_id=cit_id,
                        footnote_counter=footnote_counter,
                    ):
                        logger.info("Citation Numbering Missed")
            message.text = remove_dangling_citation(message_text=message.text)

        context = {
            "chat_id": chat_id,
            "messages": messages,
            "chat_grouped_by_date_group": chat_grouped_by_date_group,
            "current_chat": current_chat,
            "streaming": {"endpoint": str(endpoint)},
            "contact_email": settings.CONTACT_EMAIL,
            "completed_files": completed_files,
            "processing_files": processing_files,
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
        }

        return render(
            request,
            template_name="chats.html",
            context=context,
        )

    @staticmethod
    def decorate_selected_files(all_files: Sequence[File], messages: Sequence[ChatMessage]) -> None:
        if messages:
            last_user_message = [m for m in messages if m.role == ChatMessage.Role.user][-1]
            selected_files: Sequence[File] = last_user_message.selected_files.all() or []
        else:
            selected_files = []

        for file in all_files:
            file.selected = file in selected_files


class ChatsTitleView(View):
    @dataclass_json(undefined=Undefined.EXCLUDE)
    @dataclass(frozen=True)
    class Title:
        name: str

    @method_decorator(login_required)
    def post(self, request: HttpRequest, chat_id: uuid.UUID) -> HttpResponse:
        chat: Chat = get_object_or_404(Chat, id=chat_id)
        user_rating = ChatsTitleView.Title.schema().loads(request.body)

        chat.name = user_rating.name
        chat.save(update_fields=["name"])

        return HttpResponse(status=HTTPStatus.NO_CONTENT)


class UpdateChatFeedback(View):
    @method_decorator(login_required)
    def post(self, request: HttpRequest, chat_id: uuid.UUID) -> HttpResponse:
        def convert_to_boolean(value: str):
            return value == "Yes"

        chat: Chat = get_object_or_404(Chat, id=chat_id)
        chat.feedback_achieved = convert_to_boolean(request.POST.get("achieved"))
        chat.feedback_saved_time = convert_to_boolean(request.POST.get("saved_time"))
        chat.feedback_improved_work = convert_to_boolean(request.POST.get("improved_work"))
        chat.feedback_notes = request.POST.get("notes")
        chat.save()
        return HttpResponse(status=HTTPStatus.NO_CONTENT)


class DeleteChat(View):
    @method_decorator(login_required)
    def post(self, request: HttpRequest, chat_id: uuid.UUID) -> HttpResponse:  # noqa: ARG002
        chat: Chat = get_object_or_404(Chat, id=chat_id)
        chat.archived = True
        chat.save()
        return HttpResponse(status=HTTPStatus.NO_CONTENT)
