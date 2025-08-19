import io
import json
import logging
import uuid
from dataclasses import dataclass
from html.parser import HTMLParser
from http import HTTPStatus

from dataclasses_json import Undefined, dataclass_json
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from docx import Document

from redbox_app.redbox_core.models import Chat
from redbox_app.redbox_core.services import chats as chat_service
from redbox_app.redbox_core.utils import render_with_oob

logger = logging.getLogger(__name__)


class ChatsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest, chat_id: uuid.UUID | None = None) -> HttpResponse:
        context = chat_service.get_context(request, chat_id)

        if chat_id != context["chat_id"]:
            return redirect(reverse("chats"))

        return render(
            request,
            template_name="chats.html",
            context=context,
        )


class ChatsTitleView(View):
    @dataclass_json(undefined=Undefined.EXCLUDE)
    @dataclass(frozen=True)
    class Title:
        value: str

    @method_decorator(login_required)
    def post(self, request: HttpRequest, chat_id: uuid.UUID) -> HttpResponse:
        chat: Chat = get_object_or_404(Chat, id=chat_id)
        request_body = ChatsTitleView.Title.schema().loads(request.body)

        chat.name = request_body.value
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
    def post(self, request: HttpRequest, chat_id: uuid.UUID) -> HttpResponse:
        active_chat_deleted = False

        chat: Chat = get_object_or_404(Chat, id=chat_id)
        chat.archived = True
        chat.save()

        session_id = request.POST.get("session-id")
        active_chat_id = session_id if session_id else request.POST.get("active_chat_id")

        if active_chat_id != "None":
            active_chat_id = uuid.UUID(active_chat_id)

            if active_chat_id == chat_id:
                active_chat_id = None
                active_chat_deleted = True
        else:
            active_chat_id = None

        if active_chat_deleted:
            context = chat_service.get_context(request, active_chat_id)
            return render_with_oob(
                [
                    {"template": "side_panel/recent_chats_list.html", "context": context, "request": request},
                    {"template": "chat/chat_window.html", "context": context, "request": request},
                ]
            )
        return chat_service.render_recent_chats(request, active_chat_id)


class RecentChats(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest, active_chat_id: uuid.UUID) -> HttpResponse:
        return chat_service.render_recent_chats(request, active_chat_id)


class ChatWindow(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest, active_chat_id: uuid.UUID) -> HttpResponse:
        return chat_service.render_chat_window(request, active_chat_id)


@csrf_exempt
@require_POST
def generate_docx(request):
    try:
        data = json.loads(request.body)
        html_content = data.get("html_content", "")
        text_content = data.get("text_content", "")
        message_id = data.get("message_id", "document")

        doc = Document()

        class HTMLStripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []

            def handle_data(self, data):
                self.text.append(data)

        parser = HTMLStripper()
        parser.feed(html_content)
        cleaned_text = " ".join(parser.text).strip()

        doc.add_paragraph(cleaned_text if cleaned_text else text_content)

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        response = HttpResponse(
            content=buffer.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        response["Content-Disposition"] = f"attachment; filename=draft-{message_id}.docx"
        return response

    except Exception as e:
        return HttpResponse(status=500, content=str(e))
