import logging
import uuid

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View

from redbox_app.redbox_core.models import Chat, ChatMessage, File, Skill
from redbox_app.redbox_core.services import url as url_service

logger = logging.getLogger(__name__)


class CitationsView(View):
    @method_decorator(login_required)
    def get(
        self,
        request: HttpRequest,
        message_id: uuid.UUID | None = None,
        slug: str | None = None,
        chat_id: uuid.UUID | None = None,
    ) -> HttpResponse:
        message = get_object_or_404(ChatMessage, id=message_id)
        skill = get_object_or_404(Skill, slug=slug) if slug else None
        chat = get_object_or_404(Chat, id=chat_id) if chat_id else None

        if message.chat.user != request.user:
            return redirect(reverse("chats"))

        source_files = File.get_ordered_by_citation_priority(message_id)
        citations_url = url_service.get_citation_url(message_id=message.id, chat_id=chat.id, slug=slug)

        context = {
            "message": message,
            "source_files": source_files,
            "skill": skill,
            "chat": chat,
            "citations_url": citations_url,
        }

        return render(
            request,
            template_name="citations.html",
            context=context,
        )
