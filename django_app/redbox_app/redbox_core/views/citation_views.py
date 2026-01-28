import logging
import uuid

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View

from redbox_app.redbox_core.models import ChatMessage, File
from redbox_app.redbox_core.services import chats as chat_service

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

        if message.chat.user != request.user:
            return redirect(reverse("chats"))

        context = {
            **chat_service.get_context(request, chat_id, slug),
            "message": message,
            "source_files": File.get_ordered_by_citation_priority(message_id),
        }

        return render(
            request,
            template_name="citations.html",
            context=context,
        )
