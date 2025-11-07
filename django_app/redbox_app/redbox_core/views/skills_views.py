import logging
import uuid
from http import HTTPStatus

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods

from redbox_app.redbox_core.models import Skill
from redbox_app.redbox_core.services import chats as chat_service

User = get_user_model()
logger = logging.getLogger(__name__)


class SkillsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        return render(
            request,
            template_name="skills/skills.html",
            context={"skills": Skill.objects.all()},
        )


class SkillChatsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest, skill_slug: str, chat_id: uuid.UUID | None = None) -> HttpResponse:
        skill = get_object_or_404(Skill, slug=skill_slug)

        context = chat_service.get_context(request, chat_id)
        context["active_skill"] = skill

        if chat_id != context["chat_id"]:
            return redirect(reverse("skill-chats", args=(skill_slug)))

        return chat_service.render_chats(request, context)


@require_http_methods(["GET"])
def skill_info_page_view(request: HttpRequest, skill_slug: str):
    skill = get_object_or_404(Skill, slug=skill_slug)

    if not skill.has_info_page:
        return HttpResponse(
            f"Skill info page not found: {skill_slug}",
            status=HTTPStatus.NOT_FOUND,
        )

    return render(request, skill.info_template, context={"skill": skill})
