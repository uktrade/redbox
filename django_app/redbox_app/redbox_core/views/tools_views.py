import logging
from http import HTTPStatus

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods

from redbox_app.redbox_core.models import Tool
from redbox_app.redbox_core.services import chats as chat_service

User = get_user_model()
logger = logging.getLogger(__name__)


class ToolsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        context = chat_service.get_context(request)
        context["tools"] = Tool.objects.all()

        return render(
            request,
            template_name="tools/tools.html",
            context=context,
        )


@require_http_methods(["GET"])
def tool_info_page_view(request: HttpRequest, slug: str) -> HttpResponse:
    tool = get_object_or_404(Tool, slug=slug)
    context = chat_service.get_context(request)
    context["tool"] = tool

    if not tool.has_info_page:
        return HttpResponse(
            f"Tool info page not found: {slug}",
            status=HTTPStatus.NOT_FOUND,
        )

    return render(request, tool.info_template, context=context)
