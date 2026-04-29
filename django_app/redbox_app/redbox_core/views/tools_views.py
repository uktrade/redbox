import logging
import uuid
from http import HTTPStatus
from typing import Any

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods

from redbox_app.redbox_core.models import Tool, UserTool
from redbox_app.redbox_core.services import chats as chat_service

User = get_user_model()
logger = logging.getLogger(__name__)


class ToolsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        return render(
            request,
            template_name="tools/tools.html",
            context=chat_service.get_context(request),
        )


@require_http_methods(["GET"])
def tool_info_page_view(request: HttpRequest, slug: str) -> HttpResponse:
    tool = get_object_or_404(Tool, slug=slug)
    context = chat_service.get_context(request, slug=slug)

    if not tool.has_info_page:
        return HttpResponse(
            f"Tool info page not found: {slug}",
            status=HTTPStatus.NOT_FOUND,
        )

    return render(request, tool.info_template, context=context)


class ToolSettingsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest, slug: str) -> HttpResponse:
        tool = get_object_or_404(Tool, slug=slug)

        return render(
            request,
            template_name="tools/settings.html",
            context={
                **chat_service.get_context(request, slug=slug),
                "users": User.objects.all(),
                "user_tool_role_choices": UserTool.RoleType.choices,
                "user_tool_access_choices": UserTool.AccessType.choices,
                "user_tool_memberships": UserTool.objects.filter(tool=tool),
                "tool": tool,
            },
        )


@require_http_methods(["GET"])
@login_required
def edit_tool_user_row_view(request: HttpRequest, slug: str, user_tool_id: uuid.UUID | Any):
    user_tool_membership = get_object_or_404(UserTool, id=user_tool_id)
    cancel = request.GET.get("cancel") or "false"

    if not user_tool_membership.tool.is_manager(request.user):
        return HttpResponse("You are not a manager for this tool.", status=HTTPStatus.FORBIDDEN)

    context = {
        "user_tool_membership": user_tool_membership,
        "user_tool_role_choices": UserTool.RoleType.choices,
        "user_tool_access_choices": UserTool.AccessType.choices,
        "editing": cancel.lower() != "true",
        "tool": get_object_or_404(Tool, slug=slug),
    }

    return render(request, "tools/tool-user-row.html", context)


@require_http_methods(["DELETE"])
@login_required
def delete_tool_user_row_view(request: HttpRequest, slug: str, user_tool_id: uuid.UUID | Any):
    user_tool_membership = UserTool.objects.filter(id=user_tool_id).select_related("tool").first()
    tool = get_object_or_404(Tool, slug=slug)

    if tool:
        logger.debug("placeholder")

    if user_tool_membership:
        if not user_tool_membership.tool.is_manager(request.user):
            return HttpResponse("You are not a manager for this tool.", status=HTTPStatus.FORBIDDEN)
        user_tool_membership.delete()
    else:
        return HttpResponse(status=HTTPStatus.NOT_FOUND)

    return HttpResponse(status=HTTPStatus.OK)


@require_http_methods(["POST"])
@login_required
def edit_tool_user_view(request: HttpRequest, slug: str, user_tool_id: uuid.UUID | Any):
    user_tool_membership = get_object_or_404(UserTool, id=user_tool_id)
    role_type = request.POST.get("role-type")

    if not user_tool_membership.tool.is_manager(request.user):
        return HttpResponse("You are not a manager for this tool.", status=HTTPStatus.FORBIDDEN)

    try:
        user_tool_membership.role = role_type
        user_tool_membership.save()
    except Exception as e:
        error = "Failed to update UserTool object"
        error_exception = f"{error}: {e}"
        logger.exception(error_exception)
        return HttpResponse(error, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    context = {
        "user_tool_membership": user_tool_membership,
        "user_tool_role_choices": UserTool.RoleType.choices,
        "user_tool_access_choices": UserTool.AccessType.choices,
        "show_select": False,
        "tool": get_object_or_404(Tool, slug=slug),
    }

    return render(request, "tools/tool-user-row.html", context)


@require_http_methods(["POST"])
@login_required
def add_tool_user_view(request: HttpRequest, slug: str):
    tool = get_object_or_404(Tool, slug=slug)

    user_ids = request.POST.getlist("user_ids")
    role = request.POST.get("role")
    access = request.POST.get("access")
    next_url = request.GET.get("next")

    for user_id in user_ids:
        try:
            tool.add_user(user=user_id, role=role, access=access)
        except Exception as e:
            error = "Failed to add tool user"
            error_exception = f"{error}: {e}"
            logger.exception(error_exception)
            return HttpResponse(error, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    if next_url:
        return redirect(next_url)

    return HttpResponse(status=HTTPStatus.OK)
