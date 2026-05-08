import logging
import uuid
from http import HTTPStatus
from typing import Any

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods
from django.views.generic import CreateView, FormView, UpdateView

from redbox_app.redbox_core.forms import ToolAccessRuleForm, ToolSettingsForm, UserToolBulkAddForm, UserToolForm
from redbox_app.redbox_core.models import Tool, ToolAccessRule, UserTool
from redbox_app.redbox_core.services import chats as chat_service
from redbox_app.redbox_core.services import url as url_service
from redbox_app.redbox_core.utils import is_htmx_request
from redbox_app.redbox_core.views.mixins import AppContextMixin

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


class ToolSettingsView(LoginRequiredMixin, AppContextMixin, UpdateView):
    model = Tool
    form_class = ToolSettingsForm
    template_name = "tools/settings.html"
    slug_field = "slug"
    slug_url_kwarg = "slug"

    def get_success_url(self):
        return reverse("tool-settings", kwargs={"slug": self.object.slug})

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        tool = self.object

        context.update(
            {
                "user_tools": UserTool.objects.filter(tool=tool),
                "tool_access_rules": ToolAccessRule.objects.filter(tool=tool),
                "add_tool_access_rule_url": url_service.get_add_tool_access_rule_url(slug=tool.slug),
                "bulk_add_user_tool_url": url_service.get_bulk_add_user_tool_url(slug=tool.slug),
            }
        )

        return context


class ToolAccessRuleCreateView(LoginRequiredMixin, AppContextMixin, CreateView):
    model = ToolAccessRule
    form_class = ToolAccessRuleForm
    template_name = "tools/rules/form.html"

    def dispatch(self, request, *args, **kwargs):
        self.tool = get_object_or_404(Tool, slug=kwargs["slug"])
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        form.instance.tool = self.tool
        return super().form_valid(form)

    def get_success_url(self):
        return reverse("tool-settings", kwargs={"slug": self.tool.slug}, fragment="access-rules")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context.update(
            {
                "tool": self.tool,
                "form_mode": "create",
                "affected_users_preview_url": url_service.get_affected_users_preview_url(self.tool.slug),
            }
        )

        return context


class ToolAccessRuleUpdateView(LoginRequiredMixin, AppContextMixin, UpdateView):
    model = ToolAccessRule
    form_class = ToolAccessRuleForm
    template_name = "tools/rules/form.html"
    pk_url_kwarg = "rule_id"

    def dispatch(self, request, *args, **kwargs):
        self.tool = get_object_or_404(Tool, slug=kwargs["slug"])
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        form.instance.tool = self.tool
        return super().form_valid(form)

    def get_queryset(self):
        return ToolAccessRule.objects.filter(tool__slug=self.kwargs["slug"])

    def get_success_url(self):
        return reverse("tool-settings", kwargs={"slug": self.tool.slug}, fragment="access-rules")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        tool = self.tool

        context.update(
            {
                "tool": tool,
                "form_mode": "edit",
                "affected_users_preview_url": url_service.get_affected_users_preview_url(tool.slug),
            }
        )

        return context


class ToolAccessRuleDeleteView(LoginRequiredMixin, View):
    def delete(self, request, **kwargs):
        rule = get_object_or_404(
            ToolAccessRule,
            pk=kwargs["rule_id"],
            tool__slug=kwargs["slug"],
        )

        rule.delete()

        if is_htmx_request(request):
            return HttpResponse(status=HTTPStatus.OK)

        return redirect(reverse("tool-settings", kwargs={"slug": kwargs["slug"]}, fragment="access-rules"))

    def post(self, request, *args, **kwargs):
        return self.delete(request, *args, **kwargs)


class UserToolBulkAddView(LoginRequiredMixin, AppContextMixin, FormView):
    template_name = "tools/users/bulk-add-form.html"
    form_class = UserToolBulkAddForm

    def dispatch(self, request, *args, **kwargs):
        self.tool = get_object_or_404(Tool, slug=kwargs["slug"])
        return super().dispatch(request, *args, **kwargs)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()

        kwargs["unassigned_users"] = self.tool.get_unassigned_users()

        return kwargs

    def form_valid(self, form):
        users = form.cleaned_data["user_ids"]
        role = form.cleaned_data["role"]

        for user in users:
            UserTool.objects.update_or_create(
                user=user,
                tool=self.tool,
                defaults={"role": role},
            )

        return super().form_valid(form)

    def get_success_url(self):
        return reverse("tool-settings", kwargs={"slug": self.tool.slug}, fragment="users")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context.update(
            {
                "tool": self.tool,
                "unassigned_users": self.tool.get_unassigned_users(),
            }
        )

        return context


@require_http_methods(["POST"])
@login_required
def tool_access_rule_preview(request: HttpRequest, slug: str):
    tool = get_object_or_404(Tool, slug=slug)
    rule = ToolAccessRule(
        tool=tool,
        rule_type=request.POST.get("rule_type"),
        value=request.POST.get("value"),
        access_type=request.POST.get("access_type"),
    )

    users_preview = rule.get_matching_users()

    context = {
        "users": users_preview[:100],
        "total_count": users_preview.count(),
        "rule": rule,
    }

    return render(request, "tools/rules/affected-users-preview.html", context)


@require_http_methods(["GET"])
@login_required
def tool_access_rule_value_input_view(request: HttpRequest):
    rule_type = request.GET.get("rule_type")
    placeholder = ToolAccessRule.get_value_placeholder(rule_type)

    form = ToolAccessRuleForm()
    form.fields["value"].widget.attrs["placeholder"] = placeholder

    context = {"form": form}

    return render(request, "tools/rules/tool-access-rule-value-input.html", context)


@require_http_methods(["GET"])
@login_required
def edit_tool_user_row_view(request, slug, user_tool_id):
    user_tool = get_object_or_404(UserTool, pk=user_tool_id)

    cancel = request.GET.get("cancel") == "true"

    if cancel:
        return render(
            request,
            "tools/tool-user-row.html",
            {
                "user_tool": user_tool,
                "tool": user_tool.tool,
                "editing": False,
            },
        )

    form = UserToolForm(instance=user_tool)

    tool = user_tool.tool

    cancel_url = url_service.get_edit_user_tool_row_url(
        slug=slug,
        user_tool_id=user_tool.pk,
    )
    save_url = url_service.get_edit_user_tool_url(
        slug=tool.slug,
        user_tool_id=user_tool.pk,
    )

    return render(
        request,
        "tools/users/expanded-row-form.html",
        {
            "form": form,
            "user_tool": user_tool,
            "tool": tool,
            "htmx": True,
            "cancel_url": cancel_url,
            "save_url": save_url,
        },
    )


@require_http_methods(["DELETE"])
@login_required
def delete_tool_user_row_view(request: HttpRequest, slug: str, user_tool_id: uuid.UUID | Any):
    user_tool = UserTool.objects.filter(id=user_tool_id).select_related("tool").first()
    tool = get_object_or_404(Tool, slug=slug)

    if tool:
        logger.debug("placeholder")

    if user_tool:
        if not user_tool.tool.is_manager(request.user):
            return HttpResponse("You are not a manager for this tool.", status=HTTPStatus.FORBIDDEN)
        user_tool.delete()
    else:
        return HttpResponse(status=HTTPStatus.NOT_FOUND)

    return HttpResponse(status=HTTPStatus.OK)


@require_http_methods(["POST"])
@login_required
def edit_tool_user_view(request: HttpRequest, slug: str, user_tool_id: uuid.UUID | Any):
    user_tool = get_object_or_404(UserTool, pk=user_tool_id)
    user = user_tool.user
    form = UserToolForm(request.POST, instance=user_tool)

    form.fields["user"].required = False

    if form.is_valid():
        obj = form.save(commit=False)
        obj.user = user
        obj.save()

        if is_htmx_request(request):
            return render(
                request,
                "tools/tool-user-row.html",
                {
                    "user_tool": user_tool,
                    "tool": user_tool.tool,
                    "editing": False,
                },
            )

        return redirect("tool-settings", slug=slug)

    errors = form.errors

    # Invalid Form
    return render(
        request,
        "tools/users/expanded-row-form.html",
        {
            "form": form,
            "user_tool": user_tool,
            "tool": user_tool.tool,
            "htmx": is_htmx_request(request),
            "errors": errors,
        },
        status=HTTPStatus.OK,
    )
