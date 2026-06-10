from http import HTTPStatus

from django.http import HttpResponse
from django.shortcuts import get_object_or_404

from redbox_app.redbox_core.models import Tool
from redbox_app.redbox_core.services import chats as chat_service


class ChatContextMixin:
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context.update(chat_service.get_context(self.request))

        return context


class AppContextMixin:
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context.update(chat_service.get_context(self.request, **self.kwargs))

        return context


class ToolUserRequiredMixin:
    def get_permission_tool(self):
        if hasattr(self, "object") and self.object:
            return self.object

        return get_object_or_404(
            Tool,
            slug=self.kwargs["slug"],
        )

    def dispatch(self, request, *args, **kwargs):
        tool = self.get_permission_tool()

        if not request.user.has_tool_access(tool):
            return HttpResponse(
                "You do not have access to this tool.",
                status=HTTPStatus.FORBIDDEN,
            )

        return super().dispatch(request, *args, **kwargs)


class ToolManagerRequiredMixin:
    def get_permission_tool(self):
        if hasattr(self, "object") and self.object:
            return self.object

        return get_object_or_404(
            Tool,
            slug=self.kwargs["slug"],
        )

    def dispatch(self, request, *args, **kwargs):
        tool = self.get_permission_tool()

        if not request.user.can_manage_tool(tool):
            return HttpResponse(
                "You do not have permission to manage this tool.",
                status=HTTPStatus.FORBIDDEN,
            )

        return super().dispatch(request, *args, **kwargs)
