import logging
from http import HTTPStatus

import pytest
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.test import RequestFactory
from django.views import View

from redbox_app.redbox_core.models import Tool, UserTool
from redbox_app.redbox_core.views.mixins import ToolManagerRequiredMixin, ToolUserRequiredMixin

User = get_user_model()

logger = logging.getLogger(__name__)


class DummyToolUserView(ToolUserRequiredMixin, View):
    def get_permission_tool(self):
        return self.tool

    def get(self, request, *args, **kwargs):
        return JsonResponse(
            {
                "request": str(request),
                "args": str(args),
                "kwargs": str(kwargs),
            },
            status=HTTPStatus.OK,
        )


class DummyToolManagerView(ToolManagerRequiredMixin, View):
    def get_permission_tool(self):
        return self.tool

    def get(self, request, *args, **kwargs):
        return JsonResponse(
            {
                "request": str(request),
                "args": str(args),
                "kwargs": str(kwargs),
            },
            status=HTTPStatus.OK,
        )


@pytest.mark.django_db
def test_tool_user_required_mixin_allows_access(alice: User, default_tool: Tool):
    # Given
    default_tool.add_user(alice, role=None, access_type=None)

    request = RequestFactory().get("/")
    request.user = alice

    view = DummyToolUserView()
    view.tool = default_tool

    # When
    response = view.dispatch(request)

    # Then
    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_tool_user_required_mixin_denies_access(
    alice: User,
    default_tool: Tool,
):
    request = RequestFactory().get("/")
    request.user = alice

    default_tool.is_public = False
    default_tool.save()

    view = DummyToolUserView()
    view.tool = default_tool

    response = view.dispatch(request)

    assert response.status_code == HTTPStatus.FORBIDDEN


@pytest.mark.django_db
def test_tool_manager_required_mixin_allows_manager(
    alice: User,
    default_tool: Tool,
):
    default_tool.add_user(
        alice,
        role=UserTool.RoleType.MANAGER,
        access_type=UserTool.AccessType.ALLOW,
    )

    request = RequestFactory().get("/")
    request.user = alice

    view = DummyToolManagerView()
    view.tool = default_tool

    response = view.dispatch(request)

    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_tool_manager_required_mixin_denies_non_manager(
    alice: User,
    default_tool: Tool,
):
    default_tool.add_user(
        alice,
        role=UserTool.RoleType.USER,
        access_type=UserTool.AccessType.ALLOW,
    )

    request = RequestFactory().get("/")
    request.user = alice

    view = DummyToolManagerView()
    view.tool = default_tool

    response = view.dispatch(request)

    assert response.status_code == HTTPStatus.FORBIDDEN
