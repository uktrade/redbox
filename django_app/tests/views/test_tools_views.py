import logging
from http import HTTPStatus
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.http import HttpResponse
from django.test import Client
from django.urls import reverse

from redbox_app.redbox_core.models import (
    Chat,
    Tool,
)

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_user_can_see_tools(alice: User, client: Client, default_tool: Tool):
    # Given
    client.force_login(alice)

    # When
    response = client.get(reverse("tools"))

    # Then
    assert response.status_code == HTTPStatus.OK
    assert default_tool.name in response.content.decode()


@pytest.mark.django_db
def test_user_can_see_active_tool(alice: User, client: Client, default_tool: Tool):
    # Given
    client.force_login(alice)

    # When
    response = client.get(reverse("chats", kwargs={"slug": default_tool.slug}))

    # Then
    assert response.status_code == HTTPStatus.OK
    assert default_tool.name in response.content.decode()


@pytest.mark.django_db
def test_tool_info_page_exists(alice: User, client: Client, default_tool: Tool):
    # Given
    client.force_login(alice)
    expected_template_path = f"tools/info/{default_tool.slug}.html"
    # When
    with (
        patch("redbox_app.redbox_core.models.get_template") as mock_get_template,
        patch("redbox_app.redbox_core.views.tools_views.render") as mock_render,
    ):
        mock_get_template.return_value = True
        mock_render.return_value = HttpResponse(f"mocked {default_tool.name}")
        response = client.get(reverse("tool-info", kwargs={"slug": default_tool.slug}))

    # Then
    assert response.status_code == HTTPStatus.OK
    mock_render.assert_called_once()
    mock_get_template.assert_called_once_with(expected_template_path)
    assert default_tool.name in response.content.decode()


@pytest.mark.django_db
def test_tool_info_page_not_found(alice: User, client: Client, default_tool: Tool):
    # Given
    client.force_login(alice)

    # When
    response = client.get(reverse("tool-info", kwargs={"slug": default_tool.slug}))

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.django_db
def test_user_can_see_tool_chats(alice: User, client: Client, default_tool: Tool, chat: Chat):
    # Given
    client.force_login(alice)
    chat.tool = default_tool
    chat.save()

    # When
    url = reverse("chats", kwargs={"slug": default_tool.slug, "chat_id": chat.id})
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.OK
    assert default_tool.name in response.content.decode()
    assert chat.name in response.content.decode()


@pytest.mark.django_db
def test_user_cannot_see_other_user_tool_chats(bob: User, client: Client, default_tool: Tool, chat_with_alice: Chat):
    # Given
    client.force_login(bob)
    url = reverse("chats", kwargs={"slug": default_tool.slug, "chat_id": chat_with_alice.id})

    # When
    response = client.get(url, follow=True)

    # Then
    assert response.status_code == HTTPStatus.OK
    assert default_tool.name not in response.content.decode()
    assert chat_with_alice.name not in response.content.decode()


@pytest.mark.django_db
def test_deselect_document_on_load_tool_setting(alice: User, client: Client, default_tool: Tool, chat_with_files: Chat):
    # Given
    client.force_login(alice)
    chat_with_files.tool = default_tool
    chat_with_files.save()
    settings = default_tool.settings

    # When
    settings.deselect_documents_on_load = True
    settings.save()
    initial_selected_files_count = chat_with_files.last_user_message.selected_files.count()
    url = reverse("chats", kwargs={"slug": default_tool.slug, "chat_id": chat_with_files.id})
    response = client.get(url)
    chat_with_files.last_user_message.refresh_from_db()

    # Then
    assert response.status_code == HTTPStatus.OK
    assert initial_selected_files_count > 0
    assert chat_with_files.last_user_message.selected_files.count() == 0
