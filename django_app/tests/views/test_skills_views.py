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
    Skill,
)

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_user_can_see_skills(alice: User, client: Client, default_skill: Skill):
    # Given
    client.force_login(alice)

    # When
    response = client.get(reverse("tools"))

    # Then
    assert response.status_code == HTTPStatus.OK
    assert default_skill.name in response.content.decode()


@pytest.mark.django_db
def test_user_can_see_active_skill(alice: User, client: Client, default_skill: Skill):
    # Given
    client.force_login(alice)

    # When
    response = client.get(reverse("chats", kwargs={"slug": default_skill.slug}))

    # Then
    assert response.status_code == HTTPStatus.OK
    assert default_skill.name in response.content.decode()


@pytest.mark.django_db
def test_skill_info_page_exists(alice: User, client: Client, default_skill: Skill):
    # Given
    client.force_login(alice)
    expected_template_path = f"skills/info/{default_skill.slug}.html"
    # When
    with (
        patch("redbox_app.redbox_core.models.get_template") as mock_get_template,
        patch("redbox_app.redbox_core.views.skills_views.render") as mock_render,
    ):
        mock_get_template.return_value = True
        mock_render.return_value = HttpResponse(f"mocked {default_skill.name}")
        response = client.get(reverse("tool-info", kwargs={"slug": default_skill.slug}))

    # Then
    assert response.status_code == HTTPStatus.OK
    mock_render.assert_called_once()
    mock_get_template.assert_called_once_with(expected_template_path)
    assert default_skill.name in response.content.decode()


@pytest.mark.django_db
def test_skill_info_page_not_found(alice: User, client: Client, default_skill: Skill):
    # Given
    client.force_login(alice)

    # When
    response = client.get(reverse("tool-info", kwargs={"slug": default_skill.slug}))

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.django_db
def test_user_can_see_skill_chats(alice: User, client: Client, default_skill: Skill, chat: Chat):
    # Given
    client.force_login(alice)
    chat.skill = default_skill
    chat.save()

    # When
    url = reverse("chats", kwargs={"slug": default_skill.slug, "chat_id": chat.id})
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.OK
    assert default_skill.name in response.content.decode()
    assert chat.name in response.content.decode()


@pytest.mark.django_db
def test_user_cannot_see_other_user_skill_chats(bob: User, client: Client, default_skill: Skill, chat_with_alice: Chat):
    # Given
    client.force_login(bob)
    url = reverse("chats", kwargs={"slug": default_skill.slug, "chat_id": chat_with_alice.id})

    # When
    response = client.get(url, follow=True)

    # Then
    assert response.status_code == HTTPStatus.OK
    assert default_skill.name not in response.content.decode()
    assert chat_with_alice.name not in response.content.decode()


@pytest.mark.django_db
def test_deselect_document_on_load_skill_setting(
    alice: User, client: Client, default_skill: Skill, chat_with_files: Chat
):
    # Given
    client.force_login(alice)
    chat_with_files.skill = default_skill
    chat_with_files.save()
    settings = default_skill.settings

    # When
    settings.deselect_documents_on_load = True
    settings.save()
    initial_selected_files_count = chat_with_files.last_user_message.selected_files.count()
    url = reverse("chats", kwargs={"slug": default_skill.slug, "chat_id": chat_with_files.id})
    response = client.get(url)
    chat_with_files.last_user_message.refresh_from_db()

    # Then
    assert response.status_code == HTTPStatus.OK
    assert initial_selected_files_count > 0
    assert chat_with_files.last_user_message.selected_files.count() == 0
