import logging
from http import HTTPStatus
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.http import HttpResponse
from django.test import Client
from django.urls import reverse

from redbox_app.redbox_core.models import (
    Skill,
)

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_user_can_see_skills(alice: User, client: Client, default_skill: Skill):
    # Given
    client.force_login(alice)

    # When
    response = client.get(reverse("skills"))

    # Then
    assert response.status_code == HTTPStatus.OK
    assert default_skill.name in response.content.decode()


@pytest.mark.django_db
def test_user_can_see_active_skill(alice: User, client: Client, default_skill: Skill):
    # Given
    client.force_login(alice)

    # When
    response = client.get(reverse("skill-chats", kwargs={"skill_slug": default_skill.slug}))

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
        response = client.get(reverse("skill-info", kwargs={"skill_slug": default_skill.slug}))

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
    response = client.get(reverse("skill-info", kwargs={"skill_slug": default_skill.slug}))

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND
