from collections.abc import Sequence
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.template import TemplateDoesNotExist
from django.test import Client

from redbox_app.redbox_core.models import (
    File,
    FileSkill,
    Skill,
)

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_auto_slugify(client: Client, alice: User):
    # Given
    client.force_login(alice)

    # When
    skill = Skill.objects.create(name="Test Skill")

    # Then
    assert skill.slug == "test-skill"


@pytest.mark.django_db(transaction=True)
def test_info_template_exists(client: Client, alice: User, default_skill: Skill):
    # Given
    client.force_login(alice)

    # When
    with patch("redbox_app.redbox_core.models.get_template") as mock_get_template:
        mock_get_template.return_value = True

        # Then
        assert default_skill.info_template == f"skills/info/{default_skill.slug}.html"


@pytest.mark.django_db(transaction=True)
def test_info_template_not_found(client: Client, alice: User, default_skill: Skill):
    # Given
    client.force_login(alice)
    expected_path = f"skills/info/{default_skill.slug}.html"

    # When
    with patch("redbox_app.redbox_core.models.get_template") as mock_get_template:
        mock_get_template.side_effect = TemplateDoesNotExist(expected_path)

        # Then
        assert default_skill.info_template is None


@pytest.mark.django_db(transaction=True)
def test_has_info_page_true(client: Client, alice: User, default_skill: Skill):
    # Given
    client.force_login(alice)

    # When
    with patch("redbox_app.redbox_core.models.get_template") as mock_get_template:
        mock_get_template.return_value = True

        # Then
        assert default_skill.has_info_page is True


@pytest.mark.django_db(transaction=True)
def test_has_info_page_false(client: Client, alice: User, default_skill: Skill):
    # Given
    client.force_login(alice)

    # When
    with patch("redbox_app.redbox_core.models.get_template") as mock_get_template:
        mock_get_template.side_effect = TemplateDoesNotExist("skills/info/test-skill.html")

        # Then
        assert default_skill.has_info_page is False


@pytest.mark.django_db(transaction=True)
def test_get_info_page_url(client: Client, alice: User, default_skill: Skill):
    # Given
    client.force_login(alice)

    # When
    url = default_skill.get_info_page_url()

    # Then
    assert url == f"/skills/{default_skill.slug}/"


@pytest.mark.django_db(transaction=True)
def test_get_files(client: Client, alice: User, default_skill: Skill, several_files: Sequence[File]):
    # Given
    client.force_login(alice)

    # When
    for file in several_files:
        file_skill = FileSkill(file=file, skill=default_skill)
        file_skill.save()

    # Then
    assert len(default_skill.get_files()) == len(several_files)
