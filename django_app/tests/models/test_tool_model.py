from collections.abc import Sequence
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.template import TemplateDoesNotExist
from django.test import Client

from redbox_app.redbox_core.models import (
    File,
    FileTool,
    Tool,
)

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_auto_slugify(client: Client, alice: User):
    # Given
    client.force_login(alice)

    # When
    tool = Tool.objects.create(name="Test Tool")

    # Then
    assert tool.slug == "test-tool"


@pytest.mark.django_db(transaction=True)
def test_info_template_exists(client: Client, alice: User, default_tool: Tool):
    # Given
    client.force_login(alice)

    # When
    with patch("redbox_app.redbox_core.models.get_template") as mock_get_template:
        mock_get_template.return_value = True

        # Then
        assert default_tool.info_template == f"tools/info/{default_tool.slug}.html"


@pytest.mark.django_db(transaction=True)
def test_info_template_not_found(client: Client, alice: User, default_tool: Tool):
    # Given
    client.force_login(alice)
    expected_path = f"tools/info/{default_tool.slug}.html"

    # When
    with patch("redbox_app.redbox_core.models.get_template") as mock_get_template:
        mock_get_template.side_effect = TemplateDoesNotExist(expected_path)

        # Then
        assert default_tool.info_template is None


@pytest.mark.django_db(transaction=True)
def test_has_info_page_true(client: Client, alice: User, default_tool: Tool):
    # Given
    client.force_login(alice)

    # When
    with patch("redbox_app.redbox_core.models.get_template") as mock_get_template:
        mock_get_template.return_value = True

        # Then
        assert default_tool.has_info_page is True


@pytest.mark.django_db(transaction=True)
def test_has_info_page_false(client: Client, alice: User, default_tool: Tool):
    # Given
    client.force_login(alice)

    # When
    with patch("redbox_app.redbox_core.models.get_template") as mock_get_template:
        mock_get_template.side_effect = TemplateDoesNotExist("tools/info/test-tool.html")

        # Then
        assert default_tool.has_info_page is False


@pytest.mark.django_db(transaction=True)
def test_get_info_page_url(client: Client, alice: User, default_tool: Tool):
    # Given
    client.force_login(alice)

    # When
    url = default_tool.info_page_url

    # Then
    assert url == f"/tools/{default_tool.slug}/"


@pytest.mark.django_db(transaction=True)
def test_get_files(client: Client, alice: User, default_tool: Tool, several_files: Sequence[File]):
    # Given
    client.force_login(alice)

    # When
    for file in several_files:
        file_tool = FileTool(file=file, tool=default_tool)
        file_tool.save()

    # Then
    assert len(default_tool.get_files()) == len(several_files)


@pytest.mark.django_db(transaction=True)
def test_get_settings(client: Client, alice: User, default_tool: Tool):
    # Given
    client.force_login(alice)

    # When
    settings = default_tool.settings

    # Then
    assert settings.__str__() == "Default Tool Settings"
    assert settings.tool == default_tool
    assert settings.deselect_documents_on_load is False
