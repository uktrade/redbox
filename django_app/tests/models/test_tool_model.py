from collections.abc import Sequence
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.template import TemplateDoesNotExist
from django.test import Client
from django.urls import reverse

from redbox_app.redbox_core.models import (
    File,
    FileTool,
    Tool,
    ToolAccessRule,
    UserTool,
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


@pytest.mark.django_db
def test_chat_url(default_tool: Tool):
    assert default_tool.chat_url == reverse("chats", kwargs={"slug": default_tool.slug})


@pytest.mark.django_db
def test_settings_url(default_tool: Tool):
    assert default_tool.settings_url == reverse("tool-settings", kwargs={"slug": default_tool.slug})


@pytest.mark.django_db
def test_is_manager_true(alice: User, default_tool: Tool):
    default_tool.add_user(
        user=alice,
        role=UserTool.RoleType.MANAGER,
        access_type=UserTool.AccessType.ALLOW,
    )

    assert default_tool.is_manager(alice) is True


@pytest.mark.django_db
def test_is_manager_false(alice: User, default_tool: Tool):
    assert default_tool.is_manager(alice) is False


@pytest.mark.django_db
def test_add_user_defaults(alice: User, default_tool: Tool):
    user_tool = default_tool.add_user(user=alice, role=None, access_type=None)

    assert user_tool.user == alice
    assert user_tool.tool == default_tool
    assert user_tool.role == UserTool.RoleType.USER
    assert user_tool.access_type == UserTool.AccessType.ALLOW


@pytest.mark.django_db
def test_get_unassigned_users_excludes_existing(alice: User, bob: User, default_tool: Tool):
    default_tool.add_user(user=alice, role=None, access_type=None)

    users = default_tool.get_unassigned_users()

    assert alice not in users
    assert bob in users


@pytest.mark.django_db
def test_settings_returns_same_instance(default_tool: Tool):
    s1 = default_tool.settings
    s2 = default_tool.settings

    assert s1.id == s2.id


@pytest.mark.django_db
def test_slug_not_overwritten_if_present():
    tool = Tool.objects.create(name="Test Tool", slug="custom-slug")

    assert tool.slug == "custom-slug"


@pytest.mark.django_db
def test_tool_ordering():
    Tool.objects.create(name="B Tool")
    Tool.objects.create(name="A Tool")

    tools = list(Tool.objects.all())

    assert tools[0].name == "A Tool"


@pytest.mark.django_db
def test_for_user_returns_public_tool(alice: User):
    tool = Tool.objects.create(name="Public Tool", is_public=True)

    qs = Tool.objects.for_user(alice)

    assert tool in qs


@pytest.mark.django_db
def test_for_user_explicit_access(alice: User):
    tool = Tool.objects.create(name="Private Tool", is_public=False)

    tool.add_user(
        user=alice,
        role=None,
        access_type=UserTool.AccessType.ALLOW,
    )

    qs = Tool.objects.for_user(alice)

    assert tool in qs


@pytest.mark.django_db
def test_for_user_deny_rule(alice: User, default_tool: Tool, sso_factory):
    sso_factory(alice, related_emails=["alice@example.com"])

    default_tool.is_public = True
    default_tool.save()

    ToolAccessRule.objects.create(
        tool=default_tool,
        rule_type=ToolAccessRule.RuleType.DOMAIN,
        value="example.com",
        access_type=ToolAccessRule.AccessType.DENY,
    )

    qs = Tool.objects.for_user(alice)

    assert default_tool not in qs


@pytest.mark.django_db(transaction=True)
def test_tool_access_rule_str(client: Client, alice: User, default_tool: Tool):
    # Given
    client.force_login(alice)

    # When
    rule = ToolAccessRule.objects.create(
        tool=default_tool,
        rule_type=ToolAccessRule.RuleType.DOMAIN,
        value="example.com",
        access_type=ToolAccessRule.AccessType.ALLOW,
    )

    # Then
    assert rule.__str__() == "Default Tool (DOMAIN) - example.com"
