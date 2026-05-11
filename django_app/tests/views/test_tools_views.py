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
    ToolAccessRule,
    UserTool,
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
    chat_with_alice.tool = default_tool
    client.force_login(bob)
    url = reverse("chats", kwargs={"slug": default_tool.slug, "chat_id": chat_with_alice.id})

    # When
    response = client.get(url, follow=True)

    # Then
    assert response.status_code == HTTPStatus.OK
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


@pytest.mark.django_db
def test_tool_settings_view(
    client: Client,
    alice: User,
    default_tool: Tool,
):
    client.force_login(alice)

    UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.MANAGER,
    )

    response = client.get(reverse("tool-settings", kwargs={"slug": default_tool.slug}))

    assert response.status_code == HTTPStatus.OK
    assert default_tool.name in response.content.decode()


@pytest.mark.django_db
def test_create_tool_access_rule(client: Client, alice: User, default_tool: Tool):
    client.force_login(alice)

    UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.MANAGER,
    )

    response = client.post(
        reverse(
            "add-tool-access-rule",
            kwargs={"slug": default_tool.slug},
        ),
        {
            "rule_type": ToolAccessRule.RuleType.DOMAIN,
            "value": "example.com",
            "access_type": ToolAccessRule.AccessType.ALLOW,
        },
    )

    assert response.status_code == HTTPStatus.FOUND

    assert ToolAccessRule.objects.filter(
        tool=default_tool,
        value="example.com",
    ).exists()


@pytest.mark.django_db
def test_update_tool_access_rule(client: Client, alice: User, default_tool: Tool):
    client.force_login(alice)

    UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.MANAGER,
    )

    rule = ToolAccessRule.objects.create(
        tool=default_tool,
        rule_type=ToolAccessRule.RuleType.DOMAIN,
        value="old.com",
        access_type=ToolAccessRule.AccessType.ALLOW,
    )

    response = client.post(
        reverse(
            "edit-tool-access-rule",
            kwargs={
                "slug": default_tool.slug,
                "rule_id": rule.id,
            },
        ),
        {
            "rule_type": ToolAccessRule.RuleType.DOMAIN,
            "value": "new.com",
            "access_type": ToolAccessRule.AccessType.DENY,
        },
    )

    rule.refresh_from_db()

    assert response.status_code == HTTPStatus.FOUND
    assert rule.value == "new.com"
    assert rule.access_type == ToolAccessRule.AccessType.DENY


@pytest.mark.django_db
def test_delete_tool_access_rule(client: Client, alice: User, default_tool: Tool):
    client.force_login(alice)

    UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.MANAGER,
    )

    rule = ToolAccessRule.objects.create(
        tool=default_tool,
        rule_type=ToolAccessRule.RuleType.DOMAIN,
        value="example.com",
        access_type=ToolAccessRule.AccessType.ALLOW,
    )

    response = client.post(
        reverse(
            "delete-tool-access-rule",
            kwargs={
                "slug": default_tool.slug,
                "rule_id": rule.id,
            },
        ),
    )

    assert response.status_code == HTTPStatus.FOUND

    assert not ToolAccessRule.objects.filter(id=rule.id).exists()


@pytest.mark.django_db
def test_delete_tool_access_rule_htmx(client: Client, alice: User, default_tool: Tool):
    client.force_login(alice)

    UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.MANAGER,
    )

    rule = ToolAccessRule.objects.create(
        tool=default_tool,
        rule_type=ToolAccessRule.RuleType.DOMAIN,
        value="example.com",
        access_type=ToolAccessRule.AccessType.ALLOW,
    )

    response = client.post(
        reverse(
            "delete-tool-access-rule",
            kwargs={
                "slug": default_tool.slug,
                "rule_id": rule.id,
            },
        ),
        HTTP_HX_REQUEST="true",
    )

    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_bulk_add_user_tool(client: Client, alice: User, bob: User, default_tool: Tool):
    client.force_login(alice)

    UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.MANAGER,
    )

    response = client.post(
        reverse(
            "bulk-add-user-tool",
            kwargs={"slug": default_tool.slug},
        ),
        {
            "user_ids": [bob.id],
            "role": UserTool.RoleType.USER,
            "access_type": UserTool.AccessType.ALLOW,
        },
    )

    assert response.status_code == HTTPStatus.FOUND

    assert UserTool.objects.filter(
        user=bob,
        tool=default_tool,
        access_type=UserTool.AccessType.ALLOW,
        role=UserTool.RoleType.USER,
    ).exists()


@pytest.mark.django_db
def test_tool_access_rule_preview(client: Client, alice: User, bob: User, default_tool: Tool, sso_factory):
    client.force_login(alice)

    sso_factory(
        bob,
        related_emails=["bob@example.com"],
    )

    response = client.post(
        reverse(
            "tool-access-rule-preview",
            kwargs={"slug": default_tool.slug},
        ),
        {
            "rule_type": ToolAccessRule.RuleType.DOMAIN,
            "value": "example.com",
            "access_type": ToolAccessRule.AccessType.ALLOW,
        },
    )

    assert response.status_code == HTTPStatus.OK
    assert bob.email in response.content.decode()


@pytest.mark.django_db
def test_tool_access_rule_value_input_view(client: Client, alice: User):
    client.force_login(alice)

    response = client.get(
        reverse("tool-access-rule-value-input"),
        {
            "rule_type": ToolAccessRule.RuleType.DOMAIN,
        },
    )

    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_edit_tool_user_row_view(client: Client, alice: User, default_tool: Tool):
    client.force_login(alice)

    user_tool = UserTool.objects.create(
        user=alice,
        tool=default_tool,
    )

    response = client.get(
        reverse(
            "edit-user-tool-row",
            kwargs={
                "slug": default_tool.slug,
                "user_tool_id": user_tool.id,
            },
        )
    )

    assert response.status_code == HTTPStatus.OK
    assert alice.email in response.content.decode()


@pytest.mark.django_db
def test_edit_tool_user_row_view_cancel(client: Client, alice: User, default_tool: Tool):
    client.force_login(alice)

    user_tool = UserTool.objects.create(
        user=alice,
        tool=default_tool,
    )

    response = client.get(
        reverse(
            "edit-user-tool-row",
            kwargs={
                "slug": default_tool.slug,
                "user_tool_id": user_tool.id,
            },
        ),
        {
            "cancel": "true",
        },
    )

    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_delete_tool_user_row(client: Client, alice: User, bob: User, default_tool: Tool):
    client.force_login(alice)

    UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.MANAGER,
    )

    user_tool = UserTool.objects.create(
        user=bob,
        tool=default_tool,
    )

    response = client.delete(
        reverse(
            "delete-user-tool",
            kwargs={
                "slug": default_tool.slug,
                "user_tool_id": user_tool.id,
            },
        )
    )

    assert response.status_code == HTTPStatus.OK

    assert not UserTool.objects.filter(id=user_tool.id).exists()


@pytest.mark.django_db
def test_delete_tool_user_row_forbidden(client: Client, alice: User, bob: User, default_tool: Tool):
    client.force_login(alice)

    user_tool = UserTool.objects.create(
        user=bob,
        tool=default_tool,
    )

    response = client.delete(
        reverse(
            "delete-user-tool",
            kwargs={
                "slug": default_tool.slug,
                "user_tool_id": user_tool.id,
            },
        )
    )

    assert response.status_code == HTTPStatus.FORBIDDEN


@pytest.mark.django_db
def test_edit_tool_user_view_htmx(client: Client, alice: User, default_tool: Tool):
    client.force_login(alice)

    user_tool = UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.USER,
    )

    response = client.post(
        reverse(
            "edit-user-tool",
            kwargs={
                "slug": default_tool.slug,
                "user_tool_id": user_tool.id,
            },
        ),
        {
            "role": UserTool.RoleType.MANAGER,
            "access_type": UserTool.AccessType.ALLOW,
        },
        HTTP_HX_REQUEST="true",
    )

    user_tool.refresh_from_db()

    assert response.status_code == HTTPStatus.OK
    assert user_tool.role == UserTool.RoleType.MANAGER


@pytest.mark.django_db
def test_edit_tool_user_view_invalid_form(client: Client, alice: User, default_tool: Tool):
    client.force_login(alice)

    user_tool = UserTool.objects.create(
        user=alice,
        tool=default_tool,
    )

    response = client.post(
        reverse(
            "edit-user-tool",
            kwargs={
                "slug": default_tool.slug,
                "user_tool_id": user_tool.id,
            },
        ),
        {
            "role": "",
            "access_type": "",
        },
        HTTP_HX_REQUEST="true",
    )

    assert response.status_code == HTTPStatus.OK
    assert "govuk-error-summary" in response.content.decode()
