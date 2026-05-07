import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import (
    Chat,
    Tool,
    ToolAccessRule,
    UserTool,
)

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_first_time_user(client: Client, bob: User, chat_with_alice: Chat):
    # Given
    client.force_login(chat_with_alice.user)

    # When
    response_1 = chat_with_alice.user.first_time_user
    response_2 = bob.first_time_user

    # Then
    assert response_1 is False
    assert response_2 is True


@pytest.mark.django_db
def test_has_tool_access_public_tool(alice: User):
    tool = Tool.objects.create(
        name="Public Tool",
        is_public=True,
    )

    assert alice.has_tool_access(tool) is True


@pytest.mark.django_db
def test_has_tool_access_explicit_allow(
    alice: User,
    default_tool: Tool,
):
    default_tool.is_public = False
    default_tool.save()

    UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.USER,
        access_type=UserTool.AccessType.ALLOW,
    )

    assert alice.has_tool_access(default_tool) is True


@pytest.mark.django_db
def test_has_tool_access_deny_rule(
    alice: User,
    default_tool: Tool,
    sso_factory,
):
    default_tool.is_public = True
    default_tool.save()

    sso_factory(
        alice,
        related_emails=["alice@example.com"],
    )

    ToolAccessRule.objects.create(
        tool=default_tool,
        rule_type=ToolAccessRule.RuleType.DOMAIN,
        value="example.com",
        access_type=ToolAccessRule.AccessType.DENY,
    )

    assert alice.has_tool_access(default_tool) is False


@pytest.mark.django_db
def test_has_tool_access_no_access(
    alice: User,
    default_tool: Tool,
):
    default_tool.is_public = False
    default_tool.save()

    assert alice.has_tool_access(default_tool) is False


@pytest.mark.django_db
def test_can_manage_tool_true(
    alice: User,
    default_tool: Tool,
):
    UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.MANAGER,
        access_type=UserTool.AccessType.ALLOW,
    )

    assert alice.can_manage_tool(default_tool) is True


@pytest.mark.django_db
def test_can_manage_tool_false(
    alice: User,
    default_tool: Tool,
):
    UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.USER,
        access_type=UserTool.AccessType.ALLOW,
    )

    assert alice.can_manage_tool(default_tool) is False


@pytest.mark.django_db
def test_display_name_uses_name(alice: User):
    alice.name = "Alice Smith"

    assert alice.display_name == "Alice Smith"


@pytest.mark.django_db
def test_display_name_uses_sso_name(
    alice: User,
    sso_factory,
):
    alice.name = ""

    sso_factory(
        alice,
        first_name="Alice",
        last_name="SSO",
    )

    assert alice.display_name == "Alice SSO"


@pytest.mark.django_db
def test_display_name_uses_email(alice: User):
    alice.name = ""

    assert alice.display_name == alice.email


@pytest.mark.django_db
def test_display_name_uses_username(alice: User):
    alice.name = ""
    alice.email = ""

    assert alice.display_name == alice.username


@pytest.mark.django_db
def test_sso_property_returns_sso(
    alice: User,
    sso_factory,
):
    sso = sso_factory(alice)

    assert alice.sso == sso


@pytest.mark.django_db
def test_sso_property_returns_none(alice: User):
    assert alice.sso is None


@pytest.mark.django_db
def test_all_emails(
    alice: User,
    sso_factory,
):
    sso_factory(
        alice,
        related_emails=[
            "related@example.com",
        ],
        contact_email="contact@example.com",
    )

    emails = alice.all_emails

    assert alice.email in emails
    assert "related@example.com" in emails
    assert "contact@example.com" in emails


@pytest.mark.django_db
def test_all_emails_without_sso(alice: User):
    emails = alice.all_emails

    assert emails == {alice.email}
