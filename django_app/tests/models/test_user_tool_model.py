import pytest
from django.contrib.auth import get_user_model
from django.db import IntegrityError

from redbox_app.redbox_core.models import (
    Tool,
    UserTool,
)

User = get_user_model()


@pytest.mark.django_db
def test_user_tool_str(alice, default_tool):
    user_tool = UserTool.objects.create(
        user=alice,
        tool=default_tool,
    )

    assert str(user_tool) == f"{alice.email} - {default_tool.name}"


@pytest.mark.django_db
def test_user_tool_defaults(alice, default_tool):
    user_tool = UserTool.objects.create(
        user=alice,
        tool=default_tool,
    )

    assert user_tool.access_type == UserTool.AccessType.ALLOW
    assert user_tool.role == UserTool.RoleType.USER


@pytest.mark.django_db
def test_is_enabled_public_tool(alice, default_tool):
    default_tool.is_public = True
    default_tool.save()

    user_tool = UserTool.objects.create(
        user=alice,
        tool=default_tool,
        access_type=UserTool.AccessType.DENY,
    )

    assert user_tool.is_enabled is True


@pytest.mark.django_db
def test_is_enabled_private_allow(alice, default_tool):
    default_tool.is_public = False
    default_tool.save()

    user_tool = UserTool.objects.create(
        user=alice,
        tool=default_tool,
        access_type=UserTool.AccessType.ALLOW,
    )

    assert user_tool.is_enabled is True


@pytest.mark.django_db
def test_is_enabled_private_deny(alice, default_tool):
    default_tool.is_public = False
    default_tool.save()

    user_tool = UserTool.objects.create(
        user=alice,
        tool=default_tool,
        access_type=UserTool.AccessType.DENY,
    )

    assert user_tool.is_enabled is False


@pytest.mark.django_db
def test_role_choices(alice, default_tool):
    user_tool = UserTool.objects.create(
        user=alice,
        tool=default_tool,
    )

    assert user_tool.role_choices == UserTool.RoleType.choices


@pytest.mark.django_db
def test_user_tool_unique_constraint(alice, default_tool):
    UserTool.objects.create(
        user=alice,
        tool=default_tool,
    )

    with pytest.raises(IntegrityError):
        UserTool.objects.create(
            user=alice,
            tool=default_tool,
        )


@pytest.mark.django_db
def test_user_tool_ordering(alice, bob, default_tool):
    first = UserTool.objects.create(
        user=alice,
        tool=default_tool,
    )

    second_tool = Tool.objects.create(name="Second Tool")

    second = UserTool.objects.create(
        user=bob,
        tool=second_tool,
    )

    user_tools = list(UserTool.objects.all())

    assert user_tools == [first, second]


@pytest.mark.django_db
def test_user_tool_manager_role(alice, default_tool):
    user_tool = UserTool.objects.create(
        user=alice,
        tool=default_tool,
        role=UserTool.RoleType.MANAGER,
    )

    assert user_tool.role == UserTool.RoleType.MANAGER
