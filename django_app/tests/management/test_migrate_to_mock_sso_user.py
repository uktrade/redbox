from unittest.mock import patch
from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError

from redbox_app.redbox_core.models import Chat, File, UserTeamMembership, UserTool
from redbox_app.setting_enums import Environment
from redbox_app.settings import MOCK_SSO_USERNAME

User = get_user_model()


@pytest.mark.django_db
class TestMigrateToMockSSO:
    def test_migrate_user_throws_expected_error_when_user_id_is_not_provided(self):
        with pytest.raises(CommandError):
            call_command("migrate_to_mock_sso_user")

    def test_migrate_user_throws_expected_error_when_environment_is_not_local(self):
        with patch.object(Environment, "is_local", False), pytest.raises(CommandError) as exc:
            call_command("migrate_to_mock_sso_user", 1)
        assert str(exc.value) == "This command can only be run on a local environment"

    @patch("redbox_app.redbox_core.management.commands.migrate_to_mock_sso_user.MOCK_SSO_USERNAME", new=None)
    def test_migrate_user_throws_expected_error_when_mock_sso_username_is_none(self):
        with pytest.raises(CommandError) as exc:
            call_command("migrate_to_mock_sso_user", 1)
        assert str(exc.value) == "The MOCK_SSO_USERNAME env var is not set"

    def test_migrate_user_throws_expected_error_when_mock_user_cannot_be_found(self):
        with pytest.raises(CommandError) as exc:
            call_command("migrate_to_mock_sso_user", 2)
        assert str(exc.value) == f"User with username '{MOCK_SSO_USERNAME}' not found"

    def test_migrate_user_throws_expected_error_when_user_id_is_unknown(self, mock_sso_user_factory):
        _ = mock_sso_user_factory
        user_id = uuid4()
        with pytest.raises(CommandError) as exc:
            call_command("migrate_to_mock_sso_user", user_id)
        assert str(exc.value) == f"User with id '{user_id}' not found"

    def test_migrate_user_ignores_existing_user_tools(
        self,
        alice: User,
        tool_factory,
        mock_sso_user_factory,
    ):

        for tool in [tool_factory(tool_name) for tool_name in ["tool_1", "tool_3", "tool_4"]]:
            tool.add_user(alice, role=None, access_type=None)

        tool_factory(name="tool_2").add_user(mock_sso_user_factory)

        call_command("migrate_to_mock_sso_user", alice.id)

        for tool_user in UserTool.objects.all():
            assert tool_user.user.username == mock_sso_user_factory.username

    def test_migrate_user_updates_all_linked_objects(
        self,
        alice: User,
        chat_factory,
        tool_factory,
        file_factory,
        mock_sso_user_factory,
        user_team_membership_factory,
    ):

        for tool in [tool_factory(tool_name) for tool_name in ["tool_1", "tool2"]]:
            tool.add_user(alice, role=None, access_type=None)

        [chat_factory(alice, chat) for chat in ["chat_1", "chat_2"]]

        [file_factory(alice) for _ in ["file_1", "file_2"]]

        [user_team_membership_factory(alice) for _ in ["membership"]]

        call_command("migrate_to_mock_sso_user", alice.id)

        for tool_user in UserTool.objects.all():
            assert tool_user.user.username == mock_sso_user_factory.username

        for chat in Chat.objects.all():
            assert chat.user.username == mock_sso_user_factory.username

        for file in File.objects.all():
            assert file.user.username == mock_sso_user_factory.username

        for membership in UserTeamMembership.objects.all():
            assert membership.user.username == mock_sso_user_factory.username
