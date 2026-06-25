from unittest.mock import patch

import pytest
from django.core.management import call_command

from redbox_app.redbox_core.management.commands.local_db_init import NotLocalError
from redbox_app.redbox_core.models import Agent, Tool, ToolSettings
from redbox_app.setting_enums import Environment


@pytest.fixture(autouse=True)
def mock_local_env():
    with patch.object(Environment, "is_local", new=True):
        yield


@pytest.fixture(autouse=True)
def mock_migrations():
    "Don't run the migrations at the start of the command to reduce test time"

    with patch("django.core.management.call_command") as mock:

        def side_effect(command, *args, **kwargs) -> str | None:
            if command == "loaddata":
                return call_command(command, *args, **kwargs)

            return None

        mock.side_effect = side_effect
        yield mock


@pytest.mark.django_db
class TestInitialiseDbLoadsData:
    def test_tools_are_created(self):
        call_command("local_db_init")
        assert Tool.objects.count() > 0
        assert ToolSettings.objects.count() > 0
        assert Agent.objects.count() > 0

    def test_running_twice_does_not_duplicate_tools(self):
        call_command("local_db_init")
        initial_count = Tool.objects.count()
        call_command("local_db_init")
        second_count = Tool.objects.count()

        assert initial_count == second_count

    def test_raises_if_not_local_environment(self):
        with patch.object(Environment, "is_local", new=False), pytest.raises(NotLocalError):
            call_command("local_db_init")
