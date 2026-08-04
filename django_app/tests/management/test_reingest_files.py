from datetime import datetime, timedelta
from unittest import mock
from unittest.mock import patch

import pytest
import pytz
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from freezegun import freeze_time

from redbox_app.redbox_core.models import File, FileTool, Tool

User = get_user_model()


@pytest.mark.django_db
class TestMigrateToMockSSO:
    def test_reingest_files_throws_expected_error_when_no_args_are_provided(self):
        with pytest.raises(CommandError):
            call_command("reingest_files")

    def test_reingest_files_throws_expected_error_when_only_start_date_is_provided(self):
        with pytest.raises(CommandError):
            call_command("reingest_files", "2026-06-06")

    def test_reingest_files_throws_expected_error_when_start_date_is_an_invalid_format(self):
        with pytest.raises(CommandError):
            call_command("reingest_files", "200-06-06")

    def test_reingest_files_throws_expected_error_when_end_date_is_an_invalid_format(self):
        with pytest.raises(CommandError):
            call_command("reingest_files", "2026-06-06", "abc")

    @pytest.mark.parametrize("file_status", [File.Status.deleted, File.Status.errored, File.Status.processing])
    def test_reingest_files_excludes_file_with_invalid_status_from_async_task(self, alice: User, file_status):
        test_date = datetime(2026, 6, 4, 0, 0, 0, tzinfo=pytz.UTC)

        with (
            patch("redbox_app.redbox_core.management.commands.reingest_files.reingest_file") as mock_reingest_file,
            freeze_time(test_date),
        ):
            complete_file = File.objects.create(
                status=File.Status.complete,
                original_file="complete.json",
                user=alice,
            )
            File.objects.create(
                status=file_status,
                original_file="invalid.json",
                user=alice,
            )
            call_command("reingest_files", (test_date.strftime("%Y-%m-%d")), (test_date.strftime("%Y-%m-%d")))

            mock_reingest_file.assert_has_calls([mock.call(complete_file)])

    def test_reingest_files_excludes_tool_files_from_admin(self, alice: User):
        with patch("redbox_app.redbox_core.management.commands.reingest_files.reingest_file") as mock_reingest_file:
            test_date = datetime(2026, 6, 4, 0, 0, 0, tzinfo=pytz.UTC)
            with freeze_time(test_date + timedelta(hours=10)):
                included_file_1 = File.objects.create(
                    status=File.Status.complete,
                    original_file="complete_1.json",
                    user=alice,
                )

                tool = Tool.objects.create(name="test tool")
                member_file = File.objects.create(
                    status=File.Status.complete,
                    original_file="member_tool.json",
                    user=alice,
                )
                FileTool.objects.create(file=member_file, tool=tool, file_type=FileTool.FileType.MEMBER)

                admin_file = File.objects.create(
                    status=File.Status.complete,
                    original_file="admin_tool.json",
                    user=alice,
                )
                FileTool.objects.create(file=admin_file, tool=tool, file_type=FileTool.FileType.ADMIN)

                call_command("reingest_files", "2026-06-04", "2026-06-05")

            mock_reingest_file.assert_has_calls(
                [
                    mock.call(included_file_1),
                    mock.call(member_file),
                ],
            )
            assert mock_reingest_file.call_count == 2

    def test_reingest_files_calls_async_task_with_only_files_inside_date_range(self, alice: User):
        with patch("redbox_app.redbox_core.management.commands.reingest_files.reingest_file") as mock_reingest_file:
            test_date = datetime(2026, 6, 4, 0, 0, 0, tzinfo=pytz.UTC)
            with freeze_time(test_date + timedelta(hours=10)):
                included_file_1 = File.objects.create(
                    status=File.Status.complete,
                    original_file="complete_1.json",
                    user=alice,
                )
            with freeze_time(test_date):
                included_file_2 = File.objects.create(
                    status=File.Status.complete,
                    original_file="complete_2.json",
                    user=alice,
                )
            with freeze_time(test_date + timedelta(hours=48) + timedelta(seconds=-1)):
                included_file_3 = File.objects.create(
                    status=File.Status.complete,
                    original_file="complete_3.json",
                    user=alice,
                )
            with freeze_time(test_date + timedelta(hours=24)):
                included_file_4 = File.objects.create(
                    status=File.Status.complete,
                    original_file="complete_4.json",
                    user=alice,
                )
            with freeze_time(test_date + timedelta(seconds=-1)):
                File.objects.create(
                    status=File.Status.complete,
                    original_file="excluded_1.json",
                    user=alice,
                )
            with freeze_time(test_date + timedelta(hours=48)):
                File.objects.create(
                    status=File.Status.complete,
                    original_file="excluded_2.json",
                    user=alice,
                )

            call_command("reingest_files", "2026-06-04", "2026-06-05")

            mock_reingest_file.assert_has_calls(
                [
                    mock.call(included_file_1),
                    mock.call(included_file_2),
                    mock.call(included_file_3),
                    mock.call(included_file_4),
                ],
            )
            assert mock_reingest_file.call_count == 4
