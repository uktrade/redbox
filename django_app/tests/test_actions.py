import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.test import Client, RequestFactory

from redbox_app.redbox_core.actions import backfill_original_file_names, reupload
from redbox_app.redbox_core.models import File
from redbox_app.worker import ingest

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_backfill_original_file_names_action(client: Client, alice: User):
    # Given
    client.force_login(alice)
    file_name = "test.pdf"

    file = File.objects.create(
        user=alice,
        original_file=f"alice/{file_name}",
        original_file_name="",
    )
    file.original_file_name = ""

    file.save()
    file.refresh_from_db()
    assert file.original_file_name == ""

    # When
    mock_admin = SimpleNamespace(message_user=lambda *_args, **_kwargs: None)
    request = RequestFactory().get("/admin/")
    queryset = File.objects.filter(id=file.id)

    backfill_original_file_names(mock_admin, request, queryset)

    # Then
    file.refresh_from_db()
    assert file.original_file_name == "test.pdf"


@pytest.mark.django_db
def test_reupload_action_triggers_async_task(client, alice):
    # Given
    client.force_login(alice)

    file1 = File.objects.create(
        user=alice,
        original_file="alice/test1.pdf",
    )
    file2 = File.objects.create(
        user=alice,
        original_file="alice/test2.pdf",
    )

    queryset = File.objects.filter(id__in=[file1.id, file2.id])

    mock_admin = SimpleNamespace(message_user=lambda *_args, **_kwargs: None)
    request = RequestFactory().get("/admin/")

    # When
    with patch("redbox_app.redbox_core.actions.async_task") as mock_async_task:
        reupload(mock_admin, request, queryset)

    # Then
    assert mock_async_task.call_count == 2

    mock_async_task.assert_any_call(ingest, file1.id)
    mock_async_task.assert_any_call(ingest, file2.id)
