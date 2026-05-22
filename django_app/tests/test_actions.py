import logging
from types import SimpleNamespace

import pytest
from django.contrib.auth import get_user_model
from django.test import Client, RequestFactory

from redbox_app.redbox_core.actions import backfill_original_file_names
from redbox_app.redbox_core.models import File

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_backfill_original_file_names_action(client: Client, alice: User):
    # Given
    client.force_login(alice)

    file = File.objects.create(
        user=alice,
        original_file="alice/test.pdf",
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
