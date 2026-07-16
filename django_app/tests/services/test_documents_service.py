import uuid
from unittest import mock
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile, UploadedFile
from django.test import Client, RequestFactory

from redbox_app.redbox_core.models import Chat, File, FileTeamMembership, FileTool, Tool, UserTeamMembership
from redbox_app.redbox_core.services import documents as documents_service

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_get_file_context(client: Client, chat_with_files: Chat):
    # Given
    user = chat_with_files.user
    client.force_login(user)
    factory = RequestFactory()
    request = factory.get("/chats/")
    request.user = user

    # When
    files = documents_service.get_file_context(request)

    # Then
    assert len(files["completed_files"]) > 0
    assert len(files["processing_files"]) == 0


@pytest.mark.django_db(transaction=True)
def test_get_file_context_with_tool(client: Client, chat_with_files: Chat, default_tool: Tool, uploaded_file: File):
    # Given
    user = chat_with_files.user
    client.force_login(user)
    factory = RequestFactory()
    request = factory.get("/chats/")
    request.user = user

    # When
    FileTool.objects.create(file=uploaded_file, tool=default_tool)
    files = documents_service.get_file_context(request, default_tool)

    # Then
    assert len(files["completed_files"]) == 0
    assert len(files["processing_files"]) == 1


@pytest.mark.django_db
def test_process_uploads_no_files(alice):
    result = documents_service.process_uploads([], alice)

    assert result.errors == ["No document selected"]
    assert result.files == []


@pytest.mark.django_db
def test_process_uploads_invalid_file_type(alice):
    file = SimpleUploadedFile("test.exe", b"data")
    result = documents_service.process_uploads([file], alice)

    assert any("not supported" in e for e in result.errors)
    assert result.files == []


@pytest.mark.django_db
def test_process_uploads_invalid_file_name(alice, original_file: UploadedFile):
    original_file.name = None
    result = documents_service.process_uploads([original_file], alice)

    assert any("File has no name" in e for e in result.errors)
    assert result.files == []


@pytest.mark.django_db
@patch("redbox_app.redbox_core.services.documents.ingest_file")
def test_process_uploads_success(mock_ingest, alice, original_file: UploadedFile):
    mock_ingest.return_value = ([], original_file)

    result = documents_service.process_uploads([original_file], alice)

    assert result.errors == []
    assert len(result.files) == 1
    assert result.files[0] == original_file


@pytest.mark.django_db
@patch("redbox_app.redbox_core.services.documents.ingest_file")
def test_process_uploads_team_membership_created(
    mock_ingest, alice, redbox_team, original_file: UploadedFile, uploaded_file: File
):
    mock_ingest.return_value = ([], uploaded_file)

    UserTeamMembership.objects.create(
        user=alice,
        team=redbox_team,
        role_type=UserTeamMembership.RoleType.ADMIN,
    )

    result = documents_service.process_uploads(
        [original_file],
        alice,
        team_id=redbox_team.id,
        visibility="TEAM",
    )

    file_team = FileTeamMembership.objects.filter(file=uploaded_file, team=redbox_team).first()

    assert result.errors == []
    assert uploaded_file in result.files
    assert file_team


@pytest.mark.django_db
@patch("redbox_app.redbox_core.services.documents.ingest_file")
def test_process_uploads_team_membership_no_permissions(mock_ingest, alice, redbox_team, original_file: UploadedFile):
    mock_ingest.return_value = ([], original_file)

    result = documents_service.process_uploads(
        [original_file],
        alice,
        team_id=redbox_team.id,
        visibility="TEAM",
    )

    assert original_file in result.files
    assert any("You are not a lead for the selected team" in e for e in result.ingest_errors)


@pytest.mark.django_db
@patch("redbox_app.redbox_core.services.documents.ingest_file")
def test_process_uploads_invalid_team(mock_ingest, alice, original_file: UploadedFile):
    mock_ingest.return_value = ([], original_file)

    result = documents_service.process_uploads(
        [original_file],
        alice,
        team_id=str(uuid.uuid4()),
    )

    assert any("does not exist" in e for e in result.ingest_errors)


@pytest.mark.django_db
@patch("redbox_app.redbox_core.services.documents.ingest_file")
def test_process_uploads_skips_none_file_obj(mock_ingest, alice, original_file: UploadedFile):
    mock_ingest.return_value = (["ingest failed"], None)

    result = documents_service.process_uploads([original_file], alice)

    assert result.files == []
    assert result.ingest_errors == ["ingest failed"]
    assert result.errors == []


@pytest.mark.django_db
@patch("redbox_app.redbox_core.services.documents.async_task")
def test_reingest_file_resets_file_during_reingest(uploaded_file: File):
    uploaded_file.status = File.Status.complete
    uploaded_file.save()
    documents_service.reingest_file(uploaded_file)
    assert uploaded_file.status is File.Status.processing


@pytest.mark.django_db
@patch("redbox_app.redbox_core.services.documents.async_task")
def test_reingest_file_calls_async_task_with_expected_args(mock_async_task, uploaded_file: File):
    documents_service.reingest_file(uploaded_file)
    mock_async_task.assert_has_calls(
        [mock.call(mock.ANY, uploaded_file.id, task_name=uploaded_file.unique_name, group="re-ingest")],
    )
