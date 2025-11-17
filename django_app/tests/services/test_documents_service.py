import pytest
from django.contrib.auth import get_user_model
from django.test import Client, RequestFactory

from redbox_app.redbox_core.models import Chat, File, FileSkill, Skill
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
def test_get_file_context_with_skill(client: Client, chat_with_files: Chat, default_skill: Skill, uploaded_file: File):
    # Given
    user = chat_with_files.user
    client.force_login(user)
    factory = RequestFactory()
    request = factory.get("/chats/")
    request.user = user

    # When
    FileSkill.objects.create(file=uploaded_file, skill=default_skill)
    files = documents_service.get_file_context(request, default_skill)

    # Then
    assert len(files["completed_files"]) == 0
    assert len(files["processing_files"]) == 1
