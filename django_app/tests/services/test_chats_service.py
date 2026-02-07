from collections.abc import Sequence
from http import HTTPStatus

import pytest
from django.contrib.auth import get_user_model
from django.test import Client, RequestFactory

from redbox_app.redbox_core.models import Chat, ChatMessage, File
from redbox_app.redbox_core.services import chats as chats_service

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_get_context(
    client: Client,
    user_with_chats_with_messages_over_time: User,
    several_files: Sequence[File],
    chat_with_files: Chat,
):
    # Given
    alice = user_with_chats_with_messages_over_time
    client.force_login(alice)
    factory = RequestFactory()
    request = factory.get("/chats/")
    request.user = alice

    # When
    context = chats_service.get_context(request)
    chat_context = chats_service.get_context(request, chat_with_files.id)

    # Then
    assert context["chat_id"] is None
    assert context["messages"] == []
    assert len(context["chats"]) == len(Chat.get_ordered_by_last_message_date(alice))
    assert context["current_chat"] is None
    assert len(context["completed_files"]) == len(several_files)

    assert chat_context["chat_id"] == chat_with_files.id
    assert len(chat_context["messages"]) == len(ChatMessage.get_messages_ordered_by_citation_priority(chat_with_files))
    assert len(chat_context["chats"]) == len(Chat.get_ordered_by_last_message_date(alice))
    assert chat_context["current_chat"] == chat_with_files
    assert len(context["completed_files"]) == len(several_files)


@pytest.mark.django_db(transaction=True)
def test_render_chats(
    client: Client,
    user_with_chats_with_messages_over_time: User,
    chat_with_files: Chat,
):
    # Given
    alice = user_with_chats_with_messages_over_time
    client.force_login(alice)
    factory = RequestFactory()
    request = factory.get("/chats/")
    request.user = alice
    context = chats_service.get_context(request)
    chat_context = chats_service.get_context(request, chat_with_files.id)

    # When
    response = chats_service.render_chats(request, context)
    chat_response = chats_service.render_chats(request, chat_context)

    # Then
    assert response.status_code == HTTPStatus.OK
    assert "canned-prompts" in response.content.decode()

    assert chat_response.status_code == HTTPStatus.OK
    assert "canned-prompts" not in chat_response.content.decode()


@pytest.mark.django_db(transaction=True)
def test_render_conversations(
    client: Client,
    user_with_chats_with_messages_over_time: User,
    chat_with_files: Chat,
):
    # Given
    alice = user_with_chats_with_messages_over_time
    client.force_login(alice)
    factory = RequestFactory()
    request = factory.get("/chats/")
    request.user = alice
    chat_context = chats_service.get_context(request, chat_with_files.id)

    # When
    response = chats_service.render_conversations(request)
    chat_response = chats_service.render_conversations(request, chat_context)

    # Then
    assert response.status_code == HTTPStatus.OK
    assert "rbds-list-row--selected" not in response.content.decode()

    assert chat_response.status_code == HTTPStatus.OK
    assert "rbds-list-row--selected" in chat_response.content.decode()
