import json
import logging
import uuid
from http import HTTPStatus

import pytest
from bs4 import BeautifulSoup
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

from redbox_app.redbox_core.models import (
    Chat,
    ChatMessage,
)

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_user_can_see_their_own_chats(chat_with_message: Chat, alice: User, client: Client):
    # Given
    client.force_login(alice)

    # When
    response = client.get(f"/chats/{chat_with_message.id}/")

    # Then
    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_user_cannot_see_other_users_chats(chat: Chat, bob: User, client: Client):
    # Given
    client.force_login(bob)

    # When
    response = client.get(f"/chats/{chat.id}/")

    # Then
    assert response.status_code == HTTPStatus.FOUND
    assert response.headers.get("Location") == "/chats/"


@pytest.mark.django_db
def test_view_session_with_documents(chat_message: ChatMessage, client: Client):
    # Given
    client.force_login(chat_message.chat.user)
    chat_id = chat_message.chat.id

    # When
    response = client.get(f"/chats/{chat_id}/")

    # Then
    assert response.status_code == HTTPStatus.OK
    assert b"original_file" in response.content


@pytest.mark.django_db
def test_chat_grouped_by_age(user_with_chats_with_messages_over_time: User, client: Client):
    # Given
    client.force_login(user_with_chats_with_messages_over_time)

    # When
    response = client.get(reverse("chats"))

    # Then
    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_nonexistent_chats(alice: User, client: Client):
    # Given
    client.force_login(alice)
    nonexistent_uuid = uuid.uuid4()

    # When
    url = reverse("chats", kwargs={"chat_id": nonexistent_uuid})
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.django_db
def test_post_chat_title(alice: User, chat: Chat, client: Client):
    # Given
    client.force_login(alice)

    # When
    url = reverse("chat-titles", kwargs={"chat_id": chat.id})
    response = client.post(url, json.dumps({"value": "New chat name"}), content_type="application/json")

    # Then
    status = HTTPStatus(response.status_code)
    assert status.is_success
    chat.refresh_from_db()
    assert chat.name == "New chat name"


@pytest.mark.django_db
def test_post_chat_title_with_naughty_string(alice: User, chat: Chat, client: Client):
    # Given
    client.force_login(alice)

    # When
    url = reverse("chat-titles", kwargs={"chat_id": chat.id})
    response = client.post(url, json.dumps({"value": "New chat name \x00"}), content_type="application/json")

    # Then
    status = HTTPStatus(response.status_code)
    assert status.is_success
    chat.refresh_from_db()
    assert chat.name == "New chat name \ufffd"


@pytest.mark.django_db
def test_staff_user_can_see_route(chat_with_files: Chat, client: Client):
    # Given
    chat_with_files.user.is_staff = True
    chat_with_files.user.save()
    client.force_login(chat_with_files.user)

    # When
    response = client.get(f"/chats/{chat_with_files.id}/")

    # Then
    assert response.status_code == HTTPStatus.OK
    assert b"redbox-message-route" in response.content
    assert b"redbox-message-route govuk-!-display-none" not in response.content


@pytest.mark.django_db
def test_recent_chats_with_chat(user_with_chats_with_messages_over_time: User, client: Client):
    # Given
    user = user_with_chats_with_messages_over_time
    client.force_login(user)
    chats = Chat.get_ordered_by_last_message_date(user)

    # When
    response = client.get(reverse("recent-chats", kwargs={"active_chat_id": chats[0].id}))
    soup = BeautifulSoup(response.content)
    selected_chat = soup.find(
        "div",
        class_=["chat-list-item", "selected"],
        attrs={"data-chatid": str(chats[0].id)},
    )

    # Then
    assert response.status_code == HTTPStatus.OK
    assert list(response.context_data["chats"]) == list(chats)
    assert selected_chat is not None


@pytest.mark.django_db
def test_recent_chats_without_chat(user_with_chats_with_messages_over_time: User, client: Client):
    # Given
    user = user_with_chats_with_messages_over_time
    client.force_login(user)
    chats = Chat.get_ordered_by_last_message_date(user)

    # When
    response = client.get(reverse("recent-chats"))
    soup = BeautifulSoup(response.content)
    chat_items = soup.find_all("div", class_="chat-list-item")
    rendered_ids = [item["data-chatid"] for item in chat_items]

    # Then
    assert response.status_code == HTTPStatus.OK
    assert list(response.context_data["chats"]) == list(chats)
    for chat in chats:
        assert str(chat.id) in rendered_ids
    for chat_item in chat_items:
        assert "selected" not in chat_item.get("class", [])


@pytest.mark.django_db
def test_chat_window_with_chat(chat_with_message: Chat, client: Client):
    # Given
    user = chat_with_message.user
    client.force_login(user)
    message = ChatMessage.objects.filter(chat=chat_with_message).first()

    # When
    response = client.get(reverse("chat-window", kwargs={"active_chat_id": chat_with_message.id}))
    response_content = response.content.decode()

    # Then
    assert response.status_code == HTTPStatus.OK
    assert response.context_data["current_chat"] == chat_with_message
    assert f"{message.id}" in response_content


@pytest.mark.django_db
def test_chat_window_without_chat(alice: User, client: Client):
    # Given
    client.force_login(alice)

    # When
    response = client.get(reverse("chat-window"))
    soup = BeautifulSoup(response.content)
    canned_prompt = soup.find("canned-prompts")

    # Then
    assert response.status_code == HTTPStatus.OK
    assert not response.context_data["current_chat"]
    assert canned_prompt is not None
