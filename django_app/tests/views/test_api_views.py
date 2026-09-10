# ruff: noqa: ARG001

import logging
from http import HTTPStatus
from typing import Any

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse
from tests.playwright.pages import ChatMessage

from redbox_app.redbox_core.models import ChatMessageFeedback

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.fixture
def api_key_header(api_key) -> dict[str, Any]:
    return {"HTTP_X_API_KEY": api_key}


@pytest.mark.django_db
def test_api_view(client: Client, api_key_header: dict[str, Any]):
    # When
    url = reverse("user-view")
    response = client.get(url, **api_key_header)

    # Then
    assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize("path_name", ["user-view", "message-view", "get-all-message-feedback"])
@pytest.mark.django_db
def test_api_view_fail(path_name, client: Client):
    # Given that the user does not pass an API key

    # When
    url = reverse(path_name)
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {"detail": "No API key provided"}


@pytest.mark.django_db
def test_superuser_client_querying_v0_messages_returns_200(
    user_with_chats_with_messages_over_time: User, client: Client, api_key_header: dict[str, Any]
):

    # When
    url = reverse("message-view")
    response = client.get(url, **api_key_header)

    # Then
    assert response.status_code == HTTPStatus.OK
    assert len(response.json()["results"]) == sum(
        len(chat.chatmessage_set.all()) for chat in user_with_chats_with_messages_over_time.chat_set.all()
    )


# --- get all message feedback ---


@pytest.fixture
def negative_feedback(chat_message: ChatMessage) -> ChatMessageFeedback:
    return ChatMessageFeedback.objects.create(
        message=chat_message,
        is_positive=False,
        reason=[ChatMessageFeedback.Reason.INACCURATE, ChatMessageFeedback.Reason.LACKED_DETAIL],
        detail="It made things up.",
    )


def test_feedback_returns_correct_object(
    client: Client, api_key_header: dict[str, Any], chat_message: ChatMessage, negative_feedback: ChatMessageFeedback
):
    response = client.get(reverse("get-all-message-feedback"), **api_key_header)

    assert response.status_code == HTTPStatus.OK
    results = response.json()["results"]
    assert len(results) == 1

    result = results[0]
    assert result["message"] == str(chat_message.id)
    assert result["is_positive"] is False
    assert result["reason"] == ["INACCURATE", "LACKED_DETAIL"]
    assert result["reason_labels"] == ["It was inaccurate", "It was lacking detail"]
    assert result["detail"] == "It made things up."


def test_feedback_returns_oldest_first(
    client: Client, api_key_header: dict[str, Any], user_with_chats_with_messages_over_time: User
):
    messages = list(ChatMessageFeedback.objects.order_by("created_at"))

    for message in reversed(messages):
        feedback = ChatMessageFeedback.objects.create(message=message, is_positive=True)
        # .update() bypasses auto_now_add so we can backdate created_at
        ChatMessageFeedback.objects.filter(pk=feedback.pk).update(created_at=message.created_at)

    response = client.get(reverse("get-all-message-feedback"), **api_key_header)

    assert response.status_code == HTTPStatus.OK
    assert [r["message"] for r in response.json()["results"]] == [str(m.id) for m in messages]


def test_feedback_empty_when_none_exists(client: Client, api_key_header: dict[str, Any]):
    response = client.get(reverse("get-all-message-feedback"), **api_key_header)

    assert response.status_code == HTTPStatus.OK
    assert response.json()["results"] == []
