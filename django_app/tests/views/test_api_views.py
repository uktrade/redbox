# ruff: noqa: ARG001

import logging
from datetime import UTC, datetime, timedelta
from http import HTTPStatus

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

User = get_user_model()

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.django_db


@pytest.fixture
def api_key_header(api_key):
    return {"HTTP_X_API_KEY": api_key}


def test_api_view(client: Client, api_key_header: dict[str, str]):

    # When
    url = reverse("user-view")
    response = client.get(url, **api_key_header)

    # Then
    assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize("path_name", ["user-view", "message-view", "message-view-v1"])
def test_api_view_fail(path_name, client: Client):

    # When
    url = reverse(path_name)
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {"detail": "No API key provided"}


def test_superuser_client_querying_v0_messages_returns_200(
    user_with_chats_with_messages_over_time: User, client: Client, api_key_header
):

    # When
    url = reverse("message-view")
    response = client.get(url, **api_key_header)

    # Then
    assert response.status_code == HTTPStatus.OK
    assert len(response.json()["results"]) == sum(
        len(chat.chatmessage_set.all()) for chat in user_with_chats_with_messages_over_time.chat_set.all()
    )


def test_messages_v1_anonymous_user_is_rejected(client):
    response = client.get(reverse("message-view-v1"))

    assert response.status_code == HTTPStatus.FORBIDDEN


def test_messages_v1_non_admin_user_is_rejected(client, alice):
    client.force_login(alice)
    response = client.get(reverse("message-view-v1"))

    assert response.status_code == HTTPStatus.FORBIDDEN


def test_messages_v1_admin_user_with_no_api_key_is_rejected(client, superuser):
    "Due to the way the default authentication classes are set up currently, session auth isn't enabled"
    client.force_login(superuser)
    response = client.get(reverse("message-view-v1"))

    assert response.status_code == HTTPStatus.FORBIDDEN


def test_messages_v1_bad_api_key_is_rejected(client):
    bad_header = {"HTTP_X_API_KEY": "not-an-api-key"}

    response = client.get(reverse("message-view-v1"), **bad_header)

    assert response.status_code == HTTPStatus.FORBIDDEN


def test_messages_v1_returns_messages_oldest_first(client, api_key_header, user_with_chats_with_messages_over_time):
    response = client.get(reverse("message-view-v1"), **api_key_header)

    assert response.status_code == HTTPStatus.OK
    assert [r["text"] for r in response.json()["results"]] == [
        "40 days old",
        "20 days old",
        "5 days old",
        "yesterday",
        "today",
    ]


def test_messages_v1_created_after_excludes_older_messages(
    client,
    api_key_header,
    user_with_chats_with_messages_over_time,
):
    created_after = (datetime.now(tz=UTC) - timedelta(days=15)).isoformat()

    response = client.get(reverse("message-view-v1"), data={"created_after": created_after}, **api_key_header)

    assert response.status_code == HTTPStatus.OK, response.text
    assert [r["text"] for r in response.json()["results"]] == ["5 days old", "yesterday", "today"]


def test_messages_v1_returns_correct_object(client, api_key_header, chat_message, negative_feedback):
    response = client.get(reverse("message-view-v1"), **api_key_header)

    assert len(response.json()["results"]) == 1

    result = response.json()["results"][0]

    assert result["text"] == "A question?"
    assert result["feedback"]["is_positive"] is False
    assert result["feedback"]["reason"] == ["INACCURATE", "LACKED_DETAIL"]
    assert result["feedback"]["detail"] == "It made things up."

    assert result["source_files"][0]["file_name"] == "original_file.txt"


def test_messages_v1_feedback_is_null_when_message_has_none(client, api_key_header, chat_message):
    response = client.get(reverse("message-view-v1"), **api_key_header)
    assert response.json()["results"][0]["feedback"] is None
