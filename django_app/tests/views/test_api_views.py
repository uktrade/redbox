import logging
from http import HTTPStatus

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_api_view(client: Client, api_key: str):
    # Given
    headers = {"HTTP_X_API_KEY": api_key}

    # When
    url = reverse("user-view")
    response = client.get(url, **headers)

    # Then
    assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize("path_name", ["user-view", "message-view"])
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
    user_with_chats_with_messages_over_time: User, client: Client, api_key
):
    # Given
    headers = {"HTTP_X_API_KEY": api_key}

    # When
    url = reverse("message-view")
    response = client.get(url, **headers)

    # Then
    assert response.status_code == HTTPStatus.OK
    assert len(response.json()["results"]) == sum(
        len(chat.chatmessage_set.all()) for chat in user_with_chats_with_messages_over_time.chat_set.all()
    )
