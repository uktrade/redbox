import logging
from http import HTTPStatus

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db()
def test_api_view(user_with_chats_with_messages_over_time: User, client: Client, api_key: str):
    # Given
    headers = {"HTTP_X_API_KEY": api_key}

    # When
    url = reverse("user-view")
    response = client.get(url, **headers)

    # Then
    assert response.status_code == HTTPStatus.OK
    user_with_chats = next(user for user in response.json()["results"] if user["chats"])
    assert user_with_chats["ai_experience"] == user_with_chats_with_messages_over_time.ai_experience


@pytest.mark.django_db()
def test_api_view_fail(client: Client):
    # Given that the user does not pass an API key

    # When
    url = reverse("user-view")
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {"detail": "No API key provided"}
