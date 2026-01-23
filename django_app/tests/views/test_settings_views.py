import logging
from http import HTTPStatus

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_user_can_see_settings(alice: User, client: Client):
    # Given
    client.force_login(alice)

    # When
    response = client.get(reverse("settings"))

    # Then
    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_anon_user_cannot_see_settings(client: Client):
    # Given user is unauthorised

    # When
    response = client.get(reverse("settings"))

    # Then
    assert response.status_code == HTTPStatus.FOUND
    assert response.url.startswith("/auth/login/")
