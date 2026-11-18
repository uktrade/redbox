import logging
from http import HTTPStatus

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

from redbox_app.redbox_core.auth.views import AuthCallbackView

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_sign_in_view_redirect_to_sign_in(alice: User, client: Client):
    # Given a user that does exist in the db Alice

    # When
    url = reverse("sign-in")
    response = client.post(url, data={"email": alice.email})

    # Then
    assert response.status_code == HTTPStatus.FOUND
    assert response.url == "/auth/login/"


@pytest.mark.django_db
def test_sign_in_view_redirect_sign_up(client: Client):
    # Given a user that does not exist in the database

    # When
    url = reverse("sign-in")
    response = client.post(url, data={"email": "not.a.real.user@gov.uk"})

    # Then
    assert response.status_code == HTTPStatus.FOUND
    assert response.url == "/auth/login/"


@pytest.mark.django_db
def test_authenticate_user_reuses_account_with_same_name_and_email_local_part(alice):
    view = AuthCallbackView()
    profile = {
        "email": "example.user@digital.example.com",
        "given_name": alice.first_name,
        "family_name": alice.last_name,
    }
    existing = User.objects.create(
        username="example.user@example.com",
        email="example.user@example.com",
        first_name=alice.first_name,
        last_name=alice.last_name,
    )

    user = view.find_existing_user(User, profile, profile["email"])

    assert user == existing
    assert User.objects.count() == 2
