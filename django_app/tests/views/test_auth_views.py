import logging
from http import HTTPStatus
from urllib.parse import parse_qs, urlsplit

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse
from django.utils import timezone

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
@pytest.mark.parametrize(
    ("has_valid_sso_token", "expected_route"),
    [
        (False, "reauthenticate"),
        (True, "chats"),
    ],
)
def test_sign_in_requires_valid_sso(
    alice: User,
    client: Client,
    has_valid_sso_token: bool,
    expected_route: str,
):
    client.force_login(alice)

    if has_valid_sso_token:
        session = client.session
        session["_authbroker_token"] = {"access_token": "test-token", "expires_at": timezone.now().timestamp() + 3600}
        session.save()

    response = client.get(reverse("sign-in"))

    assert response.status_code == HTTPStatus.FOUND
    assert response.url == reverse(expected_route)


@pytest.mark.django_db
def test_reauthenticate_preserves_next_and_logs_out(alice: User, client: Client):
    client.force_login(alice)
    next_url = "/chats/?q=fake%20prompt"

    response = client.get(reverse("reauthenticate"), {"next": next_url})

    assert response.status_code == HTTPStatus.FOUND

    redirect_url = urlsplit(response.url)
    assert redirect_url.path == reverse("authbroker_client:login")
    assert parse_qs(redirect_url.query) == {"next": [next_url]}

    assert "_auth_user_id" not in client.session


@pytest.mark.django_db
@pytest.mark.parametrize(("unsafe_next"), ["https://example.com", "//example.com/", None])
def test_reauthenticate_rejects_unsafe_next(client: Client, unsafe_next: str | None):
    params = {"next": unsafe_next} if unsafe_next else {}

    response = client.get(reverse("reauthenticate"), params)

    assert response.status_code == HTTPStatus.FOUND

    redirect_url = urlsplit(response.url)
    assert redirect_url.path == reverse("authbroker_client:login")
    assert parse_qs(redirect_url.query) == {"next": [reverse("chats")]}
