import logging
from http import HTTPStatus

import pytest
from django.contrib.auth import get_user_model
from django.core.exceptions import SuspiciousOperation
from django.test import Client, RequestFactory
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


@pytest.mark.parametrize("invalid_token", ["opaque-access-token", "opaque-id-token"])
def test_auth_callback_rejects_non_jwt_tokens(mocker, invalid_token):
    request = RequestFactory().get("/auth/callback/?code=auth-code&state=oauth-state")
    request.session = {"_authbroker_token_oauth_state": "oauth-state"}
    token = {
        "id_token": "header.payload.signature",
        "access_token": "header.payload.signature",
    }
    token["access_token" if "access" in invalid_token else "id_token"] = invalid_token

    mocker.patch.object(AuthCallbackView, "fetch_token", return_value=token)

    with pytest.raises(SuspiciousOperation, match="JWT"):
        AuthCallbackView().get(request)
