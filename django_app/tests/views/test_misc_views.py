import json
import logging
from http import HTTPStatus

import pytest
from django.conf import Settings, settings
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse
from yarl import URL

from redbox_app.redbox_core.models import (
    Chat,
    Tool,
)

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_declaration_view_get(peter_rabbit: User, client: Client):
    client.force_login(peter_rabbit)
    response = client.get("/")
    assert HTTPStatus(response.status_code).is_redirection
    assert response.headers["Cache-control"] == "no-store"
    assert "Report-To" not in response.headers


@pytest.mark.django_db
def test_declaration_view_get_with_sentry_security_header_endpoint(
    peter_rabbit: User, client: Client, settings: Settings
):
    settings.SENTRY_REPORT_TO_ENDPOINT = URL("http://example.com")
    client.force_login(peter_rabbit)
    response = client.get("/")
    assert HTTPStatus(response.status_code).is_redirection
    assert json.loads(response.headers["Report-To"]) == {
        "group": "csp-endpoint",
        "max_age": 10886400,
        "endpoints": [{"url": "http://example.com"}],
        "include_subdomains": True,
    }


@pytest.mark.parametrize("path", ["/security", "/.well-known/security.txt"])
def test_security_txt_redirect(path: str, client: Client):
    response = client.get(path)

    assert HTTPStatus(response.status_code).is_redirection
    assert response.headers["Location"] == f"{settings.SECURITY_TXT_REDIRECT}"


@pytest.mark.django_db
def test_user_can_see_privacy_notice(alice: User, client: Client):
    # Given
    url_name = "privacy-notice"
    anonymous_response = response = client.get(reverse(url_name))
    client.force_login(alice)

    # When
    response = client.get(reverse(url_name))

    # Then
    assert response.status_code == HTTPStatus.OK
    assert anonymous_response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_user_can_see_accessibility_statement(alice: User, client: Client):
    # Given
    url_name = "accessibility-statement"
    anonymous_response = response = client.get(reverse(url_name))
    client.force_login(alice)

    # When
    response = client.get(reverse(url_name))

    # Then
    assert response.status_code == HTTPStatus.OK
    assert anonymous_response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_refresh_fragments(alice: User, client: Client, chat: Chat, default_tool: Tool):
    # Given
    client.force_login(alice)
    url_name = "refresh"

    # When
    bad_response = client.get(reverse(url_name))
    response = client.get(
        reverse(url_name),
        data={
            "fragments": ["chat-cta", "conversations"],
            "chat": chat.id,
            "tool": default_tool.slug,
        },
    )

    # Then
    assert bad_response.status_code == HTTPStatus.BAD_REQUEST
    assert response.status_code == HTTPStatus.OK
    assert response.content.decode()
