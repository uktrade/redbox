import json
import logging
from http import HTTPStatus
from unittest import mock

import pytest
from django.conf import Settings, settings
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse
from yarl import URL

from redbox.models.settings import OpenSearchSettings
from redbox.models.settings import Settings as RedboxSettings
from redbox_app.redbox_core.models import Chat, Tool

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
    no_fragment_response = client.get(reverse(url_name))
    invalid_fragment_response = client.get(
        reverse(url_name),
        data={"fragments": ["invalid"]},
    )
    response = client.get(
        reverse(url_name),
        data={
            "fragments": ["chat-cta", "conversations"],
            "chat": chat.id,
            "tool": default_tool.slug,
        },
    )

    # Then
    assert no_fragment_response.status_code == HTTPStatus.BAD_REQUEST
    assert invalid_fragment_response.status_code == HTTPStatus.BAD_REQUEST
    assert response.status_code == HTTPStatus.OK
    assert response.content.decode()


class TestHealthView:
    def test_get_returns_error_when_elastic_search_cluster_is_unavailable(self, client: Client):
        with mock.patch("redbox_app.redbox_core.views.misc_views.get_settings") as mock_open_search_settings:
            elastic_settings = OpenSearchSettings(
                collection_endpoint="http://admin:password@not_a_cluster:9130",
                collection_endpoint__host="not_a_cluster",
                collection_endpoint__port="9130",
            )
            mock_settings = RedboxSettings(elastic=elastic_settings)
            mock_open_search_settings.return_value = mock_settings

            assert client.get(reverse("health")).status_code == 503

    def test_get_returns_ok_when_all_checks_pass(self, client: Client):
        assert client.get(reverse("health")).status_code == 200
