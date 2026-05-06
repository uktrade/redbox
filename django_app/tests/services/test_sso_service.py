from http import HTTPStatus
from unittest.mock import Mock, patch

import pytest
from django.contrib.auth import get_user_model
from django.contrib.sessions.middleware import SessionMiddleware
from django.test import RequestFactory

from redbox_app.redbox_core.models import UserSSO, UserSSOAttribute
from redbox_app.redbox_core.services import sso as sso_service

User = get_user_model()


def build_request(user, access_token=None):
    factory = RequestFactory()
    request = factory.get("/")
    request.user = user

    middleware = SessionMiddleware(lambda _: None)
    middleware.process_request(request)

    if access_token:
        request.session["_authbroker_token"] = {"access_token": access_token}

    request.session.save()
    return request


@pytest.mark.django_db
def test_fetch_sso_payload_success(alice: User, sso_user_me_payload_factory):
    request = build_request(alice, access_token="mock_access_token")

    mock_response_data = sso_user_me_payload_factory(alice)

    mock_response = Mock()
    mock_response.json.return_value = mock_response_data
    mock_response.status_code = HTTPStatus.OK

    with patch("requests.get", return_value=mock_response) as mock_get:
        data = sso_service.fetch_sso_payload(request)

    mock_get.assert_called_once()
    assert data == mock_response_data


@pytest.mark.django_db
def test_fetch_sso_payload_no_token(alice: User):
    request = build_request(alice, access_token=None)

    data = sso_service.fetch_sso_payload(request)

    assert data is None


@pytest.mark.django_db
def test_fetch_sso_payload_non_200(alice: User):
    request = build_request(alice, access_token="mock_access_token")

    mock_response = Mock()
    mock_response.status_code = HTTPStatus.BAD_REQUEST
    mock_response.text = "error"

    with patch("requests.get", return_value=mock_response):
        data = sso_service.fetch_sso_payload(request)

    assert data is None


@pytest.mark.django_db
def test_fetch_sso_payload_request_exception(alice: User):
    request = build_request(alice, access_token="mock_access_token")

    with patch("requests.get", side_effect=Exception("boom")):
        data = sso_service.fetch_sso_payload(request)

    assert data is None


@pytest.mark.django_db
def test_fetch_sso_payload_no_response(alice: User):
    request = build_request(alice, access_token="mock_access_token")

    with patch("requests.get", return_value=None):
        data = sso_service.fetch_sso_payload(request)

    assert data is None


@pytest.mark.django_db
def test_sync_sso_data_creates_sso(alice: User, sso_user_me_payload_factory):
    request = build_request(alice, access_token="token")

    payload = sso_user_me_payload_factory(alice)

    with patch("redbox_app.redbox_core.services.sso.fetch_sso_payload", return_value=payload):
        result = sso_service.sync_sso_data(request)

    assert result is True

    sso = UserSSO.objects.get(user=alice)

    assert sso.email == payload["email"]
    assert sso.contact_email == payload["contact_email"]
    assert sso.first_name == payload["first_name"]
    assert sso.last_name == payload["last_name"]
    assert sso.payload == payload

    related = UserSSOAttribute.objects.filter(sso=sso)
    assert related.count() == len(payload["related_emails"])


@pytest.mark.django_db
def test_sync_sso_data_updates_existing(alice: User, sso_user_me_payload_factory):
    request = build_request(alice, access_token="token")

    existing = UserSSO.objects.create(
        user=alice,
        email="old@example.com",
    )

    payload = sso_user_me_payload_factory(alice)

    with patch("redbox_app.redbox_core.services.sso.fetch_sso_payload", return_value=payload):
        result = sso_service.sync_sso_data(request)

    assert result is True

    existing.refresh_from_db()

    assert existing.email == payload["email"]
    assert UserSSO.objects.filter(user=alice).count() == 1


@pytest.mark.django_db
def test_sync_sso_data_no_payload(alice: User):
    request = build_request(alice, access_token="token")

    with patch("redbox_app.redbox_core.services.sso.fetch_sso_payload", return_value=None):
        result = sso_service.sync_sso_data(request)

    assert result is False
    assert not UserSSO.objects.filter(user=alice).exists()


@pytest.mark.django_db
def test_sync_sso_data_missing_fields(alice: User):
    request = build_request(alice, access_token="token")

    payload = {
        "email": None,
        "related_emails": [],
    }

    with patch("redbox_app.redbox_core.services.sso.fetch_sso_payload", return_value=payload):
        sso_service.sync_sso_data(request)

    sso = UserSSO.objects.get(user=alice)

    assert sso.email == ""
    assert sso.contact_email == ""


@pytest.mark.django_db
def test_sync_related_emails_creates(alice: User):
    sso = UserSSO.objects.create(user=alice)

    emails = ["a@test.com", "b@test.com"]

    sso_service.sync_related_emails(sso, emails)

    qs = UserSSOAttribute.objects.filter(sso=sso)

    assert qs.count() == 2
    assert set(qs.values_list("value", flat=True)) == set(emails)


@pytest.mark.django_db
def test_sync_related_emails_replaces_existing(alice: User):
    sso = UserSSO.objects.create(user=alice)

    UserSSOAttribute.objects.create(
        sso=sso,
        attribute_type=UserSSOAttribute.AttributeType.RELATED_EMAILS,
        value="old@test.com",
    )

    new_emails = ["new@test.com"]

    sso_service.sync_related_emails(sso, new_emails)

    qs = UserSSOAttribute.objects.filter(sso=sso)

    assert qs.count() == 1
    assert qs.first().value == "new@test.com"


@pytest.mark.django_db
def test_sync_related_emails_empty_clears(alice: User):
    sso = UserSSO.objects.create(user=alice)

    UserSSOAttribute.objects.create(
        sso=sso,
        attribute_type=UserSSOAttribute.AttributeType.RELATED_EMAILS,
        value="old@test.com",
    )

    sso_service.sync_related_emails(sso, [])

    assert not UserSSOAttribute.objects.filter(sso=sso).exists()
