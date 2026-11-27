import pytest
from django.conf import settings
from django.utils import timezone

from redbox.setting_enums import Environment
from redbox_app.redbox_core.services.sso import get_usable_sso_token


@pytest.mark.parametrize(("seconds_past_buffer", "expected_token"), [(-10, None), (60, "test-token")])
def test_sso_token_respects_expiry_buffer(seconds_past_buffer: int, expected_token: str | None):
    expires_at = timezone.now().timestamp() + settings.AUTH_EXPIRY_BUFFER + seconds_past_buffer
    session = {"_authbroker_token": {"access_token": "test-token", "expires_at": expires_at}}

    assert get_usable_sso_token(session) == expected_token


@pytest.mark.parametrize(
    ("environment", "expected_token"),
    [
        (Environment.LOCAL, "test-token"),
        (Environment.INTEGRATION, "test-token"),
        (Environment.TESTING, None),
        (Environment.PROD, None),
    ],
)
def test_sso_token_without_expiry_depends_on_environment(
    monkeypatch, environment: Environment, expected_token: str | None
):
    monkeypatch.setattr(settings, "ENVIRONMENT", environment)

    session = {"_authbroker_token": {"access_token": "test-token"}}

    assert get_usable_sso_token(session) == expected_token
