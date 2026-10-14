from unittest.mock import MagicMock

import pytest

from django_app.redbox_app.backends import TokenCaptureBackend


class TestTokenCaptureBackend:
    @pytest.fixture
    def backend(self):
        return TokenCaptureBackend()

    @pytest.fixture
    def mock_request(self):
        request = MagicMock()
        request.session = {}
        return request

    def test_authenticate_stores_token_in_session(self, backend, mock_request, mocker):
        mock_token = {"access_token": "abc-123"}
        mock_user = MagicMock(username="test_user")
        patched_super = mocker.patch(
            "django_app.redbox_app.backends.AuthbrokerBackend.authenticate", return_value=mock_user
        )
        result = backend.authenticate(mock_request, token=mock_token)
        assert mock_request.session["oauth_token"] == mock_token
        assert result == mock_user
        patched_super.assert_called_once_with(mock_request, token=mock_token)

    def test_authenticate_no_token_provided(self, backend, mock_request, mocker):
        patched_super = mocker.patch("django_app.redbox_app.backends.AuthbrokerBackend.authenticate")
        backend.authenticate(mock_request)
        assert "oauth_token" not in mock_request.session
        patched_super.assert_called_once()

    def test_authenticate_handles_none_request(self, backend, mocker):
        patched_super = mocker.patch("django_app.redbox_app.backends.AuthbrokerBackend.authenticate")
        mock_token = "token-without-request"  # noqa: S105
        result = backend.authenticate(None, token=mock_token)  # noqa: F841
        patched_super.assert_called_once_with(None, token=mock_token)
