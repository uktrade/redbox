import httpx
import pytest
from mcp_server import SSOIntrospectionVerifier

error_msg = "test"


class DummyResponse:
    def __init__(self, status_code: int, payload: dict | None):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict | None:
        return self._payload


@pytest.mark.asyncio
async def test_token_returns_none_when_no_config():
    verifier = SSOIntrospectionVerifier("", "token")
    token = await verifier.verify_token("any-token")
    assert token is None


@pytest.mark.asyncio
async def test_token_returns_none_on_http_error(monkeypatch):
    verifier = SSOIntrospectionVerifier("https://example.test/introspect", "secret")

    async def raise_http_error(*_args, **_kwargs):
        raise httpx.HTTPError(error_msg)

    monkeypatch.setattr(verifier._client, "post", raise_http_error)
    token = await verifier.verify_token("any-token")
    assert token is None


@pytest.mark.asyncio
async def test_token_returns_none_on_unexpected_status(monkeypatch):
    verifier = SSOIntrospectionVerifier("https://example.test/introspect", "secret")

    async def respond(*_args, **_kwargs):
        return DummyResponse(500, {"active": True})

    monkeypatch.setattr(verifier._client, "post", respond)
    token = await verifier.verify_token("any-token")
    assert token is None


@pytest.mark.asyncio
async def test_token_returns_none_when_inactive(monkeypatch):
    verifier = SSOIntrospectionVerifier("https://example.test/introspect", "secret")

    async def respond(*_args, **_kwargs):
        return DummyResponse(200, {"active": False})

    monkeypatch.setattr(verifier._client, "post", respond)
    token = await verifier.verify_token("any-token")
    assert token is None


@pytest.mark.asyncio
async def test_token_returns_access_token_on_active(monkeypatch):
    verifier = SSOIntrospectionVerifier("https://example.test/introspect", "secret")
    payload = {
        "active": True,
        "scope": "read write",
        "client_id": "client-123",
        "exp": 1234567890,
        "aud": "data-hub",
        "custom": "value",
    }

    async def respond(*_args, **_kwargs):
        return DummyResponse(200, payload)

    monkeypatch.setattr(verifier._client, "post", respond)
    token = await verifier.verify_token("any-token")

    assert token is not None
    assert token.token == "any-token"
    assert token.client_id == "client-123"
    assert token.scopes == ["read", "write"]
    assert token.expires_at == 1234567890
    assert token.resource == "data-hub"
    assert token.claims["custom"] == "value"
