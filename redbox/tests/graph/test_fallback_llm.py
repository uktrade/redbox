import time
import pytest
from unittest.mock import MagicMock
from botocore.exceptions import ClientError

from redbox.models.chain import ChatLLMBackend, AISettings
from redbox.chains.components import get_chat_llm, get_base_chat_llm, _FALLBACK_CACHE

pytestmark = pytest.mark.usefixtures("clear_fallback_cache")


@pytest.fixture(autouse=True)
def clear_fallback_cache():
    import redbox.chains.components as components

    if hasattr(components, "_FALLBACK_CACHE"):
        components._FALLBACK_CACHE.clear()
    yield
    if hasattr(components, "_FALLBACK_CACHE"):
        components._FALLBACK_CACHE.clear()


@pytest.fixture
def fake_model_backend():
    return ChatLLMBackend(name="bedrock.fake-model", provider="bedrock")


@pytest.fixture
def fake_ai_settings():
    return AISettings(llm_max_tokens=256)


def _make_client_error(code: str):
    return ClientError(
        error_response={"Error": {"Code": code, "Message": f"{code} occurred"}},
        operation_name="InvokeModel",
    )


def test_get_chat_llm_primary_success(mocker, fake_model_backend, fake_ai_settings):
    fake_model = MagicMock(name="FakeChatModel")
    mocker.patch("redbox.chains.components.init_chat_model", return_value=fake_model)

    result = get_chat_llm(fake_model_backend, fake_ai_settings)

    assert result == fake_model
    assert fake_model.bind_tools not in (None,)


def test_get_chat_llm_fallback_on_throttling(mocker, fake_model_backend, fake_ai_settings):
    init_mock = mocker.patch("redbox.chains.components.init_chat_model")

    init_mock.side_effect = [
        _make_client_error("ThrottlingException"),
        MagicMock(name="FallbackModel"),
    ]

    get_chat_llm(fake_model_backend, fake_ai_settings)
    assert fake_model_backend.name in _FALLBACK_CACHE


def test_get_chat_llm_fallback_on_timeout(mocker, fake_model_backend, fake_ai_settings):
    init_mock = mocker.patch("redbox.chains.components.init_chat_model")

    init_mock.side_effect = [
        TimeoutError("timed out"),
        MagicMock(name="FallbackModel"),
    ]

    get_chat_llm(fake_model_backend, fake_ai_settings)
    assert fake_model_backend.name in _FALLBACK_CACHE


def test_get_chat_llm_uses_cached_fallback(mocker, fake_model_backend, fake_ai_settings):
    import redbox.chains.components as components

    fallback_backend = ChatLLMBackend(name="anthropic.fallback", provider="bedrock")
    components._FALLBACK_CACHE[fake_model_backend.name] = {
        "until": time.time() + 60,
        "backend": fallback_backend,
    }

    init_mock = mocker.patch("redbox.chains.components.init_chat_model", return_value=MagicMock(name="CachedModel"))

    get_chat_llm(fake_model_backend, fake_ai_settings)

    init_mock.assert_called_once_with(
        model=fallback_backend.name,
        model_provider=fallback_backend.provider,
        max_tokens=fake_ai_settings.llm_max_tokens,
        configurable_fields=["base_url"],
    )


def test_get_chat_llm_cache_expires_and_returns_to_primary(mocker, fake_model_backend, fake_ai_settings):
    import redbox.chains.components as components

    components._FALLBACK_CACHE[fake_model_backend.name] = {
        "until": time.time() - 1,  # expired
        "backend": ChatLLMBackend(name="anthropic.fallback", provider="bedrock"),
    }

    mocker.patch("redbox.chains.components.init_chat_model", return_value=MagicMock(name="PrimaryModel"))

    get_chat_llm(fake_model_backend, fake_ai_settings)
    assert components._FALLBACK_CACHE.get(fake_model_backend.name)


def test_get_base_chat_llm_same_fallback_behavior(mocker, fake_model_backend, fake_ai_settings):
    init_mock = mocker.patch("redbox.chains.components.init_chat_model")

    init_mock.side_effect = [
        _make_client_error("RateLimitExceeded"),
        MagicMock(name="BaseFallbackModel"),
    ]

    get_base_chat_llm(fake_model_backend, fake_ai_settings)
    assert fake_model_backend.name in _FALLBACK_CACHE
