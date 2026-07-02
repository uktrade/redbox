import time
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError

import redbox.chains.components as components
from redbox.chains.components import _FallbackCacheCallback, get_chat_llm
from redbox.models.chain import AISettings, ChatLLMBackend

pytestmark = pytest.mark.usefixtures("clear_fallback_cache")


@pytest.fixture(autouse=True)
def clear_fallback_cache():
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
    primary_mock = MagicMock(name="PrimaryModel")
    fallback_mock = MagicMock(name="FallbackModel")
    mocker.patch("redbox.chains.components.init_chat_model", side_effect=[primary_mock, fallback_mock])

    get_chat_llm(fake_model_backend, fake_ai_settings)

    primary_mock.with_config.assert_called_once()
    primary_mock.with_config.return_value.with_fallbacks.assert_called_once()

    fallbacks_arg = primary_mock.with_config.return_value.with_fallbacks.call_args[0][0]

    assert fallback_mock in fallbacks_arg


def test_get_chat_llm_fallback_on_throttling(mocker, fake_model_backend, fake_ai_settings):
    primary_mock = MagicMock(name="PrimaryModel")
    fallback_mock = MagicMock(name="FallbackModel")
    mocker.patch("redbox.chains.components.init_chat_model", side_effect=[primary_mock, fallback_mock])

    get_chat_llm(fake_model_backend, fake_ai_settings)

    call_kwargs = primary_mock.with_config.return_value.with_fallbacks.call_args[1]
    assert ClientError in call_kwargs["exceptions_to_handle"]

    callback_list = primary_mock.with_config.call_args[1]["callbacks"]
    assert any(isinstance(cb, _FallbackCacheCallback) for cb in callback_list)

    fallbacks_arg = primary_mock.with_config.return_value.with_fallbacks.call_args[0][0]
    assert fallback_mock in fallbacks_arg


def test_get_chat_llm_wires_connection_error_fallback(mocker, fake_model_backend, fake_ai_settings):
    primary_mock = MagicMock(name="PrimaryModel")
    fallback_mock = MagicMock(name="FallbackModel")
    mocker.patch("redbox.chains.components.init_chat_model", side_effect=[primary_mock, fallback_mock])

    get_chat_llm(fake_model_backend, fake_ai_settings)

    call_kwargs = primary_mock.with_config.return_value.with_fallbacks.call_args[1]
    handled = call_kwargs["exceptions_to_handle"]

    assert TimeoutError in handled
    assert ConnectTimeoutError in handled
    assert EndpointConnectionError in handled
    assert ReadTimeoutError in handled


def test_get_chat_llm_uses_cached_fallback(mocker, fake_model_backend, fake_ai_settings):
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
    components._FALLBACK_CACHE[fake_model_backend.name] = {
        "until": time.time() - 1,  # expired
        "backend": ChatLLMBackend(name="anthropic.fallback", provider="bedrock"),
    }

    init_mock = mocker.patch("redbox.chains.components.init_chat_model", return_value=MagicMock(name="PrimaryModel"))
    get_chat_llm(fake_model_backend, fake_ai_settings)

    # once the cache has expired; the primary model should be the first call
    assert init_mock.call_args_list[0].kwargs["model"] == fake_model_backend.name


def test_fallback_cache_callback_updates_cache_on_throttling():
    components._FALLBACK_CACHE.clear()
    fallback_backend = ChatLLMBackend(name="anthropic.claude-3-7-sonnet-20250219-v1:0", provider="bedrock")
    cb = _FallbackCacheCallback("primary-model", fallback_backend)

    cb.on_llm_error(_make_client_error("ServiceUnavailableException"))

    assert "primary-model" in components._FALLBACK_CACHE
    entry = components._FALLBACK_CACHE["primary-model"]
    assert entry["until"] > time.time()
    assert entry["backend"] == fallback_backend


def test_fallback_cache_callback_ignores_non_throttling_error():
    components._FALLBACK_CACHE.clear()
    fallback_backend = ChatLLMBackend(name="anthropic.claude-3-7-sonnet-20250219-v1:0", provider="bedrock")
    cb = _FallbackCacheCallback("primary-model", fallback_backend)

    cb.on_llm_error(_make_client_error("ValidationException"))

    assert "primary-model" not in components._FALLBACK_CACHE
