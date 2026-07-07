import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from redbox.loader.chunking.tokeniser import (
    titan_tokeniser,
    _fallback_estimate,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """titan_tokeniser is lru_cache'd — clear between tests so mocks don't bleed across."""
    titan_tokeniser.cache_clear()
    yield
    titan_tokeniser.cache_clear()


def _make_bedrock_response(token_count: int | None):
    """Build a fake boto3 invoke_model response with (or without) the token-count header."""
    headers = {}
    if token_count is not None:
        headers["x-amzn-bedrock-input-token-count"] = str(token_count)
    return {"ResponseMetadata": {"HTTPHeaders": headers}}


class TestFallbackEstimate:
    def test_empty_string(self):
        assert _fallback_estimate("") == 0

    def test_basic_ratio(self):
        # len // 3
        assert _fallback_estimate("abcdef") == 2  # 6 // 3
        assert _fallback_estimate("abcdefg") == 2  # 7 // 3

    def test_is_pessimistic_relative_to_normal_token_ratio(self):
        # Real tokenisers average ~4 chars/token; //3 should overestimate token count
        text = "a" * 120  # ~30 tokens at 4 chars/token, //3 gives 40
        assert _fallback_estimate(text) == 40
        assert _fallback_estimate(text) > 120 // 4

    def test_dense_punctuation_text(self):
        text = '"2026-03-08T14:00:00.000Z","Free AI","0"\n'
        result = _fallback_estimate(text)
        assert result == len(text) // 3
        assert result > 0


class TestTitanTokeniserHappyPath:
    def test_returns_header_token_count(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(42)

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            result = titan_tokeniser("some text")

        assert result == 42

    def test_calls_invoke_model_with_expected_payload(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(10)

        with (
            patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client),
            patch("redbox.loader.chunking.tokeniser.env") as mock_env,
        ):
            mock_env.embedding_backend = "amazon.titan-embed-text-v2:0"
            titan_tokeniser("hello world")

        _, kwargs = mock_client.invoke_model.call_args
        assert kwargs["modelId"] == "amazon.titan-embed-text-v2:0"
        assert json.loads(kwargs["body"]) == {"inputText": "hello world"}
        assert kwargs["accept"] == "application/json"
        assert kwargs["contentType"] == "application/json"

    def test_empty_string_short_circuits_without_calling_bedrock(self):
        mock_client = MagicMock()

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            result = titan_tokeniser("")

        assert result == 0
        mock_client.invoke_model.assert_not_called()

    def test_result_is_int_even_if_header_is_string(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(7)

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            result = titan_tokeniser("x")

        assert isinstance(result, int)
        assert result == 7


class TestTitanTokeniserMissingHeader:
    def test_falls_back_when_header_absent(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(None)

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            text = "abcdefghi"  # len 9 -> fallback = 3
            result = titan_tokeniser(text)

        assert result == 3

    def test_logs_warning_when_header_absent(self, caplog):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(None)

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            with caplog.at_level(logging.WARNING):
                titan_tokeniser("abcdef")

        assert any("falling back to estimate" in rec.message.lower() for rec in caplog.records)


class TestTitanTokeniserErrorHandling:
    def test_falls_back_on_boto_exception(self):
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = Exception("boom")

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            text = "abcdefghi"  # len 9 -> fallback = 3
            result = titan_tokeniser(text)

        assert result == 3

    def test_logs_exception_on_failure(self, caplog):
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = Exception("throttled")

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            with caplog.at_level(logging.ERROR):
                titan_tokeniser("some text")

        assert any("titan tokeniser call failed" in rec.message.lower() for rec in caplog.records)

    def test_fallback_never_raises_even_on_client_error(self):
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = ConnectionError("network down")

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            # Should not raise — must degrade gracefully
            result = titan_tokeniser("resilient text")

        assert isinstance(result, int)
        assert result >= 0

    def test_fallback_result_matches_manual_calc_on_error(self):
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = Exception("fail")
        text = "x" * 300

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            result = titan_tokeniser(text)

        assert result == 100  # 300 // 3


class TestTitanTokeniserCaching:
    def test_identical_calls_use_cache_not_repeated_bedrock_calls(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(15)

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            titan_tokeniser("repeat me")
            titan_tokeniser("repeat me")
            titan_tokeniser("repeat me")

        # lru_cache means invoke_model should only be hit once for identical input
        assert mock_client.invoke_model.call_count == 1

    def test_different_inputs_each_trigger_a_call(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(5)

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            titan_tokeniser("text one")
            titan_tokeniser("text two")
            titan_tokeniser("text three")

        assert mock_client.invoke_model.call_count == 3


class TestTitanTokeniserRealisticData:
    """Regression coverage for the tabular/CSV case that originally caused the overflow bug."""

    def test_csv_timestamp_row_uses_mocked_bedrock_count(self):
        row = '"2026-03-08T14:00:00.000Z","Free AI","0"\n'
        mock_client = MagicMock()
        # Simulate Titan's real (higher) count for this punctuation-dense row
        mock_client.invoke_model.return_value = _make_bedrock_response(18)

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            result = titan_tokeniser(row)

        assert result == 18

    def test_large_table_text_falls_back_pessimistically_on_bedrock_failure(self):
        large_table = "\n".join(f'"2026-03-08T{h:02d}:00:00.000Z","Free AI","0"' for h in range(24))
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = Exception("service unavailable")

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            result = titan_tokeniser(large_table)

        assert result == len(large_table) // 3
        assert result > 0


class TestTitanTokeniserEdgeCases:
    def test_whitespace_only_string(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(1)

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            result = titan_tokeniser("   ")

        assert result == 1

    def test_unicode_text(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(6)

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            result = titan_tokeniser("héllo wörld 你好")

        assert result == 6

    def test_zero_token_count_header_is_respected_not_treated_as_falsy_none(self):
        """Regression: `if token_count is None` must not misfire on the string '0'."""
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(0)

        with patch("redbox.loader.chunking.tokeniser._bedrock_client", mock_client):
            result = titan_tokeniser("a")

        assert result == 0
        mock_client.invoke_model.assert_called_once()
