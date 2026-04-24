from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from redbox.api.callbacks import TokenAnnotationCallback


class TestTokenCallback:
    @pytest.mark.parametrize(
        "case, llm_response",
        [
            ("no generation", LLMResult(generations=[[]])),
            ("empty response", LLMResult(generations=[[ChatGeneration(message=AIMessage(content=""))]])),
            ("success", LLMResult(generations=[[ChatGeneration(message=AIMessage(content="Fake"))]])),
        ],
    )
    def test_token_callback(self, case, llm_response):
        callback = TokenAnnotationCallback()
        callback.on_llm_end(response=llm_response)

    @pytest.mark.parametrize(
        "metadata, metrics",
        [
            (
                {"model_name": "fake", "stop_reason": "end_turn"},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
        ],
    )
    def test_tokenn_callback_metadata(self, metadata, metrics, mocker):
        mock_span = MagicMock()
        mocker.patch("redbox.api.callbacks.tracer.current_span", return_value=mock_span)
        datadog_mock = mocker.patch("redbox.api.callbacks.LLMObs.annotate")
        chats = ChatGeneration(message=AIMessage(content="", response_metadata=metadata, usage_metadata=metrics))
        callback = TokenAnnotationCallback()
        callback.on_llm_end(response=LLMResult(generations=[[chats]]))

        datadog_mock.assert_called_once_with(
            span=mock_span,
            metadata=metadata,
            metrics=metrics,
            tags={"func": "token_annotation_callback"},
        )

    @pytest.mark.parametrize(
        "metadata, expected_result",
        [
            (
                {"model_name": "fake", "stop_reason": "end_turn"},
                "fake",
            ),
            ({"model_name": "arn:random:blah:not-real/some.model.hello-world-01-01-1999"}, "hello-world-01-01-1999"),
            ({}, None),
        ],
    )
    def test_extract_model_name(self, metadata, expected_result):
        chats = ChatGeneration(message=AIMessage(content="", response_metadata=metadata))
        assert TokenAnnotationCallback()._extract_model_name(chats.message.response_metadata) == expected_result
