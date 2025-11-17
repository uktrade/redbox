from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessageChunk
from langchain_core.outputs import Generation
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from redbox.chains.parser import ClaudeParser, StreamingJsonOutputParser, StreamingPlanner
from redbox.chains.runnables import prompt_budget_calculation, truncate_chat_history
from redbox.models.errors import QuestionLengthError
from redbox.transform import bedrock_tokeniser


class TestResponseModel(BaseModel):
    answer: str
    citations: list[dict[str, Any]] = {}


class TestPlannerModel(BaseModel):
    steps: list[dict[str, str]]


# Helper functions for tests
def create_generation(text: str) -> list[Generation]:
    return [Generation(text=text)]


def create_chat_generation_chunk(text: str) -> BaseMessageChunk:
    return AIMessage(content=text)


# ClaudeParser Tests
class TestClaudeParser:
    def test_extract_json(self):
        parser = ClaudeParser(pydantic_object=TestResponseModel)
        text = 'Some text before the JSON {"answer": "test answer", "citations": []} and some text after.'
        extracted = parser.extract_json(text)
        assert extracted == '{"answer": "test answer", "citations": []}'

    def test_extract_json_with_list_input(self):
        parser = ClaudeParser(pydantic_object=TestResponseModel)
        text_list = [{"text": 'Some text {"answer": "test answer", "citations": []} more text'}]
        extracted = parser.extract_json(text_list)
        assert extracted == '{"answer": "test answer", "citations": []}'

    def test_parse_result_valid_json(self):
        parser = ClaudeParser(pydantic_object=TestResponseModel)
        result = create_generation('{"answer": "test answer", "citations": []}')
        parsed = parser.parse_result(result)
        assert isinstance(parsed, TestResponseModel)
        assert parsed.answer == "test answer"
        assert parsed.citations == []

    def test_parse_result_invalid_json_partial_true(self):
        parser = ClaudeParser(pydantic_object=TestResponseModel)
        result = create_generation('{"invalid": "json"}')
        parsed = parser.parse_result(result, partial=True)
        assert parsed is None

    def test_parse_result_with_correction(self):
        class ListModel(BaseModel):
            items: list[str] = []

        parser = ClaudeParser(pydantic_object=ListModel)
        result = create_generation('["item1", "item2"]')
        parsed = parser.parse_result(result)
        assert isinstance(parsed, ListModel)
        assert parsed.items == ["item1", "item2"]

    def test_get_format_instructions(self):
        parser = ClaudeParser(pydantic_object=TestResponseModel)
        instructions = parser.get_format_instructions()
        assert "The output should be formatted as a JSON instance" in instructions
        assert "answer" in instructions
        assert "citations" in instructions

    def test_parse_method(self):
        parser = ClaudeParser(pydantic_object=TestResponseModel)
        parsed = parser.parse('{"answer": "test answer", "citations": []}')
        assert isinstance(parsed, TestResponseModel)
        assert parsed.answer == "test answer"

    def test_parse_with_error(self):
        parser = ClaudeParser(pydantic_object=TestResponseModel)
        parsed = parser.parse("invalid json")
        assert parsed is None


# StreamingJsonOutputParser Tests
class TestStreamingJsonOutputParser:
    def test_transform_valid_json(self):
        parser = StreamingJsonOutputParser(pydantic_schema_object=TestResponseModel)

        def chunk_generator(input_data):
            yield '{"answer": "test'
            yield '", "citations": []}'

        chain = RunnableLambda(chunk_generator) | parser

        result = list(chain.stream(None))

        assert len(result) > 0
        assert isinstance(result[0], TestResponseModel)

    def test_transform_non_json(self):
        parser = StreamingJsonOutputParser(pydantic_schema_object=TestResponseModel)

        def chunk_generator(input_data):
            yield "This is just a plain text response without JSON"

        chain = RunnableLambda(chunk_generator) | parser
        result = list(chain.stream(None))

        assert len(result) > 0
        assert isinstance(result[0], TestResponseModel)
        assert result[0].answer == "This is just a plain text response without JSON"
        assert result[0].citations == []

    @patch("langchain_core.callbacks.manager.dispatch_custom_event")
    def test_parse_partial_json(self, mock_dispatch):
        parser = StreamingJsonOutputParser(pydantic_schema_object=TestResponseModel)

        # Test with valid JSON
        valid_json = '{"answer": "test", "citations": []}'
        parsed = parser.parse_partial_json(valid_json)
        assert parsed is not None
        assert parsed.get("answer") == "test"

        # Test with invalid JSON
        invalid_json = "not a json"
        parsed = parser.parse_partial_json(invalid_json)
        assert parsed is None

        # Test with None values that need conversion
        none_json = '{"answer": "test", "citations": None}'
        parsed = parser.parse_partial_json(none_json)
        assert parsed is not None
        assert parsed.get("answer") == "test"

    def test_get_format_instructions(self):
        parser = StreamingJsonOutputParser(pydantic_schema_object=TestResponseModel)
        instructions = parser.get_format_instructions()
        assert "The output should be formatted as a JSON instance that conforms" in instructions
        assert "answer" in instructions
        assert "citations" in instructions

    def test_parse_partial_streaming_json(self):
        parser = StreamingPlanner(pydantic_schema_object=TestResponseModel, name_of_streamed_field="answer")

        # Test with valid JSON
        valid_json = '{"answer": "test", "citations": []}'
        parsed = parser.parse_partial_json(valid_json)
        assert parsed is not None
        assert parsed.get("answer") == "test"

        # Test with invalid JSON
        invalid_json = "not a json"
        parsed = parser.parse_partial_json(invalid_json)
        assert parsed is None


@pytest.mark.parametrize(("exceeding_budget", "prompts_budget"), [(True, 1000000), (False, 1000)])
def test_prompt_budget_calculation(fake_state, exceeding_budget, prompts_budget):
    if not exceeding_budget:
        response = prompt_budget_calculation(state=fake_state, prompts_budget=prompts_budget)
        assert isinstance(response, int)
    else:
        with pytest.raises(QuestionLengthError):
            prompt_budget_calculation(state=fake_state, prompts_budget=prompts_budget)


def test_truncate_chat_history(fake_state):
    # check we got chat history back
    tokeniser = bedrock_tokeniser
    prompts_budget = tokeniser("This is a fake system prompt")
    chat_history = truncate_chat_history(state=fake_state, prompts_budget=prompts_budget, tokeniser=tokeniser)
    assert len(chat_history) > 0
