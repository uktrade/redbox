from typing import Any, Dict, List
from unittest.mock import patch

from langchain_core.outputs import Generation
from langchain_core.messages import AIMessage, BaseMessageChunk
from pydantic import BaseModel

from redbox.chains.parser import ClaudeParser, StreamingJsonOutputParser, StreamingPlanner


class TestResponseModel(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = {}


class TestPlannerModel(BaseModel):
    steps: List[Dict[str, str]]


# Helper functions for tests
def create_generation(text: str) -> List[Generation]:
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
            items: List[str] = []

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
        print("very obvious")
        print(parsed)
        assert isinstance(parsed, TestResponseModel)
        assert parsed.answer == "test answer"

    def test_parse_with_error(self):
        parser = ClaudeParser(pydantic_object=TestResponseModel)
        parsed = parser.parse("invalid json")
        assert parsed is None


# StreamingJsonOutputParser Tests
class TestStreamingJsonOutputParser:
    @patch("langchain_core.callbacks.manager.dispatch_custom_event")
    @patch("langchain_core.callbacks.manager.CallbackManager")
    def test_transform_valid_json(self, mock_callback_manager, mock_dispatch):
        mock_callback_manager.return_value.get_current_run.return_value = "mock_parent_run_id"

        parser = StreamingJsonOutputParser(pydantic_schema_object=TestResponseModel)
        chunks = ['{"answer": "test', '", "citations": []}']
        result = list(parser._transform(chunks))

        assert len(result) > 0
        assert isinstance(result[0], TestResponseModel)

    @patch("langchain_core.callbacks.manager.dispatch_custom_event")
    @patch("langchain_core.callbacks.manager.CallbackManager")
    def test_transform_non_json(self, mock_callback_manager, mock_dispatch):
        mock_callback_manager.return_value.get_current_run.return_value = "mock_parent_run_id"
        parser = StreamingJsonOutputParser(pydantic_schema_object=TestResponseModel)
        chunks = ["This is just a plain text response without JSON"]
        result = list(parser._transform(chunks))

        assert len(result) > 0
        assert isinstance(result[0], TestResponseModel)
        mock_dispatch.assert_called

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
