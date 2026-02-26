from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessageChunk
from langchain_core.outputs import Generation
from pydantic import BaseModel

from redbox.chains.parser import ClaudeParser, StreamingJsonOutputParser, StreamingPlanner
from langchain_core.runnables import RunnableLambda
from redbox.chains.runnables import prompt_budget_calculation, truncate_chat_history
from redbox.models.errors import QuestionLengthError
from redbox.transform import bedrock_tokeniser


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


class TestStreamingJsonOutputParserExtraction:
    @pytest.fixture
    def parser(self):
        return StreamingJsonOutputParser(pydantic_schema_object=TestResponseModel)

    # _find_answer_start

    @pytest.mark.parametrize(
        "buffer,expected",
        [
            ('{"answer": "hello world"}', "h"),
            ('{"citations": [], "answer": "hello"}', "h"),
            ('{"answer" :  "hello"}', "h"),  # whitespace around colon
        ],
    )
    def test_find_answer_start_found(self, parser, buffer, expected):
        pos = parser._find_answer_start(buffer)
        assert pos != -1
        assert buffer[pos] == expected

    @pytest.mark.parametrize(
        "buffer",
        [
            '{"ans',  # key not present
            '{"answer"',  # missing colon
            '{"answer":',  # missing value quote
        ],
    )
    def test_find_answer_start_not_found(self, parser, buffer):
        assert parser._find_answer_start(buffer) == -1

    def test_find_answer_start_custom_field(self):
        parser = StreamingJsonOutputParser(pydantic_schema_object=TestResponseModel, name_of_streamed_field="summary")
        buffer = '{"summary": "hello"}'
        pos = parser._find_answer_start(buffer)
        assert pos != -1
        assert buffer[pos] == "h"

    # _extract_answer basic

    def test_extract_answer_complete(self, parser):
        buffer = '{"answer": "hello world", "citations": []}'
        start = parser._find_answer_start(buffer)
        text, pos = parser._extract_answer(buffer, start, start)
        assert text == "hello world"
        assert buffer[pos] == '"'

    def test_extract_answer_empty_string(self, parser):
        buffer = '{"answer": "", "citations": []}'
        start = parser._find_answer_start(buffer)
        text, _ = parser._extract_answer(buffer, start, start)
        assert text == ""

    def test_extract_answer_incomplete(self, parser):
        buffer = '{"answer": "hello wor'
        start = parser._find_answer_start(buffer)
        text, pos = parser._extract_answer(buffer, start, start)
        assert text == "hello wor"
        assert pos == len(buffer)

    def test_extract_answer_incremental_resume(self, parser):
        buffer1 = '{"answer": "hello'
        buffer2 = '{"answer": "hello world"}'
        start = parser._find_answer_start(buffer1)

        text1, pos1 = parser._extract_answer(buffer1, start, start)
        assert text1 == "hello"

        text2, _ = parser._extract_answer(buffer2, start, pos1)
        assert text2 == " world"

    def test_extract_answer_incremental_resume_no_new_chars(self, parser):
        buffer = '{"answer": "hello'
        start = parser._find_answer_start(buffer)
        _, pos1 = parser._extract_answer(buffer, start, start)
        text2, _ = parser._extract_answer(buffer, start, pos1)
        assert text2 == ""

    # escape sequences

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (r"line1\nline2", "line1\nline2"),
            (r"col1\tcol2", "col1\tcol2"),
            (r"line1\rline2", "line1\rline2"),
            (r"path\\to\\file", "path\\to\\file"),
            (r"say \"hello\"", 'say "hello"'),
            (r"caf\u00e9", "café"),
            (r"## Header\n\n* item1\n* item2\n\ncaf\u00e9", "## Header\n\n* item1\n* item2\n\ncafé"),
        ],
    )
    def test_extract_answer_escape_sequences(self, parser, raw, expected):
        buffer = '{"answer": "' + raw + '"}'
        start = parser._find_answer_start(buffer)
        text, _ = parser._extract_answer(buffer, start, start)
        assert text == expected

    def test_extract_answer_escape_unicode_invalid(self, parser):
        """Invalid unicode escape should not raise — backslash dropped."""
        buffer = '{"answer": "bad\\uXXXX end"}'
        start = parser._find_answer_start(buffer)
        text, _ = parser._extract_answer(buffer, start, start)
        assert "bad" in text
        assert "end" in text

    def test_extract_answer_no_forward_slash_escape(self, parser):
        """\\/ should not produce phantom slashes — backslash dropped, slash kept."""
        buffer = '{"answer": "http:\\/\\/example.com"}'
        start = parser._find_answer_start(buffer)
        text, _ = parser._extract_answer(buffer, start, start)
        assert text == "http://example.com"
        assert "\\" not in text

    # chunk boundary handling

    @pytest.mark.parametrize(
        "partial_buffer,expected_text,expected_pos_offset",
        [
            ('{"answer": "line1\\', "line1", -1),  # backslash at end — parked before it
            ('{"answer": "caf\\u00', "caf", None),  # incomplete \uXXXX
        ],
    )
    def test_extract_answer_boundary_pauses(self, parser, partial_buffer, expected_text, expected_pos_offset):
        start = parser._find_answer_start(partial_buffer)
        text, pos = parser._extract_answer(partial_buffer, start, start)
        assert text == expected_text
        if expected_pos_offset is not None:
            assert pos == len(partial_buffer) + expected_pos_offset
        else:
            assert pos < len(partial_buffer)

    def test_extract_answer_boundary_escape_then_resume(self, parser):
        """Chunk split on \\n boundary produces correct newline after resume."""
        buffer1 = '{"answer": "line1\\'
        buffer2 = '{"answer": "line1\\nline2"}'

        start1 = parser._find_answer_start(buffer1)
        text1, _ = parser._extract_answer(buffer1, start1, start1)
        assert text1 == "line1"

        start2 = parser._find_answer_start(buffer2)
        text2, _ = parser._extract_answer(buffer2, start2, start2)
        assert text2 == "line1\nline2"

    def test_extract_answer_boundary_no_literal_n(self, parser):
        """Escaped \\n must produce newline not literal 'n'."""
        buffer = '{"answer": "line1\\nline2"}'
        start = parser._find_answer_start(buffer)
        text, _ = parser._extract_answer(buffer, start, start)
        assert text == "line1\nline2"
        assert "\\n" not in text


@pytest.mark.parametrize("exceeding_budget, prompts_budget", [(True, 1000000), (False, 1000)])
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
