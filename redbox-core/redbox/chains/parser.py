from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from json import JSONDecodeError
from typing import Any, Iterator, List, Optional, Type, Union

import jsonpatch  # type: ignore[import]
import pydantic  # pydantic: ignore
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.output_parsers.format_instructions import JSON_FORMAT_INSTRUCTIONS
from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser
from langchain_core.outputs import ChatGenerationChunk, Generation, GenerationChunk
from langchain_core.utils.json import parse_json_markdown
from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION
from pydantic import BaseModel

from redbox.models.graph import RedboxEventType


class ClaudeParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse the output of an Claude LLM call to a pydantic object.

    When used in streaming mode, it will yield partial JSON objects containing
    all the keys that have been returned so far.

    In streaming, if `diff` is set to `True`, yields JSONPatch operations
    describing the difference between the previous and the current object.
    """

    pydantic_object: Optional[Type] = None  # type: ignore
    """The Pydantic object to use for validation.
    If None, no validation is performed."""

    def _diff(self, prev: Optional[Any], next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    def _get_schema(self, pydantic_object: Type) -> dict[str, Any]:
        if PYDANTIC_MAJOR_VERSION == 2:
            if issubclass(pydantic_object, pydantic.BaseModel):
                return pydantic_object.model_json_schema()
            elif issubclass(pydantic_object, pydantic.v1.BaseModel):
                return pydantic_object.schema()
        return pydantic_object.schema()

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a pydantic object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.
                If True, the output will be a JSON object containing
                all the keys that have been returned so far.
                If False, the output will be the full JSON object.
                Default is False.

        Returns:
            The parsed pydantic object.

        Raises:
            OutputParserException: If the output is not valid JSON.
        """
        text = result[0].text
        text = text.strip()
        text = self.extract_json(text)
        if partial:
            try:
                return self.pydantic_object.model_validate(parse_json_markdown(text))
            except JSONDecodeError:
                return None
        else:
            try:
                return self.pydantic_object.model_validate(parse_json_markdown(text))
            except JSONDecodeError as e:
                msg = f"Invalid json output: {text}"
                raise OutputParserException(msg, llm_output=text) from e

    def extract_json(self, text):
        if isinstance(text, list):
            text = text[0].get("text")

        try:
            # Find content between first { and last }
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if not match:
                return text

            json_str = match.group(1)

            return json_str

        except Exception as e:
            print(f"Error processing JSON: {e}")
            return text

    def parse(self, text: str) -> Any:
        """Parse the output of an LLM call to a pydantic object.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed pydantic object.
        """
        try:
            return self.parse_result([Generation(text=text)])

        except json.JSONDecodeError:
            return None

    def get_format_instructions(self) -> str:
        """Return the format instructions for the JSON output.

        Returns:
            The format instructions for the JSON output.
        """
        if self.pydantic_object is None:
            return "Return a JSON object."
        else:
            # Copy schema to avoid altering original Pydantic schema.
            schema = {k: v for k, v in self._get_schema(self.pydantic_object).items()}

            # Remove extraneous fields.
            reduced_schema = schema
            if "title" in reduced_schema:
                del reduced_schema["title"]
            if "type" in reduced_schema:
                del reduced_schema["type"]
            # Ensure json in context is well-formed with double quotes.
            schema_str = json.dumps(reduced_schema)
            return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)


class StreamingJsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    """
    A Pydantic output parser which emits token events for a given field from the JSON intermediate stage.
    This allows streaming a field (answer) while maintaining the Pydantic object through the pipeline for tracking
    citations.

    This class is mostly based on existing implementations in BaseCumulativeTransformOutputParser and JsonOutputParser.
    This custom parser is here to allow emitting custom events and maintaining state for each parse, every invocation of the
    parser tracks the current length of the answer field to allow emitting delta token events.
    """

    diff: bool = False  # Ignored
    name_of_streamed_field: str = "answer"
    pydantic_schema_object: type[BaseModel]

    def extract_json(self, text):
        if isinstance(text, list):
            text = text[0].get("text")

        try:
            # Find content between first { and last }
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if not match:
                return text

            json_str = match.group(1)

            return json_str

        except Exception as e:
            print(f"Error processing JSON: {e}")
            return text

    def parse_partial_json(self, text: str):
        try:
            text = self.extract_json(text)
            return parse_json_markdown(text)
        except json.JSONDecodeError:
            return None

    def _to_generation_chunk(self, chunk: Union[str, BaseMessage]):
        chunk_gen: Union[GenerationChunk, ChatGenerationChunk]
        if isinstance(chunk, BaseMessageChunk):
            chunk_gen = ChatGenerationChunk(message=chunk)
        elif isinstance(chunk, BaseMessage):
            chunk_gen = ChatGenerationChunk(message=BaseMessageChunk(**chunk.model_dump()))
        else:
            chunk_gen = GenerationChunk(text=chunk)
        return chunk_gen

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Any]:
        acc_gen: Union[GenerationChunk, ChatGenerationChunk, None] = None
        field_length_at_last_run: int = 0
        parsed = None
        for chunk in input:
            chunk_gen = self._to_generation_chunk(chunk)
            acc_gen = chunk_gen if acc_gen is None else acc_gen + chunk_gen  # type: ignore[operator]

            if parsed := self.parse_partial_json(acc_gen.text):
                if field_content := parsed.get(self.name_of_streamed_field):
                    if new_tokens := field_content[field_length_at_last_run:]:
                        dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens)
                        field_length_at_last_run = len(field_content)
                        yield self.pydantic_schema_object.model_validate(parsed)
        if parsed:
            yield self.pydantic_schema_object.model_validate(parsed)

    async def _atransform(self, input: AsyncIterator[Union[str, BaseMessage]]) -> AsyncIterator[Any]:
        acc_gen: Union[GenerationChunk, ChatGenerationChunk, None] = None
        field_length_at_last_run: int = 0
        parsed = None
        async for chunk in input:
            chunk_gen = self._to_generation_chunk(chunk)
            acc_gen = chunk_gen if acc_gen is None else acc_gen + chunk_gen  # type: ignore[operator]

            if parsed := self.parse_partial_json(acc_gen.text):
                if field_content := parsed.get(self.name_of_streamed_field):
                    if new_tokens := field_content[field_length_at_last_run:]:
                        dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens)
                        field_length_at_last_run = len(field_content)
                        yield self.pydantic_schema_object.model_validate(parsed)
        if parsed:
            yield self.pydantic_schema_object.model_validate(parsed)

    @property
    def _type(self) -> str:
        return "streaming_json_output_parser"

    def get_format_instructions(self) -> str:
        """Return the format instructions for the JSON output.

        Returns:
            The format instructions for the JSON output.
        """
        if self.pydantic_schema_object is None:
            return "Return a JSON object."
        else:
            # Copy schema to avoid altering original Pydantic schema.
            schema = dict(self.pydantic_schema_object.model_json_schema().items())

            # Remove extraneous fields.
            reduced_schema = schema
            if "title" in reduced_schema:
                del reduced_schema["title"]
            if "type" in reduced_schema:
                del reduced_schema["type"]
            # Ensure json in context is well-formed with double quotes.
            schema_str = json.dumps(reduced_schema)
            return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    def parse(self, text: str) -> Any:
        return super().parse(text)
