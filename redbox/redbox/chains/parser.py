from __future__ import annotations

import ast
import json
import logging
import random
import re
from collections.abc import AsyncIterator
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
from pydantic import BaseModel, ValidationError

from redbox.models.graph import RedboxEventType

log = logging.getLogger(__name__)


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

    def correct_json(self, text: str):
        top_level_key = list(self.pydantic_object.__fields__.keys())  # get the name of top level key
        if len(top_level_key) == 1:  # if there is only 1 top level key
            if isinstance(
                self.pydantic_object.model_json_schema()["properties"].get(top_level_key[0])["default"], list
            ):  # if the value of the top level key is a list
                text = ast.literal_eval(text)  # convert string to list
                text_json = json.dumps({top_level_key[0]: text})
            return text_json
        else:
            return text

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
        try:
            text_json = self.extract_json(text)
            return self.pydantic_object.model_validate(parse_json_markdown(text_json))
        except ValidationError:
            try:
                text_json = self.correct_json(text)
                return self.pydantic_object.model_validate(parse_json_markdown(text_json))
            except ValidationError as e:
                if partial:
                    print(e)
                    return None
                else:
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

        except Exception as e:
            print(e)
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
    # sub_streamed_field: str = None
    pydantic_schema_object: type[BaseModel]

    # Track parser state across chunks
    _answer_start_pos: int = -1  # position in buffer after opening quote of answer value
    _in_answer_field: bool = False

    def extract_json(self, text: str) -> str:
        """Extract JSON from text more efficiently."""
        if isinstance(text, list):
            text = text[0].get("text")

        # Find first { and last } instead of using a greedy regex
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return text  # no JSON found

        return text[start_idx : end_idx + 1]

    def answer_str_to_json(self, text: str):
        text_json = json.dumps({"answer": text, "citations": []})
        return text_json

    def parse_partial_json(self, text: str):
        try:
            json_text = self.extract_json(text)
            return parse_json_markdown(json_text)
        except json.JSONDecodeError as e:
            if ": None" in json_text:
                json_text = json_text.replace(": None", ": null")
                return parse_json_markdown(json_text)
            else:
                print(e)
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

    def _find_answer_start(self, buffer: str) -> int:
        """Find the position right after the opening quote of the answer value."""
        key = f'"{self.name_of_streamed_field}"'
        key_pos = buffer.find(key)
        if key_pos == -1:
            return -1
        # Find the colon after the key
        colon_pos = buffer.find(":", key_pos + len(key))
        if colon_pos == -1:
            return -1
        # Find the opening quote of the value
        quote_pos = buffer.find('"', colon_pos + 1)
        if quote_pos == -1:
            return -1
        return quote_pos + 1  # position of first char of answer content

    def _extract_answer_fast(self, buffer: str, answer_start: int) -> str | None:
        """
        Scan forward from answer_start, stop at unescaped closing quote.
        Returns the unescaped string content.
        """
        escape_map = {
            "n": "\n",
            "t": "\t",
            "r": "\r",
            '"': '"',
            "\\": "\\",
            "b": "\b",
            "f": "\f",
            # '/' intentionally omitted — \/ is optional in JSON and causes false positives
        }

        i = answer_start
        result = []
        while i < len(buffer):
            c = buffer[i]

            if c == "\\":
                if i + 1 >= len(buffer):
                    # Buffer ends on backslash — mid-escape, stop here
                    # Don't consume it, let next chunk re-process from this point
                    break
                next_c = buffer[i + 1]
                if next_c in escape_map:
                    result.append(escape_map[next_c])
                    i += 2
                elif next_c == "u":
                    if i + 5 < len(buffer):
                        try:
                            codepoint = int(buffer[i + 2 : i + 6], 16)
                            result.append(chr(codepoint))
                            i += 6
                        except ValueError:
                            # Not a valid unicode escape, skip the backslash
                            i += 1
                    else:
                        # Incomplete \uXXXX — wait for more chunks
                        break
                else:
                    # Unknown escape — skip the backslash, keep the char
                    i += 1
                continue

            if c == '"':
                return "".join(result)

            result.append(c)
            i += 1

        return "".join(result)

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Any]:
        acc_buffer: list[str] = []
        field_length_at_last_run: int = 0
        parsed = None
        is_parsed = False
        seen_json_start = False

        # start_time = time.perf_counter()
        # last_yield_time = start_time

        for chunk in input:
            chunk_gen = self._to_generation_chunk(chunk)
            text = chunk_gen.text
            acc_buffer.append(text)

            if not seen_json_start:
                if "{" not in text:
                    continue
                seen_json_start = True

            acc_text = "".join(acc_buffer)

            partial = self.parse_partial_json(acc_text)
            if not partial:
                continue

            is_parsed = True
            parsed = partial

            field_content = parsed.get(self.name_of_streamed_field)
            if not field_content:
                continue

            if len(field_content) > field_length_at_last_run:
                # now = time.perf_counter()
                # delta_ms = (now - last_yield_time) * 1000
                # total_ms = (now - start_time) * 1000
                # last_yield_time = now

                new_tokens = field_content[field_length_at_last_run:]
                dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens)  # + f" ({delta_ms}:.3f)")
                field_length_at_last_run = len(field_content)
                yield self.pydantic_schema_object.model_validate(parsed)

        if not is_parsed:
            acc_text = "".join(acc_buffer)
            if "{" not in acc_text:
                acc_text = self.answer_str_to_json(acc_text)

            if parsed := self.parse_partial_json(acc_text):
                if field_content := parsed.get(self.name_of_streamed_field):
                    dispatch_custom_event(RedboxEventType.response_tokens, data=field_content)
                    yield self.pydantic_schema_object.model_validate(parsed)

        if parsed:
            yield self.pydantic_schema_object.model_validate(parsed)

    async def _atransform(self, input: AsyncIterator) -> AsyncIterator[Any]:
        acc_text = ""
        acc_message = None  # accumulated full message object
        field_length_at_last_run = 0
        is_parsed = False
        seen_json_start = False
        answer_start_pos = -1

        # start_time = time.perf_counter()
        # last_yield_time = start_time

        async for chunk in input:
            chunk_gen = self._to_generation_chunk(chunk)
            text = chunk_gen.text
            acc_text += text

            # Accumulate the message object so tool_calls etc. are preserved
            if acc_message is None:
                acc_message = chunk
            else:
                try:
                    acc_message = acc_message + chunk  # LangChain chunks support addition
                except Exception:
                    acc_message = chunk  # fallback if addition not supported

            def _make_yield(parsed_response=None):
                return {
                    "raw_response": acc_message,
                    "parsed_response": parsed_response,
                }

            if not seen_json_start:
                if "{" not in acc_text:
                    continue
                seen_json_start = True

            # Fast path — answer field already located
            if answer_start_pos != -1:
                field_content = self._extract_answer_fast(acc_text, answer_start_pos)
                if field_content and len(field_content) > field_length_at_last_run:
                    # now = time.perf_counter()
                    # delta_ms = (now - last_yield_time) * 1000
                    # last_yield_time = now
                    new_tokens = field_content[field_length_at_last_run:]
                    dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens)  # + f" ({delta_ms})")
                    field_length_at_last_run = len(field_content)
                    yield _make_yield()
                continue

            # Slow path — find answer start
            answer_start_pos = self._find_answer_start(acc_text)
            if answer_start_pos != -1:
                field_content = self._extract_answer_fast(acc_text, answer_start_pos)
                if field_content:
                    is_parsed = True
                    # now = time.perf_counter()
                    # delta_ms = (now - last_yield_time) * 1000
                    # last_yield_time = now
                    dispatch_custom_event(RedboxEventType.response_tokens, data=field_content)  # + f" ({delta_ms})")
                    field_length_at_last_run = len(field_content)
                    yield _make_yield()
                continue

            # Pre-answer: fall back to full parse
            try:
                partial = self.parse_partial_json(acc_text)
            except Exception:
                partial = None

            if partial:
                is_parsed = True
                field_content = partial.get(self.name_of_streamed_field)
                if field_content and len(field_content) > field_length_at_last_run:
                    # now = time.perf_counter()
                    # delta_ms = (now - last_yield_time) * 1000
                    # last_yield_time = now
                    new_tokens = field_content[field_length_at_last_run:]
                    dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens)  # + f" ({delta_ms})")
                    field_length_at_last_run = len(field_content)
                    yield _make_yield(self.pydantic_schema_object.model_validate(partial))

        # End of stream — full parse for citations
        if not is_parsed:
            if "{" not in acc_text:
                acc_text = self.answer_str_to_json(acc_text)

        final_parsed = self.parse_partial_json(acc_text)
        if final_parsed:
            field_content = final_parsed.get(self.name_of_streamed_field)
            if field_content and field_length_at_last_run == 0:
                dispatch_custom_event(RedboxEventType.response_tokens, data=field_content)
            yield {
                "raw_response": acc_message,  # full accumulated message with tool_calls intact
                "parsed_response": self.pydantic_schema_object.model_validate(final_parsed),
            }

    # async def _atransform(self, input: AsyncIterator) -> AsyncIterator[Any]:
    #     acc_text = ""
    #     field_length_at_last_run = 0
    #     parsed = None
    #     is_parsed = False
    #     seen_json_start = False
    #     answer_start_pos = -1  # cached position, computed once

    #     start_time = time.perf_counter()
    #     last_yield_time = start_time

    #     async for chunk in input:
    #         chunk_gen = self._to_generation_chunk(chunk)
    #         text = chunk_gen.text
    #         acc_text += text

    #         if not seen_json_start:
    #             if "{" not in acc_text:
    #                 continue
    #             seen_json_start = True

    #         # --- Fast path: if we already know where the answer starts ---
    #         if answer_start_pos != -1:
    #             field_content = self._extract_answer_fast(acc_text, answer_start_pos)

    #             if field_content and len(field_content) > field_length_at_last_run:
    #                 now = time.perf_counter()
    #                 delta_ms = (now - last_yield_time) * 1000
    #                 last_yield_time = now
    #                 new_tokens = field_content[field_length_at_last_run:]
    #                 dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens) # + f" ({delta_ms})")
    #                 field_length_at_last_run = len(field_content)

    #                 # Only do full parse occasionally (e.g. every 50 chars of new content)
    #                 # to get structured output, not on every chunk
    #                 yield {
    #                     "raw_response": acc_text,
    #                     "parsed_response": None,  # defer expensive parse to end
    #                 }
    #             continue  # Skip expensive full parse below

    #         # --- Slow path: haven't found answer start yet ---
    #         # Try to find the answer field start position
    #         answer_start_pos = self._find_answer_start(acc_text)
    #         if answer_start_pos != -1:
    #             # Found it — extract immediately
    #             field_content = self._extract_answer_fast(acc_text, answer_start_pos)
    #             if field_content:
    #                 is_parsed = True
    #                 now = time.perf_counter()
    #                 delta_ms = (now - last_yield_time) * 1000
    #                 last_yield_time = now
    #                 dispatch_custom_event(RedboxEventType.response_tokens, data=field_content) # + f" ({delta_ms})")
    #                 field_length_at_last_run = len(field_content)
    #                 yield {
    #                     "raw_response": acc_text,
    #                     "parsed_response": None,
    #                 }
    #             continue

    #         # Still haven't found answer key — try full parse (pre-answer preamble)
    #         try:
    #             partial = self.parse_partial_json(acc_text)
    #         except Exception:
    #             partial = None

    #         if partial:
    #             is_parsed = True
    #             parsed = partial
    #             field_content = parsed.get(self.name_of_streamed_field)
    #             if field_content and len(field_content) > field_length_at_last_run:
    #                 now = time.perf_counter()
    #                 delta_ms = (now - last_yield_time) * 1000
    #                 last_yield_time = now
    #                 new_tokens = field_content[field_length_at_last_run:]
    #                 dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens) # + f" ({delta_ms})")
    #                 field_length_at_last_run = len(field_content)
    #                 yield {
    #                     "raw_response": acc_text,
    #                     "parsed_response": self.pydantic_schema_object.model_validate(parsed),
    #                 }

    #     # --- End of stream: do ONE full parse for final structured output ---
    #     if not is_parsed:
    #         if "{" not in acc_text:
    #             acc_text = self.answer_str_to_json(acc_text)

    #     final_parsed = self.parse_partial_json(acc_text)
    #     if final_parsed:
    #         field_content = final_parsed.get(self.name_of_streamed_field)
    #         if field_content and field_length_at_last_run == 0:
    #             dispatch_custom_event(RedboxEventType.response_tokens, data=field_content)
    #         yield {
    #             "raw_response": acc_text,
    #             "parsed_response": self.pydantic_schema_object.model_validate(final_parsed),
    #         }

    # async def _atransform(self, input: AsyncIterator[Union[str, BaseMessage]]) -> AsyncIterator[Any]:
    #     # acc_buffer: list[str] = []
    #     acc_text = ""
    #     field_length_at_last_run: int = 0
    #     parsed = None
    #     is_parsed = False
    #     seen_json_start = False

    #     start_time = time.perf_counter()
    #     last_yield_time = start_time

    #     async for chunk in input:
    #         chunk_gen = self._to_generation_chunk(chunk)
    #         text = chunk_gen.text
    #         # acc_buffer.append(text)
    #         acc_text += text
    #         #

    #         yield {
    #             "raw_response": acc_text,
    #             "parsed_response": None,  # will be populated at the end
    #         }

    #         if not seen_json_start:
    #             if "{" not in text:
    #                 continue
    #             seen_json_start = True

    #         # acc_text = "".join(acc_buffer)

    #         try:
    #             partial = self.parse_partial_json(acc_text)
    #         except:
    #             partial = None

    #         if not partial:
    #             continue

    #         is_parsed = True
    #         parsed = partial

    #         field_content = parsed.get(self.name_of_streamed_field)
    #         if not field_content:
    #             continue

    #         if len(field_content) > field_length_at_last_run:
    #             now = time.perf_counter()
    #             delta_ms = (now - last_yield_time) * 1000
    #             total_ms = (now - start_time) * 1000
    #             last_yield_time = now

    #             new_tokens = field_content[field_length_at_last_run:]
    #             dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens  + f" ({delta_ms})")
    #             field_length_at_last_run = len(field_content)
    #             # yield self.pydantic_schema_object.model_validate(parsed)
    #             yield {
    #                 "raw_response": acc_text,
    #                 "parsed_response": self.pydantic_schema_object.model_validate(parsed),
    #             }

    #     if not is_parsed:
    #         # acc_text = acc_text
    #         if "{" not in acc_text:
    #             acc_text = self.answer_str_to_json(acc_text)

    #         if parsed := self.parse_partial_json(acc_text):
    #             if field_content := parsed.get(self.name_of_streamed_field):
    #                 dispatch_custom_event(RedboxEventType.response_tokens, data=field_content)
    #                 yield self.pydantic_schema_object.model_validate(parsed)

    #     if parsed:
    #         yield self.pydantic_schema_object.model_validate(parsed)

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


class StreamingPlanner(StreamingJsonOutputParser):
    sub_streamed_field: str = None
    suffix_texts: list = [""]
    prefix_texts: list = [""]

    def parse_partial_json(self, text: str):
        try:
            text = super().extract_json(text)
            return parse_json_markdown(text)
        except json.JSONDecodeError:
            return None

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Any]:
        acc_gen: Union[GenerationChunk, ChatGenerationChunk, None] = None
        field_length_at_last_run: int = 0
        item_count: int = 0
        parsed = None
        for chunk in input:
            chunk_gen = self._to_generation_chunk(chunk)
            acc_gen = chunk_gen if acc_gen is None else acc_gen + chunk_gen  # type: ignore[operator]

            if parsed := self.parse_partial_json(acc_gen.text):
                if self.sub_streamed_field:
                    try:
                        item = parsed.get(self.name_of_streamed_field)[item_count]
                        if field_content := item.get(self.sub_streamed_field):
                            if new_tokens := field_content[field_length_at_last_run:]:
                                if (item_count == 0) and (field_length_at_last_run == 0):
                                    dispatch_custom_event(
                                        RedboxEventType.response_tokens,
                                        data=f"{random.choice(self.prefix_texts)}\n\n1. ",
                                    )
                                elif (item_count > 0) and (field_length_at_last_run == 0):
                                    dispatch_custom_event(RedboxEventType.response_tokens, data=f"{item_count + 1}. ")
                                dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens)
                                field_length_at_last_run = len(field_content)
                                yield self.pydantic_schema_object.model_validate(parsed)
                            else:
                                item_count += 1
                                field_length_at_last_run = 0
                                dispatch_custom_event(RedboxEventType.response_tokens, data="\n\n")

                    except (IndexError, TypeError):
                        item = []
                else:
                    if field_content := parsed.get(self.name_of_streamed_field):
                        if new_tokens := field_content[field_length_at_last_run:]:
                            dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens)
                            field_length_at_last_run = len(field_content)
                            yield self.pydantic_schema_object.model_validate(parsed)

        if not (self.sub_streamed_field):
            if parsed:
                yield self.pydantic_schema_object.model_validate(parsed)

        # adding suffix here
        dispatch_custom_event(RedboxEventType.response_tokens, data=random.choice(self.suffix_texts))

    async def _atransform(self, input: AsyncIterator[Union[str, BaseMessage]]) -> AsyncIterator[Any]:
        self._transform(input)
