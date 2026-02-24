# _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# sys.path.insert(0, _repo_root)  # for redbox
# sys.path.insert(0, os.path.join(_repo_root, "django_app"))  # for redbox_app

import asyncio
import re
from typing import List, Union, Any, AsyncIterator

from redbox.chains.parser import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

from tests.parser.plot_util import (
    StreamingJsonOutputParserWithMetrics,
    DummySchema,
    generate_large_input,
    aggregate_metrics_with_chunks,
    plot_aggregate_tokens_and_chunks,
)

# - Parser Variants [Async]


class StreamingJsonOriginal(StreamingJsonOutputParserWithMetrics):
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

    async def _atransform(self, input: AsyncIterator[Union[str, BaseMessage]]) -> AsyncIterator[Any]:
        acc_gen: Union[GenerationChunk, ChatGenerationChunk, None] = None
        field_length_at_last_run: int = 0
        parsed = None
        is_parsed = False

        async for chunk in input:
            chunk_gen = self._to_generation_chunk(chunk)

            acc_gen = chunk_gen if acc_gen is None else acc_gen + chunk_gen  # type: ignore[operator]
            if parsed := self.parse_partial_json(acc_gen.text):
                is_parsed = True
                field_content = parsed.get(self.name_of_streamed_field)
                if field_content:
                    if new_tokens := field_content[field_length_at_last_run:]:
                        self.metrics.record_tokens(len(new_tokens))
                        field_length_at_last_run = len(field_content)
                        yield self.pydantic_schema_object.model_validate(parsed)

        if not is_parsed:  # if no tokens were parsed, parse last chunk
            match = re.search(r"(\{)", acc_gen.text, re.DOTALL)
            if not match:  # stream only when text does not contain json brackets to ensure quality of output
                transformed_text = self.answer_str_to_json(acc_gen.text)
                if parsed := self.parse_partial_json(transformed_text):
                    field_content = parsed.get(self.name_of_streamed_field)
                    if field_content:
                        self.metrics.record_tokens(len(field_content))

                        yield self.pydantic_schema_object.model_validate(parsed)

        if parsed:
            yield self.pydantic_schema_object.model_validate(parsed)


class StreamingJsonRefactored(StreamingJsonOutputParserWithMetrics):
    def extract_json(self, text: str) -> str:
        """Extract JSON from text more efficiently."""
        if isinstance(text, list):
            text = text[0].get("text")

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return text  # no JSON found

        return text[start_idx : end_idx + 1]

    async def _atransform(self, input: AsyncIterator[Union[str, BaseMessage]]) -> AsyncIterator[Any]:
        acc_buffer: list[str] = []
        field_length_at_last_run: int = 0
        parsed = None
        is_parsed = False
        seen_json_start = False

        async for chunk in input:
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
                new_tokens = field_content[field_length_at_last_run:]
                self.metrics.record_tokens(len(new_tokens))
                field_length_at_last_run = len(field_content)
                yield self.pydantic_schema_object.model_validate(parsed)

        if not is_parsed:
            acc_text = "".join(acc_buffer)
            if "{" not in acc_text:
                acc_text = self.answer_str_to_json(acc_text)

            if parsed := self.parse_partial_json(acc_text):
                if field_content := parsed.get(self.name_of_streamed_field):
                    self.metrics.record_tokens(len(field_content))
                    yield self.pydantic_schema_object.model_validate(parsed)

        if parsed:
            yield self.pydantic_schema_object.model_validate(parsed)


class StreamingJsonOptimised(StreamingJsonOutputParserWithMetrics):
    def _find_answer_start(self, buffer: str, field_name: str = "answer") -> int:
        key = f'"{field_name}"'
        key_pos = buffer.find(key)
        if key_pos == -1:
            return -1
        colon_pos = buffer.find(":", key_pos + len(key))
        if colon_pos == -1:
            return -1
        quote_pos = buffer.find('"', colon_pos + 1)
        if quote_pos == -1:
            return -1
        return quote_pos + 1

    def _extract_answer_incremental(self, buffer: str, answer_start: int, resume_raw_pos: int) -> tuple[str, int]:
        escape_map = {
            "n": "\n",
            "t": "\t",
            "r": "\r",
            '"': '"',
            "\\": "\\",
            "b": "\b",
            "f": "\f",
        }
        i = resume_raw_pos if resume_raw_pos >= answer_start else answer_start
        result = []

        while i < len(buffer):
            c = buffer[i]
            if c == "\\":
                if i + 1 >= len(buffer):
                    break  # incomplete escape at boundary
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
                            i += 1
                    else:
                        break  # incomplete \uXXXX
                else:
                    i += 1  # unknown escape, drop backslash
                continue
            if c == '"':
                return "".join(result), i
            result.append(c)
            i += 1

        return "".join(result), i

    async def _atransform(self, input: AsyncIterator) -> AsyncIterator[Any]:
        acc_text = ""
        acc_message = None
        field_length_at_last_run = 0
        is_parsed = False
        seen_json_start = False
        answer_start_pos = -1
        last_raw_pos = -1

        async for chunk in input:
            chunk_gen = self._to_generation_chunk(chunk)
            text = chunk_gen.text
            acc_text += text

            if acc_message is None:
                acc_message = chunk
            else:
                try:
                    acc_message = acc_message + chunk
                except Exception:
                    acc_message = chunk

            if not seen_json_start:
                if "{" not in acc_text:
                    continue
                seen_json_start = True

            # Fast path — answer field already located
            if answer_start_pos != -1:
                new_chars, last_raw_pos = self._extract_answer(acc_text, answer_start_pos, last_raw_pos)
                if new_chars:
                    self.metrics.record_tokens(len(new_chars))
                    field_length_at_last_run += len(new_chars)
                continue

            # Slow path — find answer field start
            answer_start_pos = self._find_answer_start(acc_text)
            if answer_start_pos != -1:
                last_raw_pos = answer_start_pos
                new_chars, last_raw_pos = self._extract_answer(acc_text, answer_start_pos, last_raw_pos)
                if new_chars:
                    is_parsed = True
                    self.metrics.record_tokens(len(new_chars))
                    field_length_at_last_run += len(new_chars)
                continue

            # Pre-answer preamble — full parse fallback
            try:
                partial = self.parse_partial_json(acc_text)
            except Exception:
                partial = None

            if partial:
                is_parsed = True
                field_content = partial.get(self.name_of_streamed_field)
                if field_content and len(field_content) > field_length_at_last_run:
                    new_tokens = field_content[field_length_at_last_run:]
                    self.metrics.record_tokens(len(new_tokens))
                    field_length_at_last_run = len(field_content)

        # End of stream — single full parse, single yield with citations
        if not is_parsed and "{" not in acc_text:
            acc_text = self.answer_str_to_json(acc_text)

        final_parsed = self.parse_partial_json(acc_text)
        if final_parsed:
            field_content = final_parsed.get(self.name_of_streamed_field)
            if field_content and field_length_at_last_run == 0:
                self.metrics.record_tokens(len(field_content))
            yield self.pydantic_schema_object.model_validate(final_parsed)

    # async def _atransform(self, input: AsyncIterator) -> AsyncIterator[Any]:
    #     acc_text = ""
    #     acc_message = None
    #     field_length_at_last_run = 0
    #     is_parsed = False
    #     seen_json_start = False
    #     answer_start_pos = -1
    #     last_raw_pos = -1

    #     async for chunk in input:
    #         chunk_gen = self._to_generation_chunk(chunk)
    #         text = chunk_gen.text
    #         acc_text += text

    #         if acc_message is None:
    #             acc_message = chunk
    #         else:
    #             try:
    #                 acc_message = acc_message + chunk
    #             except Exception:
    #                 acc_message = chunk

    #         if not seen_json_start:
    #             if "{" not in acc_text:
    #                 continue
    #             seen_json_start = True

    #         # Fast path — already found answer field
    #         if answer_start_pos != -1:
    #             new_chars, last_raw_pos = self._extract_answer_incremental(acc_text, answer_start_pos, last_raw_pos)
    #             if new_chars:
    #                 self.metrics.record_tokens(len(new_chars))
    #                 field_length_at_last_run += len(new_chars)
    #                 yield self.pydantic_schema_object.model_validate(
    #                     {self.name_of_streamed_field: acc_text[answer_start_pos:last_raw_pos], "citations": []}
    #                 )
    #             continue

    #         # Slow path — find answer field start
    #         answer_start_pos = self._find_answer_start(acc_text)
    #         if answer_start_pos != -1:
    #             last_raw_pos = answer_start_pos
    #             new_chars, last_raw_pos = self._extract_answer_incremental(acc_text, answer_start_pos, last_raw_pos)
    #             if new_chars:
    #                 is_parsed = True
    #                 self.metrics.record_tokens(len(new_chars))
    #                 field_length_at_last_run += len(new_chars)
    #                 yield self.pydantic_schema_object.model_validate(
    #                     {self.name_of_streamed_field: new_chars, "citations": []}
    #                 )
    #             continue

    #         # Pre-answer preamble — full parse fallback
    #         try:
    #             partial = self.parse_partial_json(acc_text)
    #         except Exception:
    #             partial = None

    #         if partial:
    #             is_parsed = True
    #             field_content = partial.get(self.name_of_streamed_field)
    #             if field_content and len(field_content) > field_length_at_last_run:
    #                 new_tokens = field_content[field_length_at_last_run:]
    #                 self.metrics.record_tokens(len(new_tokens))
    #                 field_length_at_last_run = len(field_content)
    #                 yield self.pydantic_schema_object.model_validate(partial)

    #     # End of stream — full parse for citations
    #     if not is_parsed and "{" not in acc_text:
    #         acc_text = self.answer_str_to_json(acc_text)

    #     final_parsed = self.parse_partial_json(acc_text)
    #     if final_parsed:
    #         field_content = final_parsed.get(self.name_of_streamed_field)
    #         if field_content and field_length_at_last_run == 0:
    #             self.metrics.record_tokens(len(field_content))
    #         yield self.pydantic_schema_object.model_validate(final_parsed)


# - Run the async parser


async def to_async_iter(chunks: List[Union[str, BaseMessage]]) -> AsyncIterator[Union[str, BaseMessage]]:
    for chunk in chunks:
        yield chunk
        await asyncio.sleep(0)  # allow event loop to switch if needed


async def benchmark_streaming(
    parser: StreamingJsonOutputParserWithMetrics, input_chunks: List[Union[str, BaseMessage]]
):
    async_chunks = to_async_iter(input_chunks)

    async for parsed in parser._atransform(async_chunks):
        pass

    return parser.metrics


async def benchmark_multiple_runs(parser_class, input_chunks, num_runs: int = 5, **parser_kwargs):
    """Run parser multiple times and collect metrics."""
    all_metrics = []

    for _ in range(num_runs):
        parser = parser_class(**parser_kwargs)
        metrics = await benchmark_streaming(parser, input_chunks)
        all_metrics.append(metrics)

    return all_metrics


# - Run multiple benchmarks and aggregate


async def main():
    input_chunks = generate_large_input(num_chunks=200, chunk_size=10)

    num_runs = 10
    original_runs = await benchmark_multiple_runs(
        StreamingJsonOriginal, input_chunks, num_runs=num_runs, pydantic_schema_object=DummySchema
    )
    refactored_runs = await benchmark_multiple_runs(
        StreamingJsonRefactored, input_chunks, num_runs=num_runs, pydantic_schema_object=DummySchema
    )
    optimised_runs = await benchmark_multiple_runs(
        StreamingJsonOptimised, input_chunks, num_runs=num_runs, pydantic_schema_object=DummySchema
    )

    # Aggregate metrics
    original_agg = aggregate_metrics_with_chunks(original_runs)
    refactored_agg = aggregate_metrics_with_chunks(refactored_runs)
    optimised_agg = aggregate_metrics_with_chunks(optimised_runs)

    # Plot mean +- std.dev
    plot_aggregate_tokens_and_chunks(
        [original_agg, refactored_agg, optimised_agg], ["Original", "Refactored", "Optimised"]
    )


if __name__ == "__main__":
    asyncio.run(main())
