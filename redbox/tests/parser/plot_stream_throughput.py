import re
from typing import List, Union, Iterator, Any

from redbox.chains.parser import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

from tests.parser.plot_util import (
    StreamingJsonOutputParserWithMetrics,
    DummySchema,
    generate_large_input,
    aggregate_metrics_with_chunks,
    plot_aggregate_tokens_and_chunks,
)


# - Parser Variants [Sync]


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

    def _transform(self, input: Iterator[Union[str, BaseMessage]]):
        acc_gen: Union[GenerationChunk, ChatGenerationChunk, None] = None
        field_length_at_last_run: int = 0
        parsed = None
        is_parsed = False
        for chunk in input:
            chunk_gen = self._to_generation_chunk(chunk)
            acc_gen = chunk_gen if acc_gen is None else acc_gen + chunk_gen  # type: ignore[operator]
            if parsed := self.parse_partial_json(acc_gen.text):
                is_parsed = True
                field_content = parsed.get(self.name_of_streamed_field)
                if field_content:
                    if new_tokens := field_content[field_length_at_last_run:]:
                        # parser_module.dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens)
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
                        # parser_module.dispatch_custom_event(RedboxEventType.response_tokens, data=field_content)
                        self.metrics.record_tokens(len(field_content))
                        yield self.pydantic_schema_object.model_validate(parsed)

        if parsed:
            yield self.pydantic_schema_object.model_validate(parsed)


class StreamingJsonRefactored(StreamingJsonOutputParserWithMetrics):
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

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Any]:
        acc_text = ""  # accumulated unparsed text
        field_length_at_last_run = 0
        parsed = None
        seen_json_start = False

        for chunk in input:
            chunk_gen = self._to_generation_chunk(chunk)
            text = chunk_gen.text
            acc_text += text  # append only new chunk

            if not seen_json_start:
                if "{" in text:
                    seen_json_start = True
                else:
                    # pre-JSON text, record immediately
                    self._record_tokens(text)
                    continue

            # parse only if new text added after last parse
            partial = self.parse_partial_json(acc_text)
            if not partial:
                continue

            parsed = partial
            field_content = parsed.get(self.name_of_streamed_field)
            if not field_content:
                continue

            # record only **new tokens**
            if len(field_content) > field_length_at_last_run:
                new_tokens = field_content[field_length_at_last_run:]
                self.metrics.record_tokens(len(new_tokens))
                field_length_at_last_run = len(field_content)
                yield self.pydantic_schema_object.model_validate(parsed)

        # final parse if anything remains
        if parsed is None and acc_text:
            if "{" not in acc_text:
                acc_text = self.answer_str_to_json(acc_text)

            if parsed := self.parse_partial_json(acc_text):
                if field_content := parsed.get(self.name_of_streamed_field):
                    self.metrics.record_tokens(len(field_content))
                    yield self.pydantic_schema_object.model_validate(parsed)

        if parsed:
            yield self.pydantic_schema_object.model_validate(parsed)


# - Run the sync parser benchmark


def benchmark_streaming(parser: StreamingJsonOutputParserWithMetrics, input_chunks: List[Union[str, BaseMessage]]):
    for parsed in parser._transform(input_chunks):
        pass

    return parser.metrics


def benchmark_multiple_runs(parser_class, input_chunks, num_runs: int = 5, **parser_kwargs):
    """Run parser multiple times and collect metrics."""
    all_metrics = []

    for _ in range(num_runs):
        parser = parser_class(**parser_kwargs)
        metrics = benchmark_streaming(parser, input_chunks)
        all_metrics.append(metrics)

    return all_metrics


if __name__ == "__main__":
    input_chunks = generate_large_input(num_chunks=200, chunk_size=10)

    num_runs = 10
    original_runs = benchmark_multiple_runs(
        StreamingJsonOriginal, input_chunks, num_runs=num_runs, pydantic_schema_object=DummySchema
    )
    refactored_runs = benchmark_multiple_runs(
        StreamingJsonRefactored, input_chunks, num_runs=num_runs, pydantic_schema_object=DummySchema
    )

    # Aggregate metrics
    original_agg = aggregate_metrics_with_chunks(original_runs)
    refactored_agg = aggregate_metrics_with_chunks(refactored_runs)

    # Plot mean +- std.dev
    plot_aggregate_tokens_and_chunks([original_agg, refactored_agg], ["Original", "Refactored"])
