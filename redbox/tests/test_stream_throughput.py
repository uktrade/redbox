import time
import re
from typing import List, Union, Iterator, Any, AsyncIterator
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import numpy as np

from redbox.chains.parser import StreamingJsonOutputParser, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk


class MetricsCollector:
    def __init__(self):
        self.tokens = 0
        self.chunks = 0
        self.timestamps = []
        self.chunk_timestamps = []

    def record_tokens(self, n_tokens):
        now = time.perf_counter()
        self.tokens += n_tokens
        self.timestamps.append((now, self.tokens))
        self.chunks += 1
        self.chunk_timestamps.append((now, self.chunks))

    def compute_throughput(self):
        """Cumulative tokens/sec"""
        if not self.timestamps:
            return []
        start_time = self.timestamps[0][0]
        throughput = [
            (t - start_time, tokens / (t - start_time) if (t - start_time) > 0 else 0) for t, tokens in self.timestamps
        ]
        return throughput

    def compute_chunk_throughput(self):
        """Cumulative chunks/sec"""
        if not self.chunk_timestamps:
            return []
        start_time = self.chunk_timestamps[0][0]
        throughput = [
            (t - start_time, chunks / (t - start_time) if (t - start_time) > 0 else 0)
            for t, chunks in self.chunk_timestamps
        ]
        return throughput

    def compute_instantaneous_throughput(self):
        """Instantaneous tokens/sec per chunk"""
        if len(self.timestamps) < 2:
            return []
        throughput = []
        for i in range(1, len(self.timestamps)):
            t_prev, tokens_prev = self.timestamps[i - 1]
            t_curr, tokens_curr = self.timestamps[i]
            dt = t_curr - t_prev
            d_tokens = tokens_curr - tokens_prev
            tps = d_tokens / dt if dt > 0 else 0
            throughput.append((t_curr - self.timestamps[0][0], tps))
        return throughput

    def compute_instantaneous_chunk_throughput(self):
        """Instantaneous chunks/sec"""
        if len(self.chunk_timestamps) < 2:
            return []
        throughput = []
        for i in range(1, len(self.chunk_timestamps)):
            t_prev, chunks_prev = self.chunk_timestamps[i - 1]
            t_curr, chunks_curr = self.chunk_timestamps[i]
            dt = t_curr - t_prev
            d_chunks = chunks_curr - chunks_prev
            cps = d_chunks / dt if dt > 0 else 0
            throughput.append((t_curr - self.chunk_timestamps[0][0], cps))
        return throughput


class StreamingJsonOutputParserWithMetrics(StreamingJsonOutputParser):
    metrics: MetricsCollector = Field(default_factory=MetricsCollector)

    class Config:
        arbitrary_types_allowed = True  # allow MetricsCollector type
        extra = "allow"  # allow other attributes

    def _record_tokens(self, tokens: str):
        self.metrics.record_tokens(len(tokens))


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
                        self._record_tokens(new_tokens)
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
                        self._record_tokens(field_content)
                        yield self.pydantic_schema_object.model_validate(parsed)

        if parsed:
            yield self.pydantic_schema_object.model_validate(parsed)

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
                    if _ := field_content[field_length_at_last_run:]:
                        # parser_module.dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens)
                        self._record_tokens(field_content)
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
                        self._record_tokens(field_content)
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
        acc_buffer: list[str] = []
        field_length_at_last_run: int = 0
        parsed = None
        seen_json_start = False

        for chunk in input:
            chunk_gen = self._to_generation_chunk(chunk)
            text = chunk_gen.text
            acc_buffer.append(text)

            if not seen_json_start and "{" in text:
                seen_json_start = True
            elif not seen_json_start:
                # If JSON hasn't started, skip but record the raw chunk as "pre-JSON"
                self._record_tokens(text)
                continue

            acc_text = "".join(acc_buffer)
            partial = self.parse_partial_json(acc_text)
            if not partial:
                continue

            parsed = partial
            field_content = parsed.get(self.name_of_streamed_field)
            if not field_content:
                continue

            if len(field_content) > field_length_at_last_run:
                new_tokens = field_content[field_length_at_last_run:]
                self._record_tokens(new_tokens)
                # parser_module.dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens)
                field_length_at_last_run = len(field_content)
                yield self.pydantic_schema_object.model_validate(parsed)

        if parsed is None and acc_buffer:
            acc_text = "".join(acc_buffer)
            if "{" not in acc_text:
                acc_text = self.answer_str_to_json(acc_text)

            if parsed := self.parse_partial_json(acc_text):
                if field_content := parsed.get(self.name_of_streamed_field):
                    # parser_module.dispatch_custom_event(RedboxEventType.response_tokens, data=field_content)
                    self._record_tokens(field_content)
                    yield self.pydantic_schema_object.model_validate(parsed)

        if parsed:
            yield self.pydantic_schema_object.model_validate(parsed)

    async def _atransform(self, input: AsyncIterator[Union[str, BaseMessage]]) -> AsyncIterator[Any]:
        acc_buffer: list[str] = []
        field_length_at_last_run: int = 0
        parsed = None
        seen_json_start = False

        async for chunk in input:
            chunk_gen = self._to_generation_chunk(chunk)
            text = chunk_gen.text
            acc_buffer.append(text)

            if not seen_json_start and "{" in text:
                seen_json_start = True
            elif not seen_json_start:
                # If JSON hasn't started, skip but record the raw chunk as "pre-JSON"
                self._record_tokens(text)
                continue

            acc_text = "".join(acc_buffer)
            partial = self.parse_partial_json(acc_text)
            if not partial:
                continue

            parsed = partial
            field_content = parsed.get(self.name_of_streamed_field)
            if not field_content:
                continue

            if len(field_content) > field_length_at_last_run:
                new_tokens = field_content[field_length_at_last_run:]
                self._record_tokens(new_tokens)
                # parser_module.dispatch_custom_event(RedboxEventType.response_tokens, data=new_tokens)
                field_length_at_last_run = len(field_content)
                yield self.pydantic_schema_object.model_validate(parsed)

        if parsed is None and acc_buffer:
            acc_text = "".join(acc_buffer)
            if "{" not in acc_text:
                acc_text = self.answer_str_to_json(acc_text)

            if parsed := self.parse_partial_json(acc_text):
                if field_content := parsed.get(self.name_of_streamed_field):
                    # parser_module.dispatch_custom_event(RedboxEventType.response_tokens, data=field_content)
                    self._record_tokens(field_content)
                    yield self.pydantic_schema_object.model_validate(parsed)

        if parsed:
            yield self.pydantic_schema_object.model_validate(parsed)


class DummySchema(BaseModel):
    answer: str
    citations: list = []


def generate_large_input(num_chunks: int = 100, chunk_size: int = 10) -> List[str]:
    base_text = "Hello world, this is a streaming benchmark test. "
    text = base_text * num_chunks

    full_json = f'```{{"answer": "{text}"}}```'

    chunks = []
    for i in range(0, len(full_json), chunk_size):
        chunks.append(full_json[i : i + chunk_size])

    return chunks


def benchmark_streaming(parser: StreamingJsonOutputParserWithMetrics, input_chunks: List[Union[str, BaseMessage]]):
    # metrics = MetricsCollector()

    # Patch module-level dispatch_custom_event to count tokens
    # original_dispatch = parser_module.dispatch_custom_event

    # def token_count_dispatch(event_type, data=None, **kwargs):
    #     if data:
    #         metrics.record_tokens(len(data))

    # parser_module.dispatch_custom_event = token_count_dispatch

    for parsed in parser._transform(input_chunks):
        pass

    # Restore original function
    # parser_module.dispatch_custom_event = original_dispatch

    return parser.metrics


# def print_metrics_summary(metrics: MetricsCollector, label: str):
#     if not metrics.timestamps:
#         print(f"{label}: No tokens recorded.")
#         return

#     times, tps = zip(*metrics.compute_throughput())
#     total_tokens = metrics.tokens
#     elapsed_time = times[-1]  # last timestamp is elapsed since first token
#     avg_tps = total_tokens / elapsed_time if elapsed_time > 0 else 0
#     max_tps = max(tps)
#     min_tps = min(tps)

#     print(f"--- {label} Summary ---")
#     print(f"Total tokens: {total_tokens}")
#     print(f"Elapsed time: {elapsed_time:.4f} s")
#     print(f"Average throughput: {avg_tps:.2f} tokens/sec")
#     print(f"Max throughput: {max_tps:.2f} tokens/sec")
#     print(f"Min throughput: {min_tps:.2f} tokens/sec")
#     print()


def benchmark_multiple_runs(parser_class, input_chunks, num_runs: int = 5, **parser_kwargs):
    """Run parser multiple times and collect metrics."""
    all_metrics = []

    for run_idx in range(num_runs):
        parser = parser_class(**parser_kwargs)
        metrics = benchmark_streaming(parser, input_chunks)
        all_metrics.append(metrics)

    return all_metrics


def aggregate_metrics_with_chunks(metrics_list: List[MetricsCollector]):
    """
    Interpolate and compute mean/std for tokens/sec and chunks/sec (cumulative and instantaneous)
    """
    max_time = max(m.timestamps[-1][0] - m.timestamps[0][0] for m in metrics_list if m.timestamps)
    common_times = np.linspace(0, max_time, 1000)

    # Tokens
    cum_tokens_runs, inst_tokens_runs = [], []
    cum_chunks_runs, inst_chunks_runs = [], []

    for m in metrics_list:
        if not m.timestamps:
            continue
        # Cumulative tokens
        t_times, cum_tokens = zip(*m.compute_throughput())
        t_times = np.array(t_times) - t_times[0]
        cum_tokens_interp = np.interp(common_times, t_times, cum_tokens)
        cum_tokens_runs.append(cum_tokens_interp)

        # Instantaneous tokens
        inst_tokens_data = m.compute_instantaneous_throughput()
        if inst_tokens_data:
            inst_times, inst_tokens = zip(*inst_tokens_data)
            inst_times = np.array(inst_times) - inst_times[0]
            inst_tokens_interp = np.interp(common_times, inst_times, inst_tokens, left=0, right=0)
        else:
            inst_tokens_interp = np.zeros_like(common_times)
        inst_tokens_runs.append(inst_tokens_interp)

        # Cumulative chunks
        t_times, cum_chunks = zip(*m.compute_chunk_throughput())
        t_times = np.array(t_times) - t_times[0]
        cum_chunks_interp = np.interp(common_times, t_times, cum_chunks)
        cum_chunks_runs.append(cum_chunks_interp)

        # Instantaneous chunks
        inst_chunks_data = m.compute_instantaneous_chunk_throughput()
        if inst_chunks_data:
            inst_times, inst_chunks = zip(*inst_chunks_data)
            inst_times = np.array(inst_times) - inst_times[0]
            inst_chunks_interp = np.interp(common_times, inst_times, inst_chunks, left=0, right=0)
        else:
            inst_chunks_interp = np.zeros_like(common_times)
        inst_chunks_runs.append(inst_chunks_interp)

    cum_tokens_runs = np.array(cum_tokens_runs)
    inst_tokens_runs = np.array(inst_tokens_runs)
    cum_chunks_runs = np.array(cum_chunks_runs)
    inst_chunks_runs = np.array(inst_chunks_runs)

    return (
        common_times,
        np.mean(cum_tokens_runs, axis=0),
        np.std(cum_tokens_runs, axis=0),
        np.mean(inst_tokens_runs, axis=0),
        np.std(inst_tokens_runs, axis=0),
        np.mean(cum_chunks_runs, axis=0),
        np.std(cum_chunks_runs, axis=0),
        np.mean(inst_chunks_runs, axis=0),
        np.std(inst_chunks_runs, axis=0),
    )


def plot_aggregate_tokens_and_chunks(agg_data_list, labels):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for agg_data, label in zip(agg_data_list, labels):
        (
            times,
            mean_cum_tokens,
            std_cum_tokens,
            mean_inst_tokens,
            std_inst_tokens,
            mean_cum_chunks,
            std_cum_chunks,
            mean_inst_chunks,
            std_inst_chunks,
        ) = agg_data

        # Cumulative tokens
        axes[0].plot(times, mean_cum_tokens, label=label)
        axes[0].fill_between(times, mean_cum_tokens - std_cum_tokens, mean_cum_tokens + std_cum_tokens, alpha=0.3)
        axes[0].set_title("Cumulative Throughput (Tokens/sec)")
        # axes[0].set_yscale("log")
        axes[0].grid(True)

        # Instantaneous tokens
        axes[1].plot(times, mean_inst_tokens, label=label)
        axes[1].fill_between(times, mean_inst_tokens - std_inst_tokens, mean_inst_tokens + std_inst_tokens, alpha=0.3)
        axes[1].set_title("Instantaneous Throughput per Chunk (Tokens/sec)")
        # axes[1].set_yscale("log")
        axes[1].grid(True)

        # Cumulative chunks
        axes[2].plot(times, mean_cum_chunks, label=label)
        axes[2].fill_between(times, mean_cum_chunks - std_cum_chunks, mean_cum_chunks + std_cum_chunks, alpha=0.3)
        axes[2].set_title("Cumulative Throughput (Chunks/sec)")
        # axes[2].set_yscale("log")
        axes[2].grid(True)

        # Instantaneous chunks
        axes[3].plot(times, mean_inst_chunks, label=label)
        axes[3].fill_between(times, mean_inst_chunks - std_inst_chunks, mean_inst_chunks + std_inst_chunks, alpha=0.3)
        axes[3].set_title("Instantaneous Throughput per Chunk (Chunks/sec)")
        # axes[3].set_yscale("log")
        axes[3].grid(True)

    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Throughput/sec")
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    original_parser = StreamingJsonOriginal(pydantic_schema_object=DummySchema)
    refactored_parser = StreamingJsonRefactored(pydantic_schema_object=DummySchema)

    input_chunks = generate_large_input(num_chunks=20, chunk_size=20)

    # metrics_refactored = benchmark_streaming(refactored_parser, input_chunks)
    # metrics_original = benchmark_streaming(original_parser, input_chunks)

    # print_metrics_summary(metrics_refactored, "Refactored Parser")
    # print_metrics_summary(metrics_original, "Original Parser")

    # plot_throughput([metrics_refactored, metrics_original], ["Refactored", "Original"])

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
