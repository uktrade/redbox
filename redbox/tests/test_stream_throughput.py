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
        self.timestamps = []
        self.per_chunk = []

    def record_tokens(self, n_tokens):
        now = time.perf_counter()
        dt = 0
        if self.timestamps:
            dt = now - self.timestamps[-1][0]
        self.tokens += n_tokens
        self.timestamps.append((now, self.tokens))
        self.per_chunk.append((now, n_tokens, dt))

    def compute_throughput(self):
        """Compute tokens/sec at each recorded timestamp"""
        if not self.timestamps:
            return []
        throughput = []
        start_time = self.timestamps[0][0]
        for t, tokens in self.timestamps:
            elapsed = t - start_time
            throughput.append((elapsed, tokens / elapsed if elapsed > 0 else 0))
        return throughput

    def compute_instantaneous_throughput(self):
        """Tokens/sec per chunk, not cumulative"""
        if len(self.timestamps) < 2:
            return []

        throughput = []
        for i in range(1, len(self.timestamps)):
            t_prev, tokens_prev = self.timestamps[i - 1]
            t_curr, tokens_curr = self.timestamps[i]
            dt = t_curr - t_prev
            d_tokens = tokens_curr - tokens_prev
            tps = d_tokens / dt if dt > 0 else 0
            throughput.append((t_curr - self.timestamps[0][0], tps))  # elapsed time vs tps
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
    text = "Hello world, this is a streaming benchmark test." * 1000
    chunks = ["```{", '"answer": "']
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    chunks.append('"}```')
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


def plot_throughput(metrics_list, labels):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for metrics, label in zip(metrics_list, labels):
        if not metrics.timestamps:
            continue

        # Normalize timestamps to start at 0
        start_time = metrics.timestamps[0][0]
        norm_times = [(t - start_time) for t, _ in metrics.timestamps]

        # Cumulative throughput (tokens/sec)
        cum_times, cum_tps = zip(*metrics.compute_throughput())
        cum_times = [t - cum_times[0] for t in cum_times]  # normalize
        axes[0].plot(cum_times, cum_tps, label=label)
        axes[0].set_title("Cumulative Throughput (Tokens/sec)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Tokens/sec")
        axes[0].set_yscale("log")
        axes[0].grid(True)

        # Instantaneous throughput per chunk (tokens/sec)
        inst_data = metrics.compute_instantaneous_throughput()
        if inst_data:
            inst_times, inst_tps = zip(*inst_data)
            inst_times = [t - inst_times[0] for t in inst_times]  # normalize
        else:
            inst_times, inst_tps = [], []
        axes[1].plot(inst_times, inst_tps, label=label)
        axes[1].set_title("Instantaneous Throughput per Chunk (Tokens/sec)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Tokens/sec")
        axes[1].set_yscale("log")
        axes[1].grid(True)

        # Cumulative chunks/sec
        chunk_counts = list(range(1, len(metrics.timestamps) + 1))
        axes[2].plot(norm_times, chunk_counts, label=label)
        axes[2].set_title("Cumulative Chunks Over Time")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Chunks")
        axes[2].set_yscale("log")
        axes[2].grid(True)

        # Instantaneous chunks/sec per interval
        inst_chunk_tps = []
        for i in range(1, len(metrics.timestamps)):
            dt = metrics.timestamps[i][0] - metrics.timestamps[i - 1][0]
            inst_chunk_tps.append(1 / dt if dt > 0 else 0)
        inst_chunk_times = [t - metrics.timestamps[0][0] for t, _ in metrics.timestamps[1:]]
        axes[3].plot(inst_chunk_times, inst_chunk_tps, label=label)
        axes[3].set_title("Instantaneous Chunks/sec")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("Chunks/sec")
        axes[3].set_yscale("log")
        axes[3].grid(True)

    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.show()


def print_metrics_summary(metrics: MetricsCollector, label: str):
    if not metrics.timestamps:
        print(f"{label}: No tokens recorded.")
        return

    times, tps = zip(*metrics.compute_throughput())
    total_tokens = metrics.tokens
    elapsed_time = times[-1]  # last timestamp is elapsed since first token
    avg_tps = total_tokens / elapsed_time if elapsed_time > 0 else 0
    max_tps = max(tps)
    min_tps = min(tps)

    print(f"--- {label} Summary ---")
    print(f"Total tokens: {total_tokens}")
    print(f"Elapsed time: {elapsed_time:.4f} s")
    print(f"Average throughput: {avg_tps:.2f} tokens/sec")
    print(f"Max throughput: {max_tps:.2f} tokens/sec")
    print(f"Min throughput: {min_tps:.2f} tokens/sec")
    print()


def benchmark_multiple_runs(parser_class, input_chunks, num_runs: int = 5, **parser_kwargs):
    """Run parser multiple times and collect metrics."""
    all_metrics = []

    for run_idx in range(num_runs):
        parser = parser_class(**parser_kwargs)
        metrics = benchmark_streaming(parser, input_chunks)
        all_metrics.append(metrics)

    return all_metrics


def aggregate_metrics(metrics_list: List[MetricsCollector]):
    """
    Align timestamps across runs by interpolating to a common timeline.
    Returns:
        times: common timeline
        mean_cum_tps: mean cumulative tokens/sec
        std_cum_tps: std deviation cumulative tokens/sec
        mean_inst_tps: mean instantaneous tokens/sec
        std_inst_tps: std deviation instantaneous tokens/sec
    """
    max_time = max(m.timestamps[-1][0] - m.timestamps[0][0] for m in metrics_list if m.timestamps)
    common_times = np.linspace(0, max_time, 1000)

    cum_tps_runs = []
    inst_tps_runs = []

    for m in metrics_list:
        if not m.timestamps:
            continue

        times, cum_tps = zip(*m.compute_throughput())
        times = np.array(times) - times[0]
        cum_tps_interp = np.interp(common_times, times, cum_tps)
        cum_tps_runs.append(cum_tps_interp)

        inst_data = m.compute_instantaneous_throughput()
        if inst_data:
            inst_times, inst_tps = zip(*inst_data)
            inst_times = np.array(inst_times) - inst_times[0]
            inst_tps_interp = np.interp(common_times, inst_times, inst_tps, left=0, right=0)
        else:
            inst_tps_interp = np.zeros_like(common_times)
        inst_tps_runs.append(inst_tps_interp)

    cum_tps_runs = np.array(cum_tps_runs)
    inst_tps_runs = np.array(inst_tps_runs)

    mean_cum_tps = np.mean(cum_tps_runs, axis=0)
    std_cum_tps = np.std(cum_tps_runs, axis=0)
    mean_inst_tps = np.mean(inst_tps_runs, axis=0)
    std_inst_tps = np.std(inst_tps_runs, axis=0)

    return common_times, mean_cum_tps, std_cum_tps, mean_inst_tps, std_inst_tps


def plot_aggregate_throughput(agg_data_list, labels):
    """
    Plot mean ± std.dev for cumulative and instantaneous throughput.
    agg_data_list: list of (times, mean_cum_tps, std_cum_tps, mean_inst_tps, std_inst_tps)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for agg_data, label in zip(agg_data_list, labels):
        times, mean_cum, std_cum, mean_inst, std_inst = agg_data

        axes[0].plot(times, mean_cum, label=label)
        axes[0].fill_between(times, mean_cum - std_cum, mean_cum + std_cum, alpha=0.3)
        axes[0].set_title("Cumulative Throughput (Tokens/sec)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Tokens/sec")
        axes[0].set_yscale("log")
        axes[0].grid(True)

        axes[1].plot(times, mean_inst, label=label)
        axes[1].fill_between(times, mean_inst - std_inst, mean_inst + std_inst, alpha=0.3)
        axes[1].set_title("Instantaneous Throughput per Chunk (Tokens/sec)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Tokens/sec")
        axes[1].set_yscale("log")
        axes[1].grid(True)

    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    original_parser = StreamingJsonOriginal(pydantic_schema_object=DummySchema)
    refactored_parser = StreamingJsonRefactored(pydantic_schema_object=DummySchema)

    input_chunks = generate_large_input(num_chunks=10, chunk_size=10)

    # metrics_refactored = benchmark_streaming(refactored_parser, input_chunks)
    # metrics_original = benchmark_streaming(original_parser, input_chunks)

    # print_metrics_summary(metrics_refactored, "Refactored Parser")
    # print_metrics_summary(metrics_original, "Original Parser")

    # plot_throughput([metrics_refactored, metrics_original], ["Refactored", "Original"])

    num_runs = 5
    original_runs = benchmark_multiple_runs(
        StreamingJsonOriginal, input_chunks, num_runs=num_runs, pydantic_schema_object=DummySchema
    )
    refactored_runs = benchmark_multiple_runs(
        StreamingJsonRefactored, input_chunks, num_runs=num_runs, pydantic_schema_object=DummySchema
    )

    # Aggregate metrics
    original_agg = aggregate_metrics(original_runs)
    refactored_agg = aggregate_metrics(refactored_runs)

    # Plot mean +- std.dev
    plot_aggregate_throughput([original_agg, refactored_agg], ["Original", "Refactored"])
