import time
from pydantic import BaseModel, Field

from redbox.chains.parser import StreamingJsonOutputParser, BaseMessage
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np


class MetricsCollector:
    def __init__(self):
        self.tokens = 0
        self.timestamps = []

    def record_tokens(self, n_tokens):
        now = time.perf_counter()
        self.tokens += n_tokens
        self.timestamps.append((now, self.tokens))

    def compute_throughput(self):
        """Cumulative tokens/sec"""
        if not self.timestamps:
            return []
        start_time = self.timestamps[0][0]
        throughput = [
            (t - start_time, tokens / (t - start_time) if (t - start_time) > 0 else 0) for t, tokens in self.timestamps
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


class StreamingJsonOutputParserWithMetrics(StreamingJsonOutputParser):
    metrics: MetricsCollector = Field(default_factory=MetricsCollector)

    class Config:
        arbitrary_types_allowed = True  # allow MetricsCollector type
        extra = "allow"  # allow other attributes


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
    for parsed in parser._transform(input_chunks):
        pass

    return parser.metrics


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

        # # Cumulative chunks
        # t_times, cum_chunks = zip(*m.compute_chunk_throughput())
        # t_times = np.array(t_times) - t_times[0]
        # cum_chunks_interp = np.interp(common_times, t_times, cum_chunks)
        # cum_chunks_runs.append(cum_chunks_interp)

        # # Instantaneous chunks
        # inst_chunks_data = m.compute_instantaneous_chunk_throughput()
        # if inst_chunks_data:
        #     inst_times, inst_chunks = zip(*inst_chunks_data)
        #     inst_times = np.array(inst_times) - inst_times[0]
        #     inst_chunks_interp = np.interp(common_times, inst_times, inst_chunks, left=0, right=0)
        # else:
        #     inst_chunks_interp = np.zeros_like(common_times)
        # inst_chunks_runs.append(inst_chunks_interp)

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
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

        # # Cumulative chunks
        # axes[2].plot(times, mean_cum_chunks, label=label)
        # axes[2].fill_between(times, mean_cum_chunks - std_cum_chunks, mean_cum_chunks + std_cum_chunks, alpha=0.3)
        # axes[2].set_title("Cumulative Throughput (Chunks/sec)")
        # # axes[2].set_yscale("log")
        # axes[2].grid(True)

        # # Instantaneous chunks
        # axes[3].plot(times, mean_inst_chunks, label=label)
        # axes[3].fill_between(times, mean_inst_chunks - std_inst_chunks, mean_inst_chunks + std_inst_chunks, alpha=0.3)
        # axes[3].set_title("Instantaneous Throughput per Chunk (Chunks/sec)")
        # # axes[3].set_yscale("log")
        # axes[3].grid(True)

    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Throughput/sec")
        ax.legend()

    plt.tight_layout()
    plt.show()
