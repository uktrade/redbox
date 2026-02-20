# from pathlib import Path
# from dotenv import load_dotenv

# BASE_DIR = Path(__file__).resolve().parent

# for env_file in ["../../../tests/.env.integration", "../../../.env", "../../../.env.local"]:
#     path = (BASE_DIR / env_file).resolve()
#     if path.exists():
#         load_dotenv(path, override=True)
#         print(f"Loaded env: {path}")
#     else:
#         print(f"Env file not found: {path}")

# from pathlib import Path
# from dotenv import load_dotenv
# import asyncio
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from uuid import uuid4

# from redbox.app import Redbox, RedboxState
# from redbox.models.chain import RedboxQuery
# from redbox.graph.agents.configs import agent_configs
# from redbox.graph.nodes.processes import build_stuff_pattern
# from redbox.models.chain import PromptSet
# from redbox.chains.runnables import Runnable

# # -------------------------------
# # Load .env files
# # -------------------------------
# BASE_DIR = Path(__file__).resolve().parent
# for env_file in ["../../../tests/.env.integration", "../../../.env", "../../../.env.local"]:
#     path = (BASE_DIR / env_file).resolve()
#     if path.exists():
#         load_dotenv(path, override=True)
#         print(f"Loaded env: {path}")
#     else:
#         print(f"Env file not found: {path}")


# # -------------------------------
# # Metrics collector for streaming
# # -------------------------------
# class CallbackMetricsCollector:
#     """Collect token throughput over time."""

#     def __init__(self):
#         self.timestamps = []
#         self.token_counts = []

#     def record(self, n_tokens: int):
#         now = time.perf_counter()
#         self.timestamps.append(now)
#         self.token_counts.append(n_tokens)

#     def get_time_series(self):
#         times = np.array(self.timestamps)
#         times -= times[0] if len(times) > 0 else 0
#         token_counts = np.array(self.token_counts)

#         # Instantaneous speeds
#         speeds = np.zeros_like(token_counts, dtype=float)
#         for i in range(1, len(token_counts)):
#             dt = times[i] - times[i - 1]
#             speeds[i] = token_counts[i] / dt if dt > 0 else 0

#         # Smoothed moving average
#         window = 5
#         if len(speeds) >= window:
#             smoothed = np.convolve(speeds, np.ones(window) / window, mode="valid")
#             smoothed_times = times[window - 1 :]
#         else:
#             smoothed, smoothed_times = speeds, times

#         return times, speeds, smoothed_times, smoothed


# # -------------------------------
# # Simulate a single streaming run
# # -------------------------------
# async def simulate_single_run(chain_runnable: Runnable, state: RedboxState) -> CallbackMetricsCollector:
#     collector = CallbackMetricsCollector()

#     async for event in chain_runnable.astream(state):
#         # Extract tokens from the event
#         tokens = event.get("raw_response") or event.get("parsed_response") or ""
#         n_tokens = len(tokens.split())
#         collector.record(n_tokens)

#     return collector


# # -------------------------------
# # Benchmark multiple runs
# # -------------------------------
# async def benchmark_runs(
#     test_queries: list[str],
#     num_runs: int = 3,
#     dt: float = 0.1,
# ):
#     app = Redbox(agents=agent_configs, debug=True)
#     all_runs_series = []

#     for query in test_queries:
#         # Build your chain (after tools have run)
#         chain_runnable = build_stuff_pattern(prompt_set=PromptSet.NewRoute)

#         for run_idx in range(num_runs):
#             print(f"Running query: '{query}' (run {run_idx + 1})")
#             state = RedboxState(
#                 request=RedboxQuery(question=query, user_uuid=uuid4(), chat_history=[]), agents_results=[]
#             )
#             collector = await simulate_single_run(chain_runnable, state)
#             times, speeds, smoothed_times, smoothed_speeds = collector.get_time_series()
#             all_runs_series.append((times, speeds))
#             await asyncio.sleep(0.1)

#     # Filter out empty runs
#     valid_runs = [(t, s) for t, s in all_runs_series if len(t) > 0]
#     if not valid_runs:
#         print("No valid runs to aggregate.")
#         return

#     # -------------------------------
#     # Align runs on common time grid
#     # -------------------------------
#     max_time = max(t[-1] for t, _ in valid_runs)
#     common_bins = np.arange(0, max_time + dt, dt)
#     interpolated_speeds = []
#     for times, speeds in valid_runs:
#         interp = np.interp(common_bins, times, speeds, left=0, right=0)
#         interpolated_speeds.append(interp)
#     interpolated_speeds = np.array(interpolated_speeds)

#     # Compute mean and std
#     mean_speeds = interpolated_speeds.mean(axis=0)
#     std_speeds = interpolated_speeds.std(axis=0)

#     # -------------------------------
#     # Plot per-run and aggregate
#     # -------------------------------
#     fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

#     # Subplot 1: individual runs
#     for i, run in enumerate(interpolated_speeds):
#         axes[0].plot(common_bins, run, alpha=0.4, label=f"Run {i + 1}")
#     axes[0].set_title("Individual Streaming Runs")
#     axes[0].set_ylabel("Tokens/sec")
#     axes[0].grid(True)
#     if len(interpolated_speeds) <= 10:
#         axes[0].legend()

#     # Subplot 2: mean ± std
#     axes[1].plot(common_bins, mean_speeds, color="red", label="Mean throughput")
#     axes[1].fill_between(
#         common_bins, mean_speeds - std_speeds, mean_speeds + std_speeds, color="red", alpha=0.2, label="±1 std dev"
#     )
#     axes[1].set_title("Aggregated Streaming Throughput")
#     axes[1].set_xlabel("Elapsed time (s)")
#     axes[1].set_ylabel("Tokens/sec")
#     axes[1].set_ylim(bottom=0)
#     axes[1].grid(True)
#     axes[1].legend()

#     plt.tight_layout()
#     plt.show()


# # -------------------------------
# # Execute benchmark
# # -------------------------------
# if __name__ == "__main__":
#     test_queries = [
#         "Count to 1000",
#         "Explain recursion",
#         "List all prime numbers below 100",
#     ]
#     asyncio.run(benchmark_runs(test_queries, num_runs=3, dt=0.1))
