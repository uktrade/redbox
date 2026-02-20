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

# import asyncio
# import time
# import numpy as np
# import matplotlib.pyplot as plt

# from redbox.app import Redbox, RedboxState
# from redbox.models.chain import RedboxQuery
# from redbox.graph.agents.configs import agent_configs
# from uuid import uuid4


# # -------------------------------
# # Metrics collector
# # -------------------------------
# class CallbackMetricsCollector:
#     """Records timestamps and token counts for one run"""

#     def __init__(self):
#         self.timestamps = []
#         self.token_counts = []

#     def record(self, n_tokens: int = 1):
#         self.timestamps.append(time.perf_counter())
#         self.token_counts.append(n_tokens)

#     def get_time_series(self, dt: float = 0.1):
#         """Return binned time series (time grid, tokens/sec)"""
#         if len(self.timestamps) < 2:
#             return np.array([]), np.array([])

#         start_time = self.timestamps[0]
#         times = np.array(self.timestamps) - start_time
#         speeds = np.array(self.token_counts) / np.diff(np.concatenate(([0], times)))

#         # Bin speeds into fixed dt intervals
#         max_time = times[-1]
#         bins = np.arange(0, max_time + dt, dt)
#         digitized = np.digitize(times, bins)
#         binned_speeds = []
#         for i in range(1, len(bins)):
#             mask = digitized == i
#             if mask.any():
#                 binned_speeds.append(speeds[mask].mean())
#             else:
#                 binned_speeds.append(0.0)
#         return bins[:-1], np.array(binned_speeds)


# async def simulate_single_run(
#     app: Redbox,
#     query: str,
# ):
#     # app.graph.astream_events = mock_astream_events
#     collector = CallbackMetricsCollector()
#     state = RedboxState(request=RedboxQuery(question=query, user_uuid=uuid4(), chat_history=[]))

#     total_tokens = 0
#     start_time = None

#     async def timed_streaming_response_handler(tokens: str):
#         nonlocal total_tokens, start_time

#         if start_time is None:
#             start_time = time.perf_counter()

#         n_tokens = len(tokens.split())

#         total_tokens += n_tokens
#         collector.record(n_tokens)

#     async def noop_callback(*args, **kwargs):
#         pass

#     await app.run(
#         input=state,
#         response_tokens_callback=timed_streaming_response_handler,
#         metadata_tokens_callback=noop_callback,
#         route_name_callback=noop_callback,
#         activity_event_callback=noop_callback,
#         documents_callback=noop_callback,
#     )

#     # Only return collector if full target reached
#     return collector


# async def benchmark_runs(
#     test_queries: list[str],
#     runs_per_query: int = 3,
#     dt: float = 0.1,
# ):
#     app = Redbox(agents=agent_configs, debug=True)
#     all_runs_series = []

#     for query in test_queries:
#         for run_idx in range(runs_per_query):
#             print(f"Running query: '{query}' (run {run_idx + 1})")

#             collector = await simulate_single_run(
#                 app,
#                 query,
#             )

#             if collector is None:
#                 continue

#             times, speeds = collector.get_time_series(dt=dt)

#             if len(times) > 0:
#                 all_runs_series.append((times, speeds))

#             await asyncio.sleep(0.1)

#     if not all_runs_series:
#         print("No valid runs to aggregate.")
#         return

#     # Align all runs to shortest duration
#     min_max_time = min(times[-1] for times, _ in all_runs_series)
#     common_bins = np.arange(0, min_max_time + dt, dt)

#     interpolated_speeds = []
#     for times, speeds in all_runs_series:
#         interp = np.interp(common_bins, times, speeds)
#         interpolated_speeds.append(interp)

#     interpolated_speeds = np.array(interpolated_speeds)

#     mean_speeds = interpolated_speeds.mean(axis=0)
#     std_speeds = interpolated_speeds.std(axis=0)

#     # -------------------------------
#     # Plot
#     # -------------------------------
#     fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

#     # Individual runs
#     for i, run in enumerate(interpolated_speeds):
#         axes[0].plot(common_bins, run, alpha=0.4, label=f"Run {i}")

#     axes[0].set_title("Individual Runs")
#     axes[0].set_ylabel("Tokens/sec")
#     axes[0].set_ylim(bottom=0)
#     axes[0].grid(True)
#     axes[0].legend()

#     # Aggregated
#     axes[1].plot(common_bins, mean_speeds, color="red", label="Mean throughput")
#     axes[1].fill_between(
#         common_bins,
#         mean_speeds - std_speeds,
#         mean_speeds + std_speeds,
#         color="red",
#         alpha=0.2,
#         label="±1 std dev",
#     )

#     axes[1].set_title("Aggregated Throughput (Mean ± Std Dev)")
#     axes[1].set_xlabel("Elapsed time (s)")
#     axes[1].set_ylabel("Tokens/sec")
#     axes[1].set_ylim(bottom=0)
#     axes[1].grid(True)
#     axes[1].legend()

#     plt.tight_layout()
#     plt.show()

#     # Print summary stats
#     overall_mean = mean_speeds.mean()
#     overall_std = mean_speeds.std()
#     print(f"\nOverall mean throughput: {overall_mean:.2f} tokens/sec")
#     print(f"Std dev of mean curve: {overall_std:.2f}")


# # -------------------------------
# # Execute benchmark
# # -------------------------------
# test_queries = [
#     "Explain recursion",
# ]

# asyncio.run(benchmark_runs(test_queries, runs_per_query=3, dt=0.1))
