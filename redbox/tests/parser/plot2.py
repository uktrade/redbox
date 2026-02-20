# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import asyncio
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Any, AsyncIterator
# from pydantic import Field

# from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_core.messages import AIMessage
# from langchain_core.outputs import ChatGeneration, ChatResult
# from tests.parser.plot_stream_throughput import StreamingJsonOutputParserWithMetrics, DummySchema

# # ---- IMPORT YOUR REAL FUNCTION ----
# # from your_module import build_llm_chain
# # from your_module import build_chat_prompt_from_messages_runnable, get_all_metadata, final_response_if_needed


# # ------------------------------------------------------------
# # Fake large streaming content
# # ------------------------------------------------------------
# def generate_large_input(num_chunks: int = 800, chunk_size: int = 20) -> List[str]:
#     base_text = "Hello world, this is a streaming benchmark test. "
#     text = base_text * num_chunks
#     full_json = f'```{{"answer": "{text}"}}```'

#     return [full_json[i : i + chunk_size] for i in range(0, len(full_json), chunk_size)]


# # ------------------------------------------------------------
# # Fake Streaming LLM
# # ------------------------------------------------------------
# class FakeStreamingLLM(BaseChatModel):
#     chunks: List[str] = Field(default_factory=list)

#     def __init__(self, chunks: List[str]):
#         super().__init__()
#         self.chunks = chunks
#         self._default_config = {"model": "fake-stream-model"}

#     @property
#     def _llm_type(self) -> str:
#         return "fake-stream"

#     async def _astream(
#         self,
#         messages: List[Any],
#         stop: List[str] | None = None,
#         **kwargs: Any,
#     ) -> AsyncIterator[ChatGeneration]:
#         for chunk in self.chunks:
#             yield ChatGeneration(message=AIMessage(content=chunk))

#     def _generate(
#         self,
#         messages: List[Any],
#         stop: List[str] | None = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         full_text = "".join(self.chunks)
#         message = AIMessage(content=full_text)
#         return ChatResult(generations=[[ChatGeneration(message=message)]])

#     async def _agenerate(
#         self,
#         messages: List[Any],
#         stop: List[str] | None = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         full_text = "".join(self.chunks)
#         message = AIMessage(content=full_text)
#         return ChatResult(generations=[[ChatGeneration(message=message)]])


# # ------------------------------------------------------------
# # Benchmark logic
# # ------------------------------------------------------------
# class StreamMetrics:
#     def __init__(self):
#         self.timestamps = []
#         self.token_counts = []

#     def record(self, text_chunk: str):
#         self.timestamps.append(time.perf_counter())
#         self.token_counts.append(len(text_chunk.split()))

#     def get_series(self):
#         times = np.array(self.timestamps)
#         if len(times) == 0:
#             return np.array([]), np.array([])

#         times -= times[0]
#         tokens = np.array(self.token_counts)

#         speeds = np.zeros_like(tokens, dtype=float)
#         for i in range(1, len(tokens)):
#             dt = times[i] - times[i - 1]
#             speeds[i] = tokens[i] / dt if dt > 0 else 0

#         return times, speeds


# async def run_single(parser_class, chunks: List[str]):
#     """Run parser on a fake streaming LLM and collect token metrics."""
#     parser = parser_class(pydantic_schema_object=DummySchema)
#     fake_llm = FakeStreamingLLM(chunks)
#     collector = parser.metrics

#     # Feed fake LLM output to parser
#     async for _ in parser._transform(fake_llm):
#         pass

#     # Return timestamps and token counts
#     return collector


# async def benchmark(parser_class, runs: int = 5):
#     chunks = generate_large_input(num_chunks=200, chunk_size=10)
#     all_runs_series = []

#     for i in range(runs):
#         print(f"Run {i + 1}/{runs}")
#         collector = await run_single(parser_class, chunks)
#         times, speeds = collector.get_time_series(dt=0.1)
#         all_runs_series.append((times, speeds))
#         await asyncio.sleep(0.1)

#     # -------------------------------
#     # Aggregate and interpolate
#     # -------------------------------
#     valid_runs = [(times, speeds) for times, speeds in all_runs_series if len(times) > 0]
#     if not valid_runs:
#         print("No valid runs")
#         return

#     max_time = max(times[-1] for times, _ in valid_runs)
#     dt = 0.1
#     common_bins = np.arange(0, max_time + dt, dt)
#     interpolated_speeds = []
#     for times, speeds in valid_runs:
#         interp = np.interp(common_bins, times, speeds, left=0, right=0)
#         interpolated_speeds.append(interp)

#     interpolated_speeds = np.array(interpolated_speeds)
#     mean_speeds = interpolated_speeds.mean(axis=0)
#     std_speeds = interpolated_speeds.std(axis=0)

#     # -------------------------------
#     # Plot results
#     # -------------------------------
#     fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

#     # Individual runs
#     for i, run in enumerate(interpolated_speeds):
#         axes[0].plot(common_bins, run, alpha=0.4, label=f"Run {i + 1}")
#     axes[0].set_title("Individual Run Throughput")
#     axes[0].set_ylabel("Tokens/sec")
#     axes[0].set_ylim(bottom=0)
#     axes[0].grid(True)
#     if len(interpolated_speeds) <= 10:
#         axes[0].legend()

#     # Mean ± Std
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


# # ------------------------------------------------------------
# # HOW TO CONNECT TO YOUR build_llm_chain
# # ------------------------------------------------------------

# # def build_chain_with_fake_llm(fake_llm):
# #     # Replace prompt_set with minimal stub if needed
# #     prompt_set = PromptSet.NewRoute  # or your real PromptSet

# #     return build_llm_chain(
# #         prompt_set=prompt_set,
# #         llm=fake_llm,
# #         output_parser=StrOutputParser(),
# #     )


# if __name__ == "__main__":
#     asyncio.run(benchmark(StreamingJsonOutputParserWithMetrics, runs=5))
