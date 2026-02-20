from redbox.chains.components import get_chat_llm
import asyncio
import time
import matplotlib.pyplot as plt
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from redbox.models.settings import ChatLLMBackend


async def benchmark_llm(model: BaseChatModel, prompt: str, num_runs: int = 5):
    run_tokens = []
    run_speeds = []

    for i in range(num_runs):
        start = time.perf_counter()
        response = await model.agenerate([[HumanMessage(content=prompt)]])
        end = time.perf_counter()

        n_tokens = len(response.generations[0][0].text.split())
        duration = end - start
        speed = n_tokens / duration if duration > 0 else 0

        run_tokens.append(n_tokens)
        run_speeds.append(speed)

        print(f"Run {i + 1}: {n_tokens} tokens in {duration:.2f}s => {speed:.1f} tokens/sec")

    # Plot throughput
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_runs + 1), run_speeds, marker="o", label="Tokens/sec")
    plt.xlabel("Run")
    plt.ylabel("Throughput (tokens/sec)")
    plt.title(f"LLM Token Throughput Across {num_runs} Runs")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Optional: print aggregate stats
    mean_speed = sum(run_speeds) / len(run_speeds)
    std_speed = (sum((s - mean_speed) ** 2 for s in run_speeds) / len(run_speeds)) ** 0.5
    print(f"Mean throughput: {mean_speed:.1f} tokens/sec ± {std_speed:.1f} std dev")


# Get the LLM used in your chain
llm = get_chat_llm(model=ChatLLMBackend(name="gpt-4o", provider="openai"), tools=None)

prompt_text = "Explain recursion with an example in Python."

asyncio.run(benchmark_llm(llm, prompt_text, num_runs=5))
