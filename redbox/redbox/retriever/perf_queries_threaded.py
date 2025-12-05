from dataclasses import dataclass
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict

from langchain.embeddings import FakeEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection, Urllib3HttpConnection

import matplotlib.pyplot as plt

# -----------------------------------------------------
# Setup
# -----------------------------------------------------

np.random.seed(42)

INDEX = "redbox-data-chunk"

client_rhc = OpenSearch(
    "http://localhost:9200",
    connection_class=RequestsHttpConnection,
    compression=False,
    timeout=120,  # seconds
    max_retries=3,
    retry_on_timeout=True,
)
client_rhc_cmpr = OpenSearch(
    "http://localhost:9200",
    connection_class=RequestsHttpConnection,
    compression=True,
    timeout=120,  # seconds
    max_retries=3,
    retry_on_timeout=True,
)
client_ul3 = OpenSearch(
    [{"host": "localhost", "port": 9200}],
    connection_class=Urllib3HttpConnection,
    compression=False,
    timeout=120,  # seconds
    max_retries=3,
    retry_on_timeout=True,
)
client_ul3_cmpr = OpenSearch(
    [{"host": "localhost", "port": 9200}],
    connection_class=Urllib3HttpConnection,
    compression=True,
    timeout=120,  # seconds
    max_retries=3,
    retry_on_timeout=True,
)

fake = FakeEmbeddings(size=1024)
query_vec = fake.embed_query("fixed query")


# -----------------------------------------------------
# Query param dataclass
# -----------------------------------------------------


@dataclass
class QueryParams:
    k: int
    size: int
    min_score: float
    files: list[str]

    def dev_body(self, vec: list[float]) -> dict:
        return {
            "size": self.size,
            "min_score": self.min_score,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "vector_field": {
                                    "vector": vec,
                                    "k": self.k,
                                    "filter": {"bool": {"must": {"terms": {"metadata.uri.keyword": self.files}}}},
                                }
                            }
                        }
                    ]
                }
            },
            "_source": {"excludes": ["vector_field"]},
        }

    def v1_body(self, vec: list[float]) -> dict:
        return {
            "size": self.size,
            "min_score": self.min_score,
            "query": {
                "knn": {
                    "vector_field": {
                        "vector": vec,
                        "k": self.k,
                        "filter": {"bool": {"filter": {"terms": {"metadata.uri.keyword": self.files}}}},
                    }
                }
            },
            "_source": {"excludes": ["vector_field"]},
        }

    def v2_body(self, vec: list[float]) -> dict:
        return {
            "size": self.size,
            "min_score": self.min_score,
            "query": {
                "bool": {
                    "filter": [{"terms": {"metadata.uri.keyword": self.files}}],
                    "must": [
                        {
                            "knn": {
                                "vector_field": {
                                    "vector": vec,
                                    "k": self.k,
                                }
                            }
                        }
                    ],
                }
            },
            "_source": {"excludes": ["vector_field"]},
        }


# -----------------------------------------------------
# Multithreaded load tester
# -----------------------------------------------------


def query_ms_parallel(
    client: OpenSearch,
    body: dict,
    num_threads: int = 20,
    num_requests_per_thread: int = 1000,
    warmup: int = 20,
    sleep_between_requests: float = 0.01,
):
    """
    Run OpenSearch queries in parallel to simulate prod-level load.
    Each thread performs (warmup + N) requests.
    Returns dict of latency statistics.
    """

    def worker(thread_id):
        # Warmup
        for _ in range(warmup):
            client.search(index=INDEX, body=body)

        results = []
        for _ in range(num_requests_per_thread):
            start = time.perf_counter()
            resp = client.search(index=INDEX, body=body)
            if sleep_between_requests > 0:
                time.sleep(sleep_between_requests)
            _ = resp.get("hits", {}).get("total")  # force access
            results.append((time.perf_counter() - start) * 1000)
        return results

    latencies = []

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = [ex.submit(worker, i) for i in range(num_threads)]

        for f in as_completed(futures):
            try:
                latencies.extend(f.result())
            except Exception as e:
                print("Thread error:", e)

    # Aggregate results
    if not latencies:
        return {"avg": None, "p95": None, "p99": None, "count": 0}

    lat_sorted = sorted(latencies)
    count = len(lat_sorted)

    def percentile(p):
        idx = int(count * p)
        idx = min(idx, count - 1)
        return lat_sorted[idx]

    return {
        "count": count,
        "avg": sum(lat_sorted) / count,
        "median": statistics.median(lat_sorted),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": lat_sorted[-1],
        "min": lat_sorted[0],
    }


# -----------------------------------------------------
# Test configuration
# -----------------------------------------------------

k = 10
size = 30
min_score = 0.6

files = [
    "harry.wixley@digital.trade.gov.uk/2025-05-30_Submission_-_OBR_Scoring_Process_for_ERB_20_X0kFtvO.pdf",
    "harry.wixley@digital.trade.gov.uk/DCE_and_TRQ_Review_2025_Final_Recommendation_Submissio_8oS5b2u.pdf",
    "harry.wixley@digital.trade.gov.uk/OBR_Economic_and_fiscal_outlook_November_2025_2-cd34fb_KpYbGtN.pdf",
    "harry.wixley@digital.trade.gov.uk/Annex_2A_part_1_20251202092721880.pdf",
    "harry.wixley@digital.trade.gov.uk/Annex_2A_part_2_20251202110301695.pdf",
    "harry.wixley@digital.trade.gov.uk/Annex_2A_part_3_20251202110302044.pdf",
]
# files = files + files  # duplicate for more combinations
files = files * 3

clients = {"RHC": {False: client_rhc, True: client_rhc_cmpr}, "UL3": {False: client_ul3, True: client_ul3_cmpr}}

# Load parameters
CONCURRENCY = 5
REQUESTS_PER_THREAD = 100


# -----------------------------------------------------
# MAIN TEST LOOP
# -----------------------------------------------------

if __name__ == "__main__":
    results = {
        "compressed": defaultdict(lambda: defaultdict(lambda: {"x": [], "y": []})),
        "uncompressed": defaultdict(lambda: defaultdict(lambda: {"x": [], "y": []})),
    }

    N = len(files)
    client_names = list(clients.keys())

    for i in range(N):
        file_list = files[: i + 1]

        print("-" * 40)
        print(f"{len(file_list)} files under parallel load...")
        print(f"Threads = {CONCURRENCY}, Requests per thread = {REQUESTS_PER_THREAD}\n")

        params = QueryParams(
            k=k,
            size=size,
            min_score=min_score,
            files=file_list,
        )

        for client_name, client_cfg in clients.items():
            print(f"=== {client_name} ===")

            for compressed, client in client_cfg.items():
                for label, body in [
                    # ("dev", params.dev_body(query_vec)),
                    ("v1", params.v1_body(query_vec)),
                    # ("v2", params.v2_body(query_vec)),
                ]:
                    stats = query_ms_parallel(
                        client=client,
                        body=body,
                        num_threads=CONCURRENCY,
                        num_requests_per_thread=REQUESTS_PER_THREAD,
                    )

                    avg_lat = stats["avg"]

                    print(
                        f"{label}: avg={avg_lat:.2f}ms, "
                        f"p95={stats['p95']:.2f}ms, p99={stats['p99']:.2f}ms, "
                        f"min={stats['min']:.2f}ms, max={stats['max']:.2f}ms "
                        f"(n={stats['count']})"
                    )

                    compression_flag = "compressed" if compressed else "uncompressed"
                    key = (client_name, label)
                    results[compression_flag][client_name][key]["x"].append(len(file_list))
                    results[compression_flag][client_name][key]["y"].append(stats["p95"])

                client.close()

            print()
        print()

    # -------------------------
    # PLOT RESULTS (single plot, all lines)
    # -------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    for comp_key in ["uncompressed", "compressed"]:
        for client_name, series_dict in results[comp_key].items():
            # Only one label now: (client_name, "")
            series = list(series_dict.values())[0]

            ax.plot(
                series["x"],
                series["y"],
                marker="o",
                label=f"{client_name} - {comp_key}",
            )

    ax.set_title("Latency vs Number of Files (v1 only)")
    ax.set_xlabel("Number of Files")
    ax.set_ylabel("Latency (p95, ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    # # -------------------------
    # # PLOT RESULTS (generic)
    # # -------------------------
    # n_clients = len(client_names)
    # fig, axes = plt.subplots(n_clients, 2, sharey=True)

    # # If only one client, axes will not be 2D array
    # if n_clients == 1:
    #     axes = axes.reshape(1, 2)

    # compression_cols = ["uncompressed", "compressed"]
    # compression_titles = ["Without Compression", "With Compression"]

    # for row_idx, client_name in enumerate(client_names):
    #     for col_idx, comp_key in enumerate(compression_cols):

    #         ax = axes[row_idx][col_idx]

    #         for query_label, series in results[comp_key][client_name].items():
    #             ax.plot(
    #                 series["x"],
    #                 series["y"],
    #                 marker="o",
    #                 label=query_label,
    #             )

    #         ax.set_title(f"{client_name}\n{compression_titles[col_idx]}")
    #         ax.set_xlabel("Number of Files")
    #         if col_idx == 0:
    #             ax.set_ylabel("Latency (ms)")

    #         ax.grid(True, alpha=0.3)
    #         ax.legend(fontsize=8)

    # plt.tight_layout()
    # plt.show()
