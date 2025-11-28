from dataclasses import dataclass
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from langchain.embeddings import FakeEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection, Urllib3HttpConnection

# -----------------------------------------------------
# Setup
# -----------------------------------------------------

np.random.seed(42)

INDEX = "redbox-data-chunk"

client_rhc = OpenSearch(
    "http://localhost:9200",
    connection_class=RequestsHttpConnection,
    compression=False,
)
client_rhc_cmpr = OpenSearch(
    "http://localhost:9200",
    connection_class=RequestsHttpConnection,
    compression=True,
)
client_ul3 = OpenSearch(
    [{"host": "localhost", "port": 9200}],
    connection_class=Urllib3HttpConnection,
    compression=False,
)
client_ul3_cmpr = OpenSearch(
    [{"host": "localhost", "port": 9200}],
    connection_class=Urllib3HttpConnection,
    compression=True,
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
    num_requests_per_thread: int = 10,
    warmup: int = 2,
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
    "harry.wixley@digital.trade.gov.uk/2025-06-12_SUBMISSION_MADDERS_NMW_Coverage_Transparenc_0y9UYks.pdf",
    "harry.wixley@digital.trade.gov.uk/2025-06-12_SUBMISSION_MADDERS_NMW_Coverage_Transparenc_vtxKxl6.pdf",
    "harry.wixley@digital.trade.gov.uk/20250303_OFFSEN_Submission_For_Decision_-_Capping_Pape_A1HaAmt.pdf",
    "harry.wixley@digital.trade.gov.uk/DCE_and_TRQ_Review_2025_Final_Recommendation_Submissio_yL899Qv.pdf",
]
files = files + files  # duplicate for more combinations

clients = {
    "RHC": client_rhc,
    "RHC [Compression=True]": client_rhc_cmpr,
    "UL3": client_ul3,
    "UL3 [Compression=True]": client_ul3_cmpr,
}

# Load parameters
CONCURRENCY = 20
REQUESTS_PER_THREAD = 10


# -----------------------------------------------------
# MAIN TEST LOOP
# -----------------------------------------------------

if __name__ == "__main__":
    for i in range(len(files)):
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

        for client_name, client in clients.items():
            print(f"=== {client_name} ===")
            for label, body in [
                ("dev", params.dev_body(query_vec)),
                ("v1", params.v1_body(query_vec)),
                ("v2", params.v2_body(query_vec)),
            ]:
                stats = query_ms_parallel(
                    client=client,
                    body=body,
                    num_threads=CONCURRENCY,
                    num_requests_per_thread=REQUESTS_PER_THREAD,
                    warmup=2,
                )

                print(
                    f"{label}: avg={stats['avg']:.2f}ms, "
                    f"p95={stats['p95']:.2f}ms, p99={stats['p99']:.2f}ms, "
                    f"min={stats['min']:.2f}ms, max={stats['max']:.2f}ms "
                    f"(n={stats['count']})"
                )
            print()
        print()
