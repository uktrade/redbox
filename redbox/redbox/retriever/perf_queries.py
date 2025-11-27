from dataclasses import dataclass
import time
import numpy as np
from langchain.embeddings import FakeEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection, Urllib3HttpConnection

np.random.seed(42)

INDEX = "redbox-data-chunk"

client_rhc = OpenSearch("http://localhost:9200", connection_class=RequestsHttpConnection)
client_ul3 = OpenSearch([{"host": "localhost", "port": 9200}], connection_class=Urllib3HttpConnection)
fake = FakeEmbeddings(size=1024)
query_vec = fake.embed_query("fixed query")


def get_k_value(file_list, desired_size=30):
    """
    Simple rule: more files filtered = lower k needed
    """
    num_files = len(file_list)

    if num_files <= 3:
        return desired_size * 3  # k=90 for very restrictive
    elif num_files <= 10:
        return desired_size * 2  # k=60 for restrictive
    elif num_files <= 30:
        return int(desired_size * 1.5)  # k=45 for moderate
    else:
        return desired_size  # k=30 for many files


def tuned_k(num_files, base_k=50):
    # scale k but capped
    return min(base_k + num_files * 5, 200)


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
                        "k": self.k,  # get_k_value(self.files, self.k),
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
                    "filter": [
                        {
                            "terms": {"metadata.uri.keyword": self.files},
                        }
                    ],
                    "must": [
                        {
                            "knn": {
                                "vector_field": {
                                    "vector": vec,
                                    "k": self.k,  # get_k_value(self.files, self.k),
                                    # "filter": {
                                    #     "bool": {
                                    #         "filter": {
                                    #             "terms": {
                                    #                 "metadata.uri.keyword": self.files
                                    #             }
                                    #         }
                                    #     }
                                    # }
                                }
                            }
                        }
                    ],
                }
            },
            "_source": {"excludes": ["vector_field"]},
        }


def query_ms(
    client: OpenSearch,
    body: dict,
    num_tests: int = 50,
):
    test_scores = []
    for test in range(num_tests):
        start = time.perf_counter()
        client.search(index=INDEX, body=body)

        if test < 10:
            score = (time.perf_counter() - start) * 1000  # ms
            test_scores.append(score)

    avg = sum(test_scores) / len(test_scores)
    # std_dev = stdev(test_scores)
    return avg  # f"{avg} +- {std_dev}"


ks = [5, 10, 30, 50]
sizes = [5, 30, 100]
min_scores = [0.0, 0.4, 0.6, 0.8]
files = [
    "harry.wixley@digital.trade.gov.uk/2025-06-12_SUBMISSION_MADDERS_NMW_Coverage_Transparenc_0y9UYks.pdf",
    "harry.wixley@digital.trade.gov.uk/2025-06-12_SUBMISSION_MADDERS_NMW_Coverage_Transparenc_vtxKxl6.pdf",
    "harry.wixley@digital.trade.gov.uk/20250303_OFFSEN_Submission_For_Decision_-_Capping_Pape_A1HaAmt.pdf",
    "harry.wixley@digital.trade.gov.uk/DCE_and_TRQ_Review_2025_Final_Recommendation_Submissio_yL899Qv.pdf",
]

files = files + files

# import matplotlib.pyplot as plt

results = {
    "num_files": [],
    "rhc_dev": [],
    "rhc_v1": [],
    "rhc_v2": [],
    "ul3_dev": [],
    "ul3_v1": [],
    "ul3_v2": [],
}

k = 10
size = 30
min_score = 0.6
for i in range(len(files)):
    file_list = files[: i + 1]
    print("-" * 40)
    print(f"{len(file_list)} files...")

    params = QueryParams(k=k, size=size, min_score=min_score, files=file_list)

    # ---- RHC client ----
    perf_dev_rhc = query_ms(client_rhc, params.dev_body(query_vec))
    perf_v1_rhc = query_ms(client_rhc, params.v1_body(query_vec))
    perf_v2_rhc = query_ms(client_rhc, params.v2_body(query_vec))

    print(f"RHC dev: {perf_dev_rhc}")
    print(f"RHC v1:  {perf_v1_rhc}")
    print(f"RHC v2:  {perf_v2_rhc}")
    print()

    # ---- UL3 client ----
    perf_dev_ul3 = query_ms(client_ul3, params.dev_body(query_vec))
    perf_v1_ul3 = query_ms(client_ul3, params.v1_body(query_vec))
    perf_v2_ul3 = query_ms(client_ul3, params.v2_body(query_vec))

    print(f"UL3 dev: {perf_dev_ul3}")
    print(f"UL3 v1:  {perf_v1_ul3}")
    print(f"UL3 v2:  {perf_v2_ul3}")
    print("\n")

    # ---- Save results ----
    results["num_files"].append(len(file_list))
    results["rhc_dev"].append(perf_dev_rhc)
    results["rhc_v1"].append(perf_v1_rhc)
    results["rhc_v2"].append(perf_v2_rhc)
    results["ul3_dev"].append(perf_dev_ul3)
    results["ul3_v1"].append(perf_v1_ul3)
    results["ul3_v2"].append(perf_v2_ul3)

# plt.figure(figsize=(12, 6))
# x = results["num_files"]

# plt.plot(x, results["rhc_dev"], marker="o", label="RHC - dev")
# plt.plot(x, results["rhc_v1"], marker="o", label="RHC - v1")
# plt.plot(x, results["rhc_v2"], marker="o", label="RHC - v2")
# plt.plot(x, results["ul3_dev"], marker="s", label="UL3 - dev")
# plt.plot(x, results["ul3_v1"], marker="s", label="UL3 - v1")
# plt.plot(x, results["ul3_v2"], marker="s", label="UL3 - v2")

# plt.title("OpenSearch Query Performance by Number of Files")
# plt.xlabel("Number of Files")
# plt.ylabel("Latency (ms)")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()

# k = 10
# size = 30
# min_score = 0.6
# for i in range(len(files)):
#     file_list = files[: i + 1]
#     print("-" * 40)
#     print(f"{len(file_list)} files...")
#     params = QueryParams(k=k, size=size, min_score=min_score, files=file_list)
#     perf_dev = query_ms(client_rhc, params.dev_body(query_vec))
#     perf_v1 = query_ms(client_rhc, params.v1_body(query_vec))
#     perf_v2 = query_ms(client_rhc, params.v2_body(query_vec))

#     print(f"dev: {perf_dev}")
#     print(f"v1: {perf_v1}")
#     print(f"v2: {perf_v2}")
#     print()

#     perf_dev = query_ms(client_ul3, params.dev_body(query_vec))
#     perf_v1 = query_ms(client_ul3, params.v1_body(query_vec))
#     perf_v2 = query_ms(client_ul3, params.v2_body(query_vec))

#     print(f"dev: {perf_dev}")
#     print(f"v1: {perf_v1}")
#     print(f"v2: {perf_v2}")
#     print()
#     print()

# for i in range(len(files)):
#     file_list = files[: i + 1]
#     print("-" * 40)
#     print(f"{len(file_list)} files...")
#     params = QueryParams(k=k, size=size, min_score=min_score, files=file_list)
#     perf_dev = query_ms(params.dev_body(query_vec))
#     perf_v1 = query_ms(params.v1_body(query_vec))
#     perf_v2 = query_ms(params.v2_body(query_vec))

#     print(f"No k tuning - '{k}'...")
#     print(f"dev: {perf_dev}")
#     print(f"v1: {perf_v1}")
#     print(f"v2: {perf_v2}")

#     tuned_k_res = get_k_value(file_list, k)
#     params = QueryParams(k=tuned_k_res, size=size, min_score=min_score, files=file_list)
#     perf_dev = query_ms(params.dev_body(query_vec))
#     perf_v1 = query_ms(params.v1_body(query_vec))
#     perf_v2 = query_ms(params.v2_body(query_vec))

#     print()
#     print(f"K tuning get_k_value - '{tuned_k_res}'...")
#     print(f"dev: {perf_dev}")
#     print(f"v1: {perf_v1}")
#     print(f"v2: {perf_v2}")
#     print()

#     tuned_k_res = tuned_k(len(file_list), k)
#     params = QueryParams(k=tuned_k_res, size=size, min_score=min_score, files=file_list)
#     perf_dev = query_ms(params.dev_body(query_vec))
#     perf_v1 = query_ms(params.v1_body(query_vec))
#     perf_v2 = query_ms(params.v2_body(query_vec))

#     print(f"K tuning tuned_k - '{tuned_k_res}'...")
#     print(f"dev: {perf_dev}")
#     print(f"v1: {perf_v1}")
#     print(f"v2: {perf_v2}")
#     print()
#     print()
