from dataclasses import dataclass
import time
import numpy as np
from langchain.embeddings import FakeEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection, Urllib3HttpConnection

np.random.seed(42)

INDEX = "redbox-data-chunk"

client_rhc = OpenSearch("http://localhost:9200", connection_class=RequestsHttpConnection, compression=False)
client_rhc_cmpr = OpenSearch("http://localhost:9200", connection_class=RequestsHttpConnection, compression=True)
client_ul3 = OpenSearch(
    [{"host": "localhost", "port": 9200}], connection_class=Urllib3HttpConnection, compression=False
)
client_ul3_cmpr = OpenSearch(
    [{"host": "localhost", "port": 9200}], connection_class=Urllib3HttpConnection, compression=True
)

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
                                    "k": self.k,
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
    return avg


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

k = 10
size = 30
min_score = 0.6
clients = {
    "RHC": client_rhc,
    "RHC [Compression=True]": client_rhc_cmpr,
    "UL3": client_ul3,
    "UL3 [Compression=True]": client_ul3_cmpr,
}


for i in range(len(files)):
    file_list = files[: i + 1]
    print("-" * 40)
    print(f"{len(file_list)} files...")

    params = QueryParams(k=k, size=size, min_score=min_score, files=file_list)

    for name, client in clients.items():
        perf_dev = query_ms(client, params.dev_body(query_vec))
        perf_v1 = query_ms(client, params.v1_body(query_vec))
        perf_v2 = query_ms(client, params.v2_body(query_vec))

        print(f"{name} dev: {perf_dev}")
        print(f"{name} v1:  {perf_v1}")
        print(f"{name} v2:  {perf_v2}")
        print()
    print()
