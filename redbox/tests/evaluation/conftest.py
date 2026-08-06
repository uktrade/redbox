"""
Pytest fixtures for the retrieval evaluation harness.

All pipeline logic and Django bootstrap live in run_eval.py.
Importing run_eval as the first action here triggers Django setup
before any other redbox.* modules load.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest
from opensearchpy import OpenSearch

# run_eval must be the first import — it bootstraps Django + env before redbox.* loads.
from tests.evaluation.run_eval import (
    BASELINE_PATH,
    CORPUS_DIR,
    DATASET_PATH,
    build_env,
    build_vector_store,
    cleanup_corpus,
    ingest_corpus,
    make_retriever,
)
from tests.evaluation.metrics.report import EvalReport
from redbox.chains.components import get_embeddings
from redbox.models.chain import AISettings
from redbox.models.settings import Settings
from redbox.retriever import ParameterisedElasticsearchRetriever


@pytest.fixture(scope="session")
def eval_env() -> Settings:
    return build_env()


@pytest.fixture(scope="session")
def eval_es_client(eval_env: Settings) -> OpenSearch:
    return eval_env.elasticsearch_client()


@pytest.fixture(scope="session")
def eval_embeddings(eval_env: Settings):
    return get_embeddings(eval_env)


@pytest.fixture(scope="session")
def eval_vector_store(eval_env: Settings, eval_embeddings, eval_es_client: OpenSearch):
    return build_vector_store(eval_env, eval_embeddings)


@pytest.fixture(scope="session")
def seeded_corpus(
    eval_env: Settings,
    eval_es_client: OpenSearch,
    eval_vector_store,
) -> Generator[dict[str, str], None, None]:
    if not list(CORPUS_DIR.glob("*.pdf")):
        pytest.skip(
            f"No PDFs in {CORPUS_DIR}. "
            "Add corpus PDFs (e.g. cptpp_impact_assessment.pdf) before running."
        )
    uri_map, uploaded_keys = ingest_corpus(eval_env, eval_es_client, eval_vector_store)
    yield uri_map
    cleanup_corpus(eval_env, eval_es_client, uploaded_keys)


@pytest.fixture(scope="session")
def seeded_retriever(
    eval_env: Settings,
    eval_es_client: OpenSearch,
    eval_embeddings,
    seeded_corpus: dict,
) -> ParameterisedElasticsearchRetriever:
    return make_retriever(eval_env, eval_es_client, eval_embeddings)


@pytest.fixture(scope="session")
def seeded_retriever_no_gaussian(
    eval_env: Settings,
    eval_es_client: OpenSearch,
    eval_embeddings,
    seeded_corpus: dict,
) -> ParameterisedElasticsearchRetriever:
    return make_retriever(eval_env, eval_es_client, eval_embeddings, enable_document_query=False)


@pytest.fixture(scope="session")
def eval_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def baseline() -> dict:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {}


@pytest.fixture(scope="session")
def eval_report(eval_env: Settings) -> Generator[EvalReport, None, None]:
    ai = AISettings()
    report = EvalReport(rag_params={
        "rag_k":                 ai.rag_k,
        "rag_num_candidates":    ai.rag_num_candidates,
        "min_score":             0.6,
        "rag_gauss_scale_size":  ai.rag_gauss_scale_size,
        "rag_gauss_scale_decay": ai.rag_gauss_scale_decay,
        "embedding_model":       eval_env.embedding_backend,
    })
    yield report
    report.write()
