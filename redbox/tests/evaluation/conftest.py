"""
Pytest fixtures for the retrieval evaluation harness.
All pipeline logic and Django bootstrap live in run_eval.py.
Importing run_eval as the first action here triggers Django setup
before any other redbox.* modules load.
"""

from __future__ import annotations

from typing import Generator

import pytest
from opensearchpy import OpenSearch

# eval_pipeline must be the first import — it bootstraps Django + env before redbox.* loads.
from tests.evaluation.eval_pipeline import CORPUS_DIR, build_env, build_vector_store, cleanup_corpus, ingest_corpus

from redbox.chains.components import get_embeddings
from redbox.models.settings import Settings


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
def eval_vector_store(eval_env: Settings, eval_embeddings):
    return build_vector_store(eval_env, eval_embeddings)


@pytest.fixture(scope="session")
def seeded_corpus(
    eval_env: Settings,
    eval_es_client: OpenSearch,
    eval_vector_store,
) -> Generator[dict[str, str], None, None]:
    if not list(CORPUS_DIR.glob("*.pdf")):
        pytest.skip(f"No PDFs in {CORPUS_DIR}.")
    uri_map, uploaded_keys = ingest_corpus(eval_env, eval_es_client, eval_vector_store)
    yield uri_map
    cleanup_corpus(eval_env, eval_es_client, uploaded_keys)
