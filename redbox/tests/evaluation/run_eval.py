#!/usr/bin/env python3
"""
Standalone RAG retrieval evaluation runner.

Can be executed directly for manual runs, or imported by conftest.py
so pytest tests use the same pipeline functions.

Usage:
    # Full eval run
    make eval-retrieval
    # or directly:
    poetry run python tests/evaluation/run_eval.py

    # Skip ingestion (index already populated from a previous run)
    poetry run python tests/evaluation/run_eval.py --skip-ingest

    # Compare against a specific baseline
    poetry run python tests/evaluation/run_eval.py --baseline baselines/baseline.json
"""

import argparse
import sys

# Importing eval_pipeline immediately executes Django bootstrapping safely
from tests.evaluation.eval_pipeline import CORPUS_DIR, build_env, build_vector_store, cleanup_corpus, ingest_corpus

from redbox.chains.components import get_embeddings


# Standalone entry point
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cleanup", action="store_true")
    args = parser.parse_args()

    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {CORPUS_DIR}.")

    env = build_env()
    es = env.elasticsearch_client()
    embeddings = get_embeddings(env)
    vstore = build_vector_store(env, embeddings)

    uri_map, uploaded_keys = ingest_corpus(env, es, vstore)

    if not args.no_cleanup:
        cleanup_corpus(env, es, uploaded_keys)


if __name__ == "__main__":
    main()
