#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tests.evaluation.eval_pipeline import (
    BASELINE_PATH,
    CORPUS_DIR,
    DATASET_PATH,
    EVAL_S3_PREFIX,
    build_env,
    build_vector_store,
    cleanup_corpus,
    compare_to_baseline,
    get_embeddings,
    ingest_corpus,
    make_retriever,
    run_eval,
)


# Standalone entry point
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval eval pipeline.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip corpus ingestion (use existing index)")
    parser.add_argument(
        "--baseline", type=Path, default=BASELINE_PATH, help="Path to baseline.json for regression check"
    )
    parser.add_argument("--no-cleanup", action="store_true", help="Leave the eval index and S3 objects after the run")
    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    baseline = {}
    if args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {CORPUS_DIR}.\nAdd corpus PDFs (e.g. cptpp_impact_assessment.pdf) before running.")

    print(f"\nRedbox retrieval eval — {len(dataset)} questions, {len(pdfs)} PDF(s)\n")

    env = build_env()
    es = env.elasticsearch_client()
    embeddings = get_embeddings(env)
    vstore = build_vector_store(env, embeddings)

    uploaded_keys: list[str] = []
    if args.skip_ingest:
        print("Skipping ingestion (--skip-ingest), using existing index.\n")
        uri_map = {p.stem: f"{EVAL_S3_PREFIX}/{p.name}" for p in pdfs}
    else:
        print("Ingesting corpus …")
        uri_map, uploaded_keys = ingest_corpus(env, es, vstore, verbose=False)

    try:
        print("\nRunning retrieval eval …")
        retriever = make_retriever(env, es, embeddings)
        report = run_eval(retriever, dataset, uri_map, env)
        json_path = report.write()
        print(f"\nReport saved to {json_path}")

        if baseline.get("aggregate"):
            regressions = compare_to_baseline(report.aggregate(), baseline)
            if regressions:
                print("\nREGRESSION DETECTED:")
                for r in regressions:
                    print(f"  • {r}")
                sys.exit(1)
            else:
                print("No regression vs baseline.")
        else:
            print("No baseline scores found. After reviewing the report, run:\n  make eval-update-baseline")
    finally:
        if not args.no_cleanup:
            cleanup_corpus(env, es, uploaded_keys)


if __name__ == "__main__":
    main()
