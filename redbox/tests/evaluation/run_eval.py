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
from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap — MUST come before any redbox.* imports.
# The dotenv plugin (tryfirst) loads tests/.env.test from the repo root which
# sets DJANGO_SETTINGS_MODULE=redbox_app.settings. 
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure redbox package root is first in sys.path.
# When run as a script Python inserts the script's own directory (tests/evaluation/)
# at sys.path[0], which would shadow the correct package root for absolute imports
# like tests.evaluation.django_settings. We always move it to position 0.
_redbox_root = str(Path(__file__).resolve().parents[2])  # …/redbox/redbox
sys.path = [_redbox_root] + [p for p in sys.path if p != _redbox_root]

load_dotenv(Path(__file__).parent / ".env.eval", override=False)
os.environ["DJANGO_SETTINGS_MODULE"] = "tests.evaluation.django_settings"

import django
import django.conf

if not django.conf.settings.configured:
    django.setup()

# ---------------------------------------------------------------------------
# Production pipeline imports (safe after Django is configured)
# ---------------------------------------------------------------------------
import argparse
import json
from uuid import uuid4

import boto3
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.messages import HumanMessage
from opensearchpy import OpenSearch
from tests.evaluation.metrics.report import EvalReport, compare_to_baseline
from tests.evaluation.metrics.retrieval import compute_scores

from redbox.chains.components import get_embeddings
from redbox.loader.chunking.service import DocumentChunkingService
from redbox.loader.extraction.metadata import MetadataExtraction
from redbox.loader.extraction.service import DocumentExtractionService
from redbox.models.chain import AISettings, RedboxQuery, RedboxState
from redbox.models.file import ChunkResolution
from redbox.models.settings import Settings
from redbox.retriever import ParameterisedElasticsearchRetriever

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

EVAL_DIR     = Path(__file__).parent
CORPUS_DIR   = EVAL_DIR / "dataset" / "corpus"
DATASET_PATH = EVAL_DIR / "dataset" / "retrieval_eval_set.json"
BASELINE_PATH = EVAL_DIR / "baselines" / "baseline.json"

EVAL_INDEX     = "redbox-data-eval"
EVAL_S3_PREFIX = "eval-corpus"


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def build_env() -> Settings:
    return Settings(django_secret_key="", postgres_password="")


def build_vector_store(env: Settings, embeddings) -> OpenSearchVectorSearch:
    return OpenSearchVectorSearch(
        index_name=EVAL_INDEX,
        opensearch_url=env.elastic.collection_endpoint,
        embedding_function=embeddings,
        query_field="text",
        vector_query_field=env.embedding_document_field_name,
        engine="lucene",
    )


def ingest_corpus(
    env: Settings,
    es: OpenSearch,
    vector_store: OpenSearchVectorSearch,
    *,
    verbose: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """
    Ingest every PDF in CORPUS_DIR through the full production pipeline.

    When verbose=False (default) only one compact line per PDF is printed to
    stdout; full step-level detail is always written to reports/ingest_latest.log.

    Returns:
        uri_map:       {pdf_stem: s3_key}
        uploaded_keys: S3 keys created, for cleanup
    """
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    log_file = EVAL_DIR / "reports" / "ingest_latest.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as _lf:
        def _step(msg: str) -> None:
            """Detail line — always logged, only printed when verbose."""
            _lf.write(msg + "\n")
            if verbose:
                print(msg, flush=True)

        def _out(msg: str) -> None:
            """Summary line — always printed and always logged."""
            _lf.write(msg + "\n")
            print(msg, flush=True)

        if es.indices.exists(index=EVAL_INDEX):
            es.indices.delete(index=EVAL_INDEX)
        es.indices.create(index=EVAL_INDEX, body=env.index_mapping)

        _out(f"Ingesting corpus ({len(pdfs)} PDF(s)) into {EVAL_INDEX}  (log → {log_file.name})")

        s3 = boto3.client("s3", region_name=env.aws_region)
        extraction_svc = DocumentExtractionService(bucket=env.bucket_name, region=env.aws_region)
        chunking_svc = DocumentChunkingService(
            chunk_resolution=ChunkResolution.normal,
            min_chunk_size=env.worker_ingest_min_chunk_size,
            max_chunk_size=env.worker_ingest_max_chunk_size,
            overlap_chars=0,
        )

        uploaded_keys: list[str] = []
        uri_map: dict[str, str] = {}

        for i, pdf in enumerate(pdfs, start=1):
            s3_key = f"{EVAL_S3_PREFIX}/{pdf.name}"
            _step(f"  [{pdf.name}] uploading …")
            s3.upload_file(str(pdf), env.bucket_name, s3_key)
            uploaded_keys.append(s3_key)

            _step(f"  [{pdf.name}] extracting …")
            strat, elements = extraction_svc.extract(s3_key)
            if not elements:
                _out(f"  [{i}/{len(pdfs)}] {pdf.name}  WARNING: no content extracted — skipping")
                continue

            _step(f"  [{pdf.name}] generating metadata …")
            metadata = MetadataExtraction(env=env).extract(file_name=s3_key, elements=elements)

            _step(f"  [{pdf.name}] chunking …")
            _cstrat, chunk_iter = chunking_svc.chunks(
                s3_key=s3_key, elements=elements,
                generated_metadata=metadata, chunks_overlap_pages=False,
            )
            docs = list(chunk_iter)
            if not docs:
                _out(f"  [{i}/{len(pdfs)}] {pdf.name}  WARNING: no chunks produced — skipping")
                continue

            _step(f"  [{pdf.name}] embedding + indexing {len(docs)} chunks …")
            vector_store.add_documents(docs)
            uri_map[pdf.stem] = s3_key

            _out(f"  [{i}/{len(pdfs)}] {pdf.name}  →  {strat} / {len(docs)} chunks  ✓")

        es.indices.refresh(index=EVAL_INDEX)
        _out(f"Corpus ready: {len(uri_map)} document(s) indexed.")

    return uri_map, uploaded_keys


def cleanup_corpus(env: Settings, es: OpenSearch, uploaded_keys: list[str]) -> None:
    if es.indices.exists(index=EVAL_INDEX):
        es.indices.delete(index=EVAL_INDEX)
    s3 = boto3.client("s3", region_name=env.aws_region)
    for key in uploaded_keys:
        try:
            s3.delete_object(Bucket=env.bucket_name, Key=key)
        except Exception:
            pass


def make_retriever(
    env: Settings,
    es: OpenSearch,
    embeddings,
    *,
    enable_document_query: bool = True,
) -> ParameterisedElasticsearchRetriever:
    return ParameterisedElasticsearchRetriever(
        es_client=es,
        index_name=EVAL_INDEX,
        embedding_model=embeddings,
        embedding_field_name=env.embedding_document_field_name,
        enable_document_query=enable_document_query,
    )


def make_eval_state(question: str, all_uris: list[str]) -> RedboxState:
    query = RedboxQuery(
        question=question,
        s3_keys=all_uris,
        user_uuid=uuid4(),
        chat_history=[],
        ai_settings=AISettings(),
        permitted_s3_keys=all_uris,
    )
    return RedboxState(request=query, messages=[HumanMessage(content=question)])


def run_eval(
    retriever: ParameterisedElasticsearchRetriever,
    dataset: list[dict],
    uri_map: dict[str, str],
    env: Settings,
) -> EvalReport:
    ai = AISettings()
    report = EvalReport(rag_params={
        "rag_k":              ai.rag_k,
        "rag_num_candidates": ai.rag_num_candidates,
        "min_score":          0.6,
        "rag_gauss_scale_size":  ai.rag_gauss_scale_size,
        "rag_gauss_scale_decay": ai.rag_gauss_scale_decay,
        "embedding_model":    env.embedding_backend,
    })

    all_uris = list(uri_map.values())
    for entry in dataset:
        state     = make_eval_state(entry["question"], all_uris)
        retrieved = retriever.invoke(state)
        scores    = compute_scores(entry["id"], retrieved, entry["relevant_snippets"])
        report.record(scores, difficulty=entry.get("difficulty", "unknown"), question=entry.get("question", ""))

    return report


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval eval pipeline.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip corpus ingestion (use existing index)")
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH,
                        help="Path to baseline.json for regression check")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Leave the eval index and S3 objects after the run")
    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    baseline = {}
    if args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(
            f"No PDFs found in {CORPUS_DIR}.\n"
            "Add corpus PDFs (e.g. cptpp_impact_assessment.pdf) before running."
        )

    print(f"\nRedbox retrieval eval — {len(dataset)} questions, {len(pdfs)} PDF(s)\n")

    env        = build_env()
    es         = env.elasticsearch_client()
    embeddings = get_embeddings(env)
    vstore     = build_vector_store(env, embeddings)

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
        report    = run_eval(retriever, dataset, uri_map, env)
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
            print(
                "No baseline scores found. "
                "After reviewing the report, run:\n"
                "  make eval-update-baseline"
            )
    finally:
        if not args.no_cleanup:
            cleanup_corpus(env, es, uploaded_keys)


if __name__ == "__main__":
    main()
