#!/usr/bin/env python3
# ruff: noqa: E402
"""
End-to-end RAG generation evaluation runner.

Executes the full production pipeline for each question in the eval dataset:
  1. Query condensation — rewrites the question using CONDENSE_SYSTEM_PROMPT
  2. Retrieval          — runs ParameterisedElasticsearchRetriever
  3. Answer generation  — calls the LLM with RETRIEVAL_SYSTEM_PROMPT + retrieved chunks
  4. LLM judge         — scores faithfulness, answer_relevance, context_precision, context_recall

Can be executed directly for manual runs, or imported by conftest.py.

Usage:
    make eval-generation
    # or directly:
    poetry run python tests/evaluation/run_generation_eval.py --skip-ingest
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap — MUST come before any redbox.* imports.
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_redbox_root = str(Path(__file__).resolve().parents[2])
sys.path = [_redbox_root] + [p for p in sys.path if p != _redbox_root]

load_dotenv(Path(__file__).parent / ".env.eval", override=False)
os.environ["DJANGO_SETTINGS_MODULE"] = "tests.evaluation.django_settings"

import django
import django.conf

if not django.conf.settings.configured:
    django.setup()

# ---------------------------------------------------------------------------
# Imports (safe after Django is configured)
# ---------------------------------------------------------------------------
import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from tests.evaluation.eval_pipeline import (
    CORPUS_DIR,
    DATASET_PATH,
    EVAL_DIR,
    EVAL_S3_PREFIX,
    build_env,
    build_vector_store,
    cleanup_corpus,
    ingest_corpus,
    make_eval_state,
    make_retriever,
)
from tests.evaluation.metrics.generation import (
    AggregateGenerationScores,
    GenerationScores,
    aggregate_generation,
    evaluate_answer_relevance,
    evaluate_context_precision,
    evaluate_context_recall,
    evaluate_faithfulness,
)

from redbox.chains.components import get_chat_llm, get_embeddings
from redbox.models.chain import AISettings
from redbox.models.prompts import CONDENSE_QUESTION_PROMPT, RETRIEVAL_QUESTION_PROMPT, RETRIEVAL_SYSTEM_PROMPT
from redbox.models.settings import Settings
from redbox.retriever import ParameterisedElasticsearchRetriever

GENERATION_BASELINE_PATH = EVAL_DIR / "baselines" / "generation_baseline.json"
GENERATION_REPORT_DIR = EVAL_DIR / "reports"

# ---------------------------------------------------------------------------
# Generation quality thresholds for the regression gate
# ---------------------------------------------------------------------------
GENERATION_REGRESSION_TOLERANCE = 0.05  # 5 pp drop triggers failure
GENERATION_FAITHFULNESS_THRESHOLD = 0.70
GENERATION_RELEVANCE_THRESHOLD = 0.70
GENERATION_PRECISION_THRESHOLD = 0.40
GENERATION_RECALL_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def condense_query(question: str, llm, ai_settings: AISettings | None = None) -> list[str]:
    """
    Run the production condensation step: CONDENSE_SYSTEM_PROMPT + CONDENSE_QUESTION_PROMPT.

    Returns a list of condensed standalone questions. If condensation fails or
    the model returns 'I don't know', falls back to [question].
    """
    ai = ai_settings or AISettings()
    system = ai.condense_system_prompt
    human = CONDENSE_QUESTION_PROMPT.format(question=question)

    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        text = response.content.strip()

        if not text or "i don't know" in text.lower():
            return [question]

        # Parse: numbered list (1. / 1) / - ) or newline-separated
        lines = re.split(r"\n+", text)
        cleaned = []
        for line in lines:
            line = re.sub(r"^\s*(\d+[\.\)]\s*|[-•]\s*)", "", line).strip()
            if line and len(line) > 10:
                cleaned.append(line)
        return cleaned if cleaned else [question]
    except Exception:
        return [question]


def _format_docs_for_prompt(docs: list[Document]) -> str:
    """Format retrieved documents for insertion into RETRIEVAL_SYSTEM_PROMPT."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        uri = doc.metadata.get("uri", doc.metadata.get("source", f"doc_{i}"))
        parts.append(f"[Document {i}]\nSource: {uri}\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def retrieve_with_condensation(
    question: str,
    condensed_queries: list[str],
    all_uris: list[str],
    retriever: ParameterisedElasticsearchRetriever,
    ai_settings: AISettings | None = None,
) -> list[Document]:
    """
    Retrieve documents for each condensed query, merge results, deduplicate by content.
    """
    seen: set[str] = set()
    merged: list[Document] = []

    for cq in condensed_queries:
        state = make_eval_state(cq, all_uris, ai_settings=ai_settings)
        docs = retriever.invoke(state)
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

    return merged


def generate_answer(
    question: str,
    retrieved_docs: list[Document],
    llm,
    ai_settings: AISettings | None = None,
) -> tuple[str, list]:
    """
    Run the production answer generation step.

    Uses RETRIEVAL_SYSTEM_PROMPT (with {formatted_documents}) and
    RETRIEVAL_QUESTION_PROMPT. Returns (answer_text, citations_list).
    """
    formatted_documents = _format_docs_for_prompt(retrieved_docs)
    system = RETRIEVAL_SYSTEM_PROMPT.format(formatted_documents=formatted_documents)
    human = RETRIEVAL_QUESTION_PROMPT.format(question=question)

    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        text = response.content.strip()

        # Try to parse JSON output {answer, citations}
        json_match = re.search(r'\{.*"answer"\s*:.*\}', text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed.get("answer", text), parsed.get("citations", [])
        return text, []
    except Exception:
        return "", []


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Artifacts from one end-to-end pipeline run for a single question."""

    question_id: str
    question: str
    condensed_queries: list[str]
    retrieved_docs: list[Document]
    answer: str
    citations: list
    expected_answer: str = ""


def run_pipeline_for_entry(
    entry: dict,
    all_uris: list[str],
    retriever: ParameterisedElasticsearchRetriever,
    llm,
    ai_settings: AISettings | None = None,
) -> PipelineResult:
    """Run the full condense → retrieve → generate pipeline for one dataset entry."""
    question = entry["question"]
    condensed = condense_query(question, llm, ai_settings=ai_settings)
    docs = retrieve_with_condensation(question, condensed, all_uris, retriever, ai_settings=ai_settings)
    answer, citations = generate_answer(question, docs, llm, ai_settings=ai_settings)
    return PipelineResult(
        question_id=entry["id"],
        question=question,
        condensed_queries=condensed,
        retrieved_docs=docs,
        answer=answer,
        citations=citations,
        expected_answer=entry.get("expected_answer", ""),
    )


def score_pipeline_result(
    result: PipelineResult,
    judge_llm,
) -> GenerationScores:
    """Run all four LLM judges on a single pipeline result."""
    faithfulness, f_raw = evaluate_faithfulness(judge_llm, result.answer, result.retrieved_docs)
    relevance, r_raw = evaluate_answer_relevance(judge_llm, result.question, result.answer)
    precision, p_raw = evaluate_context_precision(judge_llm, result.question, result.answer, result.retrieved_docs)
    recall, rc_raw = evaluate_context_recall(judge_llm, result.question, result.expected_answer, result.retrieved_docs)

    return GenerationScores(
        question_id=result.question_id,
        faithfulness=faithfulness,
        answer_relevance=relevance,
        context_precision=precision,
        context_recall=recall,
        condensed_query="; ".join(result.condensed_queries),
        answer=result.answer,
        citation_count=len(result.citations),
        answer_length_words=len(result.answer.split()),
        num_retrieved=len(result.retrieved_docs),
        faithfulness_raw=f_raw,
        answer_relevance_raw=r_raw,
        context_precision_raw=p_raw,
        context_recall_raw=rc_raw,
    )


def run_generation_eval(
    retriever: ParameterisedElasticsearchRetriever,
    judge_llm,
    dataset: list[dict],
    uri_map: dict[str, str],
    env: Settings,
    *,
    ai_settings: AISettings | None = None,
    verbose: bool = False,
) -> tuple[list[GenerationScores], AggregateGenerationScores]:
    """
    Run the full generation evaluation pipeline.

    Returns (per_question_scores, aggregate_scores).
    """
    all_uris = list(uri_map.values())
    scores: list[GenerationScores] = []

    for i, entry in enumerate(dataset, start=1):
        if verbose:
            print(f"  [{i}/{len(dataset)}] {entry['id']}: condensing…", flush=True)
        result = run_pipeline_for_entry(entry, all_uris, retriever, judge_llm, ai_settings=ai_settings)

        if verbose:
            print(f"  [{i}/{len(dataset)}] {entry['id']}: judging ({len(result.retrieved_docs)} chunks)…", flush=True)
        s = score_pipeline_result(result, judge_llm)
        scores.append(s)
        if verbose:
            print(
                f"  [{i}/{len(dataset)}] {entry['id']}: "
                f"faith={s.faithfulness:.2f} rel={s.answer_relevance:.2f} "
                f"prec={s.context_precision:.2f} rec={s.context_recall:.2f}",
                flush=True,
            )

    agg = aggregate_generation(scores)
    return scores, agg


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_generation_report(
    scores: list[GenerationScores],
    agg: AggregateGenerationScores,
    env: Settings,
    ai_settings: AISettings | None = None,
) -> Path:
    """Write JSON + Markdown + HTML generation reports to the reports directory."""
    return _write_json_report(scores, agg, env)


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------


def compare_generation_to_baseline(
    agg: AggregateGenerationScores,
    baseline: dict,
    tolerance: float = GENERATION_REGRESSION_TOLERANCE,
) -> list[str]:
    regressions = []
    baseline_agg = baseline.get("aggregate", {})
    for metric in ("faithfulness", "answer_relevance", "context_precision", "context_recall"):
        current = getattr(agg, metric)
        base = baseline_agg.get(metric)
        if base is not None and current < base - tolerance:
            regressions.append(
                f"{metric}: {current:.3f} < baseline {base:.3f} (drop {base - current:.3f} > tolerance {tolerance:.3f})"
            )
    return regressions


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end generation eval pipeline.")
    parser.add_argument("--skip-ingest", action="store_true", help="Use existing index")
    parser.add_argument("--no-cleanup", action="store_true", help="Leave index and S3 objects after run")
    parser.add_argument("--baseline", type=Path, default=GENERATION_BASELINE_PATH)
    args = parser.parse_args()

    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {CORPUS_DIR}. Add corpus PDFs before running.")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    baseline = {}
    if args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    print(f"\nRedbox generation eval — {len(dataset)} questions, {len(pdfs)} PDF(s)\n")

    env = build_env()
    es = env.elasticsearch_client()
    embeddings = get_embeddings(env)
    vstore = build_vector_store(env, embeddings)
    llm = get_chat_llm(AISettings().chat_backend)

    uploaded_keys: list[str] = []
    if args.skip_ingest:
        print("Skipping ingestion (--skip-ingest).\n")
        uri_map = {p.stem: f"{EVAL_S3_PREFIX}/{p.name}" for p in pdfs}
    else:
        print("Ingesting corpus …")
        uri_map, uploaded_keys = ingest_corpus(env, es, vstore, verbose=True)

    try:
        print("\nRunning generation eval …")
        retriever = make_retriever(env, es, embeddings)
        scores, agg = run_generation_eval(retriever, llm, dataset, uri_map, env, verbose=True)

        # Print summary
        w = 60
        print("\n" + "=" * w)
        print("Generation Evaluation Summary")
        print("=" * w)
        print(f"{'Faithfulness':<25} {agg.faithfulness:>8.3f}  (target > {GENERATION_FAITHFULNESS_THRESHOLD})")
        print(f"{'Answer Relevance':<25} {agg.answer_relevance:>8.3f}  (target > {GENERATION_RELEVANCE_THRESHOLD})")
        print(f"{'Context Precision':<25} {agg.context_precision:>8.3f}  (target > {GENERATION_PRECISION_THRESHOLD})")
        print(f"{'Context Recall':<25} {agg.context_recall:>8.3f}  (target > {GENERATION_RECALL_THRESHOLD})")
        print("=" * w)

        # Write reports
        json_path = _write_json_report(scores, agg, env)
        print(f"\nReport saved to {json_path}")

        if baseline.get("aggregate"):
            regressions = compare_generation_to_baseline(agg, baseline)
            if regressions:
                print("\nREGRESSION DETECTED:")
                for r in regressions:
                    print(f"  • {r}")
                sys.exit(1)
            else:
                print("No regression vs baseline.")
        else:
            print("No generation baseline found. After reviewing results, run:\n  make eval-update-generation-baseline")
    finally:
        if not args.no_cleanup:
            cleanup_corpus(env, es, uploaded_keys)


def _write_json_report(
    scores: list[GenerationScores],
    agg: AggregateGenerationScores,
    env: Settings,
) -> Path:
    from tests.evaluation.metrics.generation_report import build_generation_html, build_generation_markdown

    report_dir = GENERATION_REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    import subprocess

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"

    payload = {
        "meta": {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": sha,
            "embedding_model": env.embedding_backend,
        },
        "aggregate": agg.to_dict(),
        "per_question": [s.to_dict() for s in scores],
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    json_path = report_dir / f"generation_report_{timestamp}.json"

    json_path.write_text(json.dumps(payload, indent=2))
    (report_dir / "generation_report_latest.json").write_text(json.dumps(payload, indent=2))
    (report_dir / "generation_report_latest.md").write_text(build_generation_markdown(scores, agg, payload["meta"]))
    (report_dir / "generation_report_latest.html").write_text(build_generation_html(scores, agg, payload["meta"]))

    return json_path


if __name__ == "__main__":
    main()
