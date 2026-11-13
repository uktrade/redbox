"""
End-to-end RAG generation evaluation tests.

Evaluates the full production pipeline:
  query condensation → retrieval → answer generation → LLM judge

Mark: @pytest.mark.ai_generation  (separate from retrieval-only @pytest.mark.ai)

Run:
    cd redbox
    make eval-generation
    # or directly:
    poetry run pytest tests/evaluation/test_generation_eval.py -m ai_generation -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.evaluation.eval_pipeline import eval_entries_values, make_eval_state
from tests.evaluation.metrics.generation import aggregate_generation
from tests.evaluation.metrics.retrieval import compute_scores
from tests.evaluation.run_generation_eval import (
    GENERATION_FAITHFULNESS_THRESHOLD,
    GENERATION_REGRESSION_TOLERANCE,
    compare_generation_to_baseline,
    condense_query,
    run_pipeline_for_entry,
    score_pipeline_result,
)


def _entry_id(entry: dict) -> str:
    return entry["id"]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


@pytest.mark.ai_generation
def test_generation_corpus_ingested(seeded_corpus: dict) -> None:
    """Verify every PDF in dataset/corpus/ was successfully ingested."""
    pdf_count = len(list((Path(__file__).parent / "dataset" / "corpus").glob("*.pdf")))
    assert pdf_count > 0, (
        "No PDFs found in tests/evaluation/dataset/corpus/. Add at least one corpus PDF before running the eval."
    )
    assert len(seeded_corpus) == pdf_count, (
        f"Expected {pdf_count} corpus documents, got {len(seeded_corpus)}. "
        "Check that all PDFs extracted at least one page of text."
    )


# ---------------------------------------------------------------------------
# Query condensation ablation
# ---------------------------------------------------------------------------


@pytest.mark.ai_generation
def test_generation_query_condensation(
    eval_dataset: list[dict],
    seeded_corpus: dict,
    seeded_retriever,
    judge_llm,
) -> None:
    """
    Compare retrieval quality with raw vs condensed queries.

    Does not fail on degraded metrics — prints a comparison table.
    A negative delta would indicate that condensation is hurting retrieval
    and should be investigated.
    """
    all_uris = list(seeded_corpus.values())
    raw_hits, condensed_hits = [], []

    for entry in eval_dataset:
        snippets = entry["relevant_snippets"]
        q = entry["question"]

        # Raw query
        raw_state = make_eval_state(q, all_uris)
        raw_docs = seeded_retriever.invoke(raw_state)
        raw_hits.append(compute_scores(entry["id"], raw_docs, snippets).hit_at_10)

        # Condensed query — use first condensed variant only for direct comparison
        condensed = condense_query(q, judge_llm)
        condensed_q = condensed[0] if condensed else q
        cond_state = make_eval_state(condensed_q, all_uris)
        cond_docs = seeded_retriever.invoke(cond_state)
        condensed_hits.append(compute_scores(entry["id"], cond_docs, snippets).hit_at_10)

    n = len(eval_dataset)
    raw_avg = sum(raw_hits) / n
    cond_avg = sum(condensed_hits) / n
    delta = cond_avg - raw_avg

    print("\nQuery condensation effect on Hit@10:")
    print(f"  Raw query:       {raw_avg:.3f}")
    print(f"  Condensed query: {cond_avg:.3f}")
    print(f"  Delta:           {delta:+.3f}  ({'✓ improved' if delta >= 0 else '✗ degraded — investigate'})")

    ablation_path = Path(__file__).parent / "reports" / "condensation_ablation.txt"
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_path.write_text(
        f"Condensation ablation\n"
        f"Raw Hit@10:       {raw_avg:.3f}\n"
        f"Condensed Hit@10: {cond_avg:.3f}\n"
        f"Delta:            {delta:+.3f}\n"
    )
    print(f"  (saved to {ablation_path})")


# ---------------------------------------------------------------------------
# Full pipeline — all questions
# ---------------------------------------------------------------------------


@pytest.mark.ai_generation
def test_generation_all_questions(
    eval_dataset: list[dict],
    seeded_corpus: dict,
    seeded_retriever,
    judge_llm,
    generation_report,
) -> None:
    """
    Run every question through the full pipeline and record into generation_report.

    Fails if any question produces an empty answer (complete generation failure).
    Does not fail on low metric scores — those are gated in test_generation_faithfulness.
    """
    all_uris = list(seeded_corpus.values())
    empty_answers = []

    for entry in eval_dataset:
        result = run_pipeline_for_entry(entry, all_uris, seeded_retriever, judge_llm)
        s = score_pipeline_result(result, judge_llm)
        generation_report.append(s)

        if not result.answer.strip():
            empty_answers.append(entry["id"])

    if empty_answers:
        pytest.fail(
            f"{len(empty_answers)}/{len(eval_dataset)} questions produced empty answers:\n"
            + "\n".join(f"  • {qid}" for qid in empty_answers)
        )


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------


@pytest.mark.ai_generation
def test_generation_no_regression(generation_report, generation_baseline: dict) -> None:
    """
    Fail if any generation metric drops more than GENERATION_REGRESSION_TOLERANCE vs baseline.
    Skipped when generation_baseline.json has no recorded scores (first run).
    """
    if not generation_baseline.get("aggregate"):
        pytest.skip(
            "generation_baseline.json has no scores. "
            "Run the eval once, review results, then run: make eval-update-generation-baseline"
        )

    agg = aggregate_generation(generation_report)
    regressions = compare_generation_to_baseline(agg, generation_baseline)
    if regressions:
        pytest.fail(
            f"Generation regression detected (tolerance={GENERATION_REGRESSION_TOLERANCE:.0%}):\n"
            + "\n".join(f"  • {r}" for r in regressions)
        )


# ---------------------------------------------------------------------------
# Per-question faithfulness gate (parametrised)
# ---------------------------------------------------------------------------


@pytest.mark.ai_generation
@pytest.mark.parametrize("generation_entry", eval_entries_values, ids=_entry_id)
def test_generation_faithfulness(
    generation_entry: dict,
    seeded_corpus: dict,
    seeded_retriever,
    judge_llm,
) -> None:
    """
    Per-question faithfulness gate.

    Fails if the generated answer contains claims not supported by retrieved chunks.
    This is the primary hallucination guard.
    """
    all_uris = list(seeded_corpus.values())
    result = run_pipeline_for_entry(generation_entry, all_uris, seeded_retriever, judge_llm)
    s = score_pipeline_result(result, judge_llm)

    assert s.faithfulness >= GENERATION_FAITHFULNESS_THRESHOLD, (
        f"[{generation_entry['id']}] Faithfulness {s.faithfulness:.2f} < "
        f"threshold {GENERATION_FAITHFULNESS_THRESHOLD:.2f}\n"
        f"Question: {generation_entry['question']!r}\n"
        f"Condensed: {s.condensed_query!r}\n"
        f"Answer: {s.answer[:300]!r}\n"
        f"Unsupported claims: {s.faithfulness_raw.get('unsupported', [])}"
    )
