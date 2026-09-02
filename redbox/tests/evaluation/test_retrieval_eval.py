"""
Retrieval evaluation tests.

Measures the quality of ParameterisedElasticsearchRetriever against a curated
corpus of PDF documents ingested through the full production pipeline.

Requirements:
  - Live OpenSearch instance
  - AWS credentials: S3, Bedrock (embeddings + LLM), Textract
  - Corpus PDFs in tests/evaluation/dataset/corpus/
  - Golden Q&A in tests/evaluation/dataset/retrieval_eval_set.json

Run:
    cd redbox
    poetry run pytest tests/evaluation/ -m ai -v

Update baseline after a verified improvement:
    poetry run pytest tests/evaluation/ -m ai -v
    cp tests/evaluation/reports/eval_report_latest.json tests/evaluation/baselines/baseline.json
"""
from pathlib import Path

import pytest
from tests.evaluation.metrics.report import (REGRESSION_TOLERANCE,
                                             compare_to_baseline)
from tests.evaluation.metrics.retrieval import RetrievalScores, compute_scores
from tests.evaluation.run_eval import make_eval_state


def _entry_id(entry: dict) -> str:
    return entry["id"]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


@pytest.mark.ai
def test_retrieval_corpus_ingested(seeded_corpus: dict) -> None:
    """Verify every PDF in dataset/corpus/ was successfully ingested."""
    pdf_count = len(list((Path(__file__).parent / "dataset" / "corpus").glob("*.pdf")))
    assert pdf_count > 0, (
        "No PDF files found in tests/evaluation/dataset/corpus/. "
        "Add at least one corpus PDF before running the eval."
    )
    assert len(seeded_corpus) == pdf_count, (
        f"Expected {pdf_count} corpus documents (one per PDF), "
        f"got {len(seeded_corpus)}. "
        "Check that all PDFs extracted at least one page of text."
    )


# ---------------------------------------------------------------------------
# Full question-set eval — accumulates into eval_report
# ---------------------------------------------------------------------------


@pytest.mark.ai
def test_retrieval_all_questions(
    eval_dataset: list[dict],
    seeded_corpus: dict,
    seeded_retriever,
    baseline: dict,
    eval_report,
) -> None:
    """
    Run every question in retrieval_eval_set.json through the retriever.

    Collects per-question scores into eval_report (written at session teardown).
    The test fails only if any question finds zero relevant chunks in top-30 —
    a complete miss, not a ranking issue.
    """
    all_uris = list(seeded_corpus.values())
    failures: list[str] = []

    for entry in eval_dataset:
        state = make_eval_state(entry["question"], all_uris)
        retrieved = seeded_retriever.invoke(state)

        scores = compute_scores(
            question_id=entry["id"],
            retrieved=retrieved,
            relevant_snippets=entry["relevant_snippets"],
        )
        eval_report.record(scores, difficulty=entry.get("difficulty", "unknown"))

        if scores.hit_at_30 == 0.0:
            failures.append(
                f"[{entry['id']} / {entry.get('difficulty')}] "
                f"No relevant chunk in top-30 for: {entry['question']!r}"
            )

    if failures:
        pytest.fail(
            f"{len(failures)}/{len(eval_dataset)} questions returned no relevant chunk:\n"
            + "\n".join(f"  • {f}" for f in failures)
        )


# ---------------------------------------------------------------------------
# Regression gate — must run after test_retrieval_all_questions
# ---------------------------------------------------------------------------


@pytest.mark.ai
def test_retrieval_no_regression(eval_report, baseline: dict) -> None:
    """
    Fail if any key metric drops more than REGRESSION_TOLERANCE (5 pp) vs baseline.

    Skipped when baseline.json has no recorded scores (first run).
    """
    if not baseline.get("aggregate"):
        pytest.skip(
            "baseline.json has no aggregate scores. "
            "Run the eval once, then copy eval_report_latest.json → baselines/baseline.json."
        )

    agg = eval_report.aggregate()
    regressions = compare_to_baseline(agg, baseline)

    if regressions:
        pytest.fail(
            f"Retrieval regression detected (tolerance={REGRESSION_TOLERANCE:.0%}):\n"
            + "\n".join(f"  • {r}" for r in regressions)
        )


# ---------------------------------------------------------------------------
# Per-question granular tests — one test per entry for CI visibility
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    """Dynamically parametrise tests that request the eval_entry fixture."""
    if "eval_entry" in metafunc.fixturenames:
        import json
        dataset_path = Path(__file__).parent / "dataset" / "retrieval_eval_set.json"
        entries = json.loads(dataset_path.read_text())
        metafunc.parametrize("eval_entry", entries, ids=[e["id"] for e in entries])


@pytest.mark.ai
def test_retrieval_per_question(
    eval_entry: dict,
    seeded_corpus: dict,
    seeded_retriever,
) -> None:
    """
    One test per question in retrieval_eval_set.json.

    Pass threshold:
      easy / medium  → relevant chunk must appear in top-5
      hard           → relevant chunk must appear in top-10
    """
    all_uris = list(seeded_corpus.values())
    state = make_eval_state(eval_entry["question"], all_uris)
    retrieved = seeded_retriever.invoke(state)

    scores = compute_scores(
        question_id=eval_entry["id"],
        retrieved=retrieved,
        relevant_snippets=eval_entry["relevant_snippets"],
    )

    difficulty = eval_entry.get("difficulty", "easy")
    k_threshold = 10 if difficulty == "hard" else 5

    assert scores.first_relevant_rank is not None, (
        f"[{eval_entry['id']}] No relevant chunk found in any retrieved results.\n"
        f"Question: {eval_entry['question']!r}\n"
        f"Relevant snippets: {eval_entry['relevant_snippets']}\n"
        f"Retrieved {scores.num_retrieved} chunks."
    )

    assert scores.first_relevant_rank <= k_threshold, (
        f"[{eval_entry['id']} / {difficulty}] Relevant chunk at rank "
        f"{scores.first_relevant_rank}, expected within top-{k_threshold}.\n"
        f"Question: {eval_entry['question']!r}\n"
        f"Score at relevant chunk: {scores.score_at_relevant}"
    )


# ---------------------------------------------------------------------------
# Ablation: Gaussian re-ranking ON vs OFF
# ---------------------------------------------------------------------------


@pytest.mark.ai
def test_retrieval_ablation_gaussian(
    eval_dataset: list[dict],
    seeded_corpus: dict,
    seeded_retriever,
    seeded_retriever_no_gaussian,
) -> None:
    """
    Compare metrics with and without Gaussian re-ranking.

    Does not fail on metric differences — this is an experiment.
    Results are printed to the terminal for review (Should be part of 
    the generated report)

    Requires (metadata.file_name.keyword → metadata.uri.keyword in
    queries.py:345) to be meaningful; without the fix, both configurations
    produce identical results.
    """
    all_uris = list(seeded_corpus.values())
    with_g: list[RetrievalScores] = []
    without_g: list[RetrievalScores] = []

    for entry in eval_dataset:
        state = make_eval_state(entry["question"], all_uris)
        snippets = entry["relevant_snippets"]
        with_g.append(compute_scores(entry["id"], seeded_retriever.invoke(state), snippets))
        without_g.append(compute_scores(entry["id"], seeded_retriever_no_gaussian.invoke(state), snippets))


    n = len(eval_dataset)
    header = f"{'Metric':<20} {'With Gaussian':>15} {'Without':>10} {'Delta':>8}"
    separator = "-" * 56
    rows = []
    for metric in ("hit_at_5", "hit_at_10", "hit_at_30", "mrr", "ndcg_at_10"):
        w  = sum(getattr(s, metric) for s in with_g)  / n
        wo = sum(getattr(s, metric) for s in without_g) / n
        rows.append(f"{metric:<20} {w:>15.3f} {wo:>10.3f} {w - wo:>+8.3f}")


    table = "\n".join(["", "Gaussian ablation results:", header, separator] + rows)
    print(table)


    ablation_path = Path(__file__).parent / "reports" / "ablation_latest.txt"
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_path.write_text(table + "\n")
    print(f"  (saved to {ablation_path})")
