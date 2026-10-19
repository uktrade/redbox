# Test ingestion logic
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
from tests.evaluation.eval_pipeline import eval_entries_values, make_eval_state
from tests.evaluation.metrics.retrieval import RetrievalScores, compute_scores


@pytest.mark.ai
def test_retrieval_corpus_ingested(seeded_corpus: dict) -> None:
    """Verify every PDF in dataset/corpus/ was successfully ingested."""
    pdf_count = len(list((Path(__file__).parent / "dataset" / "corpus").glob("*.pdf")))
    assert pdf_count > 0
    assert len(seeded_corpus) == pdf_count, f"Expected {pdf_count} corpus documents, got {len(seeded_corpus)}."


@pytest.mark.ai
def test_retrieval_all_questions(eval_dataset, retrieval_results, baseline, eval_report):
    failures = []
    for entry in eval_dataset:
        scores = retrieval_results[entry["id"]]  # read from cache
        if scores.hit_at_30 == 0.0:
            failures.append(f"[{entry['id']} / {entry.get('difficulty')}] ...")
    assert not failures, ...


@pytest.mark.ai
@pytest.mark.parametrize("eval_entry", eval_entries_values, ids=lambda e: e["id"])
def test_retrieval_per_question(eval_entry, retrieval_results):
    scores = retrieval_results[eval_entry["id"]]  # read from cache
    k_threshold = 10 if eval_entry.get("difficulty") == "hard" else 5
    assert scores.first_relevant_rank is not None, ...
    assert scores.first_relevant_rank <= k_threshold, ...


# Ablation: Gaussian re-ranking ON vs OFF
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
        w = sum(getattr(s, metric) for s in with_g) / n
        wo = sum(getattr(s, metric) for s in without_g) / n
        rows.append(f"{metric:<20} {w:>15.3f} {wo:>10.3f} {w - wo:>+8.3f}")

    table = "\n".join(["", "Gaussian ablation results:", header, separator] + rows)
    print(table)

    ablation_path = Path(__file__).parent / "reports" / "ablation_latest.txt"
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_path.write_text(table + "\n")
    print(f"  (saved to {ablation_path})")
