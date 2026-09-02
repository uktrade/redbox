"""
Pure functions for retrieval evaluation metrics.

All functions operate on retrieved Document lists and a list of relevant text
snippets. Relevance is determined by substring match (case-insensitive), which
is deterministic, zero-cost, and requires no LLM judge for the primary signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from langchain_core.documents import Document


def _is_relevant(doc: Document, relevant_snippets: list[str]) -> bool:
    content = doc.page_content.lower()
    return any(snippet.lower() in content for snippet in relevant_snippets)


def _rank_of_first_relevant(docs: list[Document], relevant_snippets: list[str]) -> int | None:
    """1-indexed rank of first relevant doc, or None if not found."""
    for rank, doc in enumerate(docs, start=1):
        if _is_relevant(doc, relevant_snippets):
            return rank
    return None


@dataclass
class RetrievalScores:
    """Per-question retrieval scores at various k values."""

    question_id: str
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    hit_at_30: float = 0.0
    mrr: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mean_score: float = 0.0
    score_at_relevant: float | None = None
    num_retrieved: int = 0
    first_relevant_rank: int | None = None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "hit_at_1": self.hit_at_1,
            "hit_at_5": self.hit_at_5,
            "hit_at_10": self.hit_at_10,
            "hit_at_30": self.hit_at_30,
            "mrr": self.mrr,
            "precision_at_5": self.precision_at_5,
            "precision_at_10": self.precision_at_10,
            "ndcg_at_5": self.ndcg_at_5,
            "ndcg_at_10": self.ndcg_at_10,
            "mean_score": self.mean_score,
            "score_at_relevant": self.score_at_relevant,
            "num_retrieved": self.num_retrieved,
            "first_relevant_rank": self.first_relevant_rank,
        }


def hit_rate_at_k(docs: list[Document], relevant_snippets: list[str], k: int) -> float:
    return float(any(_is_relevant(d, relevant_snippets) for d in docs[:k]))


def mean_reciprocal_rank(docs: list[Document], relevant_snippets: list[str]) -> float:
    rank = _rank_of_first_relevant(docs, relevant_snippets)
    return 1.0 / rank if rank is not None else 0.0


def precision_at_k(docs: list[Document], relevant_snippets: list[str], k: int) -> float:
    if not docs:
        return 0.0
    top_k = docs[:k]
    relevant_count = sum(_is_relevant(d, relevant_snippets) for d in top_k)
    return relevant_count / len(top_k)


def ndcg_at_k(docs: list[Document], relevant_snippets: list[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k with binary relevance."""
    top_k = docs[:k]
    if not top_k:
        return 0.0

    dcg = sum(
        _is_relevant(doc, relevant_snippets) / math.log2(rank + 1)
        for rank, doc in enumerate(top_k, start=1)
    )

    # Ideal DCG: all relevant docs at the top (binary relevance, so just 1/log(2))
    num_relevant = sum(_is_relevant(d, relevant_snippets) for d in docs)
    ideal_k = min(num_relevant, k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))

    return dcg / idcg if idcg > 0 else 0.0


def score_of_first_relevant(docs: list[Document], relevant_snippets: list[str]) -> float | None:
    for doc in docs:
        if _is_relevant(doc, relevant_snippets):
            return doc.metadata.get("score")
    return None


def compute_scores(
    question_id: str,
    retrieved: list[Document],
    relevant_snippets: list[str],
) -> RetrievalScores:
    """Compute all retrieval metrics for a single question."""
    scores_list = [d.metadata.get("score", 0.0) for d in retrieved if d.metadata.get("score") is not None]
    mean_s = sum(scores_list) / len(scores_list) if scores_list else 0.0

    return RetrievalScores(
        question_id=question_id,
        hit_at_1=hit_rate_at_k(retrieved, relevant_snippets, 1),
        hit_at_5=hit_rate_at_k(retrieved, relevant_snippets, 5),
        hit_at_10=hit_rate_at_k(retrieved, relevant_snippets, 10),
        hit_at_30=hit_rate_at_k(retrieved, relevant_snippets, 30),
        mrr=mean_reciprocal_rank(retrieved, relevant_snippets),
        precision_at_5=precision_at_k(retrieved, relevant_snippets, 5),
        precision_at_10=precision_at_k(retrieved, relevant_snippets, 10),
        ndcg_at_5=ndcg_at_k(retrieved, relevant_snippets, 5),
        ndcg_at_10=ndcg_at_k(retrieved, relevant_snippets, 10),
        mean_score=mean_s,
        score_at_relevant=score_of_first_relevant(retrieved, relevant_snippets),
        num_retrieved=len(retrieved),
        first_relevant_rank=_rank_of_first_relevant(retrieved, relevant_snippets),
    )


@dataclass
class AggregateScores:
    """Macro-averaged scores across all questions."""

    num_questions: int = 0
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    hit_at_30: float = 0.0
    mrr: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mean_score: float = 0.0
    by_difficulty: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "num_questions": self.num_questions,
            "hit_at_1": round(self.hit_at_1, 4),
            "hit_at_5": round(self.hit_at_5, 4),
            "hit_at_10": round(self.hit_at_10, 4),
            "hit_at_30": round(self.hit_at_30, 4),
            "mrr": round(self.mrr, 4),
            "precision_at_5": round(self.precision_at_5, 4),
            "precision_at_10": round(self.precision_at_10, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "mean_score": round(self.mean_score, 4),
            "by_difficulty": self.by_difficulty,
        }


def aggregate(per_question: list[RetrievalScores], difficulties: dict[str, str] | None = None) -> AggregateScores:
    """Macro-average all per-question scores."""
    if not per_question:
        return AggregateScores()

    n = len(per_question)

    def avg(attr: str) -> float:
        return sum(getattr(s, attr) for s in per_question) / n

    # Stratify by difficulty if provided
    by_difficulty: dict[str, dict] = {}
    if difficulties:
        for level in ("easy", "medium", "hard"):
            subset = [s for s in per_question if difficulties.get(s.question_id) == level]
            if subset:
                m = len(subset)
                by_difficulty[level] = {
                    "n": m,
                    "hit_at_5": round(sum(s.hit_at_5 for s in subset) / m, 4),
                    "hit_at_10": round(sum(s.hit_at_10 for s in subset) / m, 4),
                    "mrr": round(sum(s.mrr for s in subset) / m, 4),
                }

    return AggregateScores(
        num_questions=n,
        hit_at_1=avg("hit_at_1"),
        hit_at_5=avg("hit_at_5"),
        hit_at_10=avg("hit_at_10"),
        hit_at_30=avg("hit_at_30"),
        mrr=avg("mrr"),
        precision_at_5=avg("precision_at_5"),
        precision_at_10=avg("precision_at_10"),
        ndcg_at_5=avg("ndcg_at_5"),
        ndcg_at_10=avg("ndcg_at_10"),
        mean_score=avg("mean_score"),
        by_difficulty=by_difficulty,
    )
