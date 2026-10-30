"""
LLM-as-judge generation quality metrics for RAG evaluation.

Four metrics:

  faithfulness       — Are the claims in the answer supported by retrieved chunks?
  answer_relevance   — Does the answer address the question?
  context_precision  — Of the retrieved chunks, which ones were actually necessary?
  context_recall     — Do the retrieved chunks cover everything in the expected answer?

All judge calls go through a single `llm` argument (a LangChain BaseChatModel),
which should be the same Bedrock Claude instance used in production.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = (
    "You are a strict fact-checking evaluator. "
    "Your task is to determine whether each factual claim in an AI-generated answer "
    "is explicitly supported by the provided source documents. "
    "Do not use any outside knowledge. Be precise and conservative — a claim is only "
    "supported if the evidence is clearly present in the documents."
)

_FAITHFULNESS_HUMAN = """\
Source documents:
{chunks}

AI-generated answer:
{answer}

Instructions:
1. List every distinct factual claim made in the answer.
2. For each claim, decide: SUPPORTED or UNSUPPORTED based solely on the source documents.
3. Return your assessment as JSON with this exact structure:
{{"supported_claims": <integer>, "total_claims": <integer>, "unsupported": [<list of unsupported claim strings>]}}

Return ONLY the JSON object. No preamble."""

# ----

_ANSWER_RELEVANCE_SYSTEM = (
    "You are evaluating whether an AI answer addresses the user's question. "
    "Focus only on relevance — not accuracy or completeness."
)

_ANSWER_RELEVANCE_HUMAN = """\
Question: {question}

Answer: {answer}

Rate how well the answer addresses the question on a scale of 0 to 5:
  0 = Does not address the question at all
  1 = Barely related
  2 = Partially addresses the question
  3 = Mostly addresses the question
  4 = Fully addresses with minor gaps
  5 = Fully and comprehensively addresses the question

Return only the integer score (0–5). No explanation."""

# ----

_CONTEXT_PRECISION_SYSTEM = (
    "You are evaluating whether retrieved document chunks were useful for generating an answer. "
    "Be strict: a chunk is only 'necessary' if removing it would make the answer noticeably "
    "less accurate or complete."
)

_CONTEXT_PRECISION_HUMAN = """\
Question: {question}

Generated answer: {answer}

Retrieved chunks (numbered in retrieval order):
{numbered_chunks}

For each chunk, decide: was it NECESSARY to generate the answer above?
Return your assessment as JSON with this exact structure:
{{"assessments": [{{"chunk_num": <int>, "necessary": <true|false>}}]}}

Return ONLY the JSON object. No preamble."""

# ----

_CONTEXT_RECALL_SYSTEM = (
    "You are evaluating whether retrieved documents contain enough information to answer a question. "
    "Compare the retrieved chunks against the expected (ground-truth) answer."
)

_CONTEXT_RECALL_HUMAN = """\
Question: {question}

Expected answer (ground truth):
{expected_answer}

Retrieved document chunks:
{chunks}

Instructions:
1. List every distinct factual statement from the expected answer.
2. For each statement, decide: is this information PRESENT in the retrieved chunks?
3. Return your assessment as JSON:
{{"covered_statements": <integer>, "total_statements": <integer>, "missing": [<list of missing statement strings>]}}

Return ONLY the JSON object. No preamble."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_chunks(docs: list[Document], numbered: bool = False) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        uri = doc.metadata.get("uri", doc.metadata.get("source", "unknown"))
        prefix = f"[{i}] " if numbered else ""
        parts.append(f"{prefix}Source: {uri}\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def _parse_json(text: str) -> dict:
    """Extract and parse the first JSON object from an LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in LLM response: {text!r}")


def _call_llm(llm: Any, system: str, human: str) -> str:
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return response.content.strip()


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def evaluate_faithfulness(
    llm: Any,
    answer: str,
    retrieved_docs: list[Document],
) -> tuple[float, dict]:
    """
    Score: supported_claims / total_claims.
    Returns (score, raw_dict).
    Returns 1.0 if the answer is empty or has no claims (not a failure).
    """
    if not answer.strip():
        return 1.0, {"supported_claims": 0, "total_claims": 0, "unsupported": []}
    chunks_text = _format_chunks(retrieved_docs)
    prompt = _FAITHFULNESS_HUMAN.format(chunks=chunks_text, answer=answer)
    try:
        raw = _parse_json(_call_llm(llm, _FAITHFULNESS_SYSTEM, prompt))
        total = max(raw.get("total_claims", 0), 1)
        supported = min(raw.get("supported_claims", 0), total)
        return round(supported / total, 4), raw
    except Exception as exc:
        return 0.0, {"error": str(exc)}


def evaluate_answer_relevance(
    llm: Any,
    question: str,
    answer: str,
) -> tuple[float, dict]:
    """
    Score: LLM rating (0–5) / 5.
    Returns (score, raw_dict).
    """
    if not answer.strip():
        return 0.0, {"rating": 0}
    prompt = _ANSWER_RELEVANCE_HUMAN.format(question=question, answer=answer)
    try:
        text = _call_llm(llm, _ANSWER_RELEVANCE_SYSTEM, prompt)
        rating = int(re.search(r"\d", text).group())
        rating = max(0, min(5, rating))
        return round(rating / 5, 4), {"rating": rating}
    except Exception as exc:
        return 0.0, {"error": str(exc)}


def evaluate_context_precision(
    llm: Any,
    question: str,
    answer: str,
    retrieved_docs: list[Document],
) -> tuple[float, dict]:
    """
    NDCG-weighted precision: necessary chunks at higher ranks score more.
    Score in [0, 1].
    """
    if not retrieved_docs:
        return 0.0, {"assessments": []}
    numbered = _format_chunks(retrieved_docs, numbered=True)
    prompt = _CONTEXT_PRECISION_HUMAN.format(question=question, answer=answer, numbered_chunks=numbered)
    try:
        raw = _parse_json(_call_llm(llm, _CONTEXT_PRECISION_SYSTEM, prompt))
        assessments = raw.get("assessments", [])
        if not assessments:
            return 0.0, raw

        dcg = sum((1.0 / math.log2(i + 2)) for i, a in enumerate(assessments) if a.get("necessary", False))
        # Ideal DCG: all necessary chunks at the top
        num_necessary = sum(1 for a in assessments if a.get("necessary", False))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(num_necessary))
        score = round(dcg / idcg, 4) if idcg > 0 else 0.0
        return score, raw
    except Exception as exc:
        return 0.0, {"error": str(exc)}


def evaluate_context_recall(
    llm: Any,
    question: str,
    expected_answer: str,
    retrieved_docs: list[Document],
) -> tuple[float, dict]:
    """
    Score: covered_statements / total_statements.
    Returns (score, raw_dict).
    """
    if not expected_answer.strip() or not retrieved_docs:
        return 0.0, {"covered_statements": 0, "total_statements": 0, "missing": []}
    chunks_text = _format_chunks(retrieved_docs)
    prompt = _CONTEXT_RECALL_HUMAN.format(question=question, expected_answer=expected_answer, chunks=chunks_text)
    try:
        raw = _parse_json(_call_llm(llm, _CONTEXT_RECALL_SYSTEM, prompt))
        total = max(raw.get("total_statements", 0), 1)
        covered = min(raw.get("covered_statements", 0), total)
        return round(covered / total, 4), raw
    except Exception as exc:
        return 0.0, {"error": str(exc)}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GenerationScores:
    """All generation quality scores for a single question."""

    question_id: str
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Pipeline artifacts
    condensed_query: str = ""
    answer: str = ""
    citation_count: int = 0
    answer_length_words: int = 0
    num_retrieved: int = 0

    # Raw judge outputs for debugging
    faithfulness_raw: dict = field(default_factory=dict)
    answer_relevance_raw: dict = field(default_factory=dict)
    context_precision_raw: dict = field(default_factory=dict)
    context_recall_raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "condensed_query": self.condensed_query,
            "answer_length_words": self.answer_length_words,
            "citation_count": self.citation_count,
            "num_retrieved": self.num_retrieved,
        }


@dataclass
class AggregateGenerationScores:
    """Macro-averaged generation scores."""

    num_questions: int = 0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    def to_dict(self) -> dict:
        return {
            "num_questions": self.num_questions,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
        }


def aggregate_generation(scores: list[GenerationScores]) -> AggregateGenerationScores:
    if not scores:
        return AggregateGenerationScores()
    n = len(scores)
    return AggregateGenerationScores(
        num_questions=n,
        faithfulness=sum(s.faithfulness for s in scores) / n,
        answer_relevance=sum(s.answer_relevance for s in scores) / n,
        context_precision=sum(s.context_precision for s in scores) / n,
        context_recall=sum(s.context_recall for s in scores) / n,
    )
