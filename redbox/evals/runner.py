from __future__ import annotations

import html
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import yaml

METRIC_NAMES = ("faithfulness", "answer_relevance", "context_precision", "context_recall")
MIN_QUESTIONS = 10
MAX_QUESTIONS = 20


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    question: str
    reference: str


@dataclass(frozen=True)
class RagResponse:
    answer: str
    contexts: list[str]
    model: str


@dataclass(frozen=True)
class EvaluationRecord:
    timestamp: str
    env: str
    service: str
    model: str
    prompt_version: str
    question: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float


class MetricScorer(Protocol):
    async def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str,
    ) -> Mapping[str, float]: ...


RagRunner = Callable[[str], Awaitable[RagResponse]]


def load_dataset(path: Path) -> tuple[str, str, list[EvaluationCase]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Evaluation dataset must be a YAML mapping: {path}")

    service = data.get("service")
    prompt_version = data.get("prompt_version")
    questions = data.get("questions")
    if not isinstance(service, str) or not service.strip():
        raise ValueError("Evaluation dataaset must define a non empty service")
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise ValueError("Evaluation dataset must define a non empty prompt version")
    if not isinstance(questions, list) or not MIN_QUESTIONS <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f"Evaluation dataset must contain between {MIN_QUESTIONS} and {MAX_QUESTIONS} questions")

    cases = []
    seen_ids = set()
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Question {index} must be a YAML mapping")
        case_id = question.get("id")
        query = question.get("question")
        reference = question.get("reference")
        if not all(isinstance(value, str) and value.strip() for value in (case_id, query, reference)):
            raise ValueError(f"question {index} must have a non empty id, question and reference fields")
        if case_id in seen_ids:
            raise ValueError(f"duplicate evaluation question id - {case_id}")
        seen_ids.add(case_id)
        cases.append(EvaluationCase(id=case_id, question=query, reference=reference))

    return service, prompt_version, cases


async def run_evaluation(
    *,
    cases: list[EvaluationCase],
    service: str,
    prompt_version: str,
    env: str,
    rag_runner: RagRunner,
    metric_scorer: MetricScorer,
    output_path: Path,
    report_path: Path,
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for case in cases:
        response = await rag_runner(case.question)
        if not response.answer.strip():
            raise ValueError(f"RAG returned an empty answer for evaluation question {case.id}.")
        if not response.contexts or any(not context.strip() for context in response.contexts):
            raise ValueError(f"RAG returned no usable contexts for evaluation question {case.id}.")

        scores = await metric_scorer.score(
            question=case.question,
            answer=response.answer,
            contexts=response.contexts,
            reference=case.reference,
        )
        if set(scores) != set(METRIC_NAMES):
            raise ValueError(f"Metric scorer must return exactly these metrics: {', '.join(METRIC_NAMES)}.")

        numeric_scores = {}
        for name in METRIC_NAMES:
            score = scores[name]
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
                raise ValueError(f"Metric {name} returned a non-finite numeric score for question {case.id}.")
            numeric_scores[name] = float(score)

        record = EvaluationRecord(
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            env=env,
            service=service,
            model=response.model,
            prompt_version=prompt_version,
            question=case.question,
            **numeric_scores,
        )
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        records.append(record)

    write_html_report(records, report_path)
    return records


def write_html_report(records: list[EvaluationRecord], path: Path) -> None:
    if not records:
        raise ValueError("Cannot write an evaluation report with no results.")

    path.parent.mkdir(parents=True, exist_ok=True)
    averages = {name: sum(getattr(record, name) for record in records) / len(records) for name in METRIC_NAMES}
    metric_headers = "".join(f"<th>{html.escape(name.replace('_', ' ').title())}</th>" for name in METRIC_NAMES)
    average_cells = "".join(f"<td>{averages[name]:.3f}</td>" for name in METRIC_NAMES)
    result_rows = []
    for record in records:
        cells = "".join(f"<td>{getattr(record, name):.3f}</td>" for name in METRIC_NAMES)
        result_rows.append(f"<tr><td>{html.escape(record.question)}</td>{cells}</tr>")

    first = records[0]
    document = f"""<!doctype html>
<html lang="yes">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAGAS evaluation — {html.escape(first.service)}</title>
  <style>
    body {{ font: 16px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccd2d8; padding: .6rem; text-align: left; }}
    th {{ background: #edf2f7; }}
    .averages {{ font-weight: 700; background: #f2f8f2; }}
  </style>
</head>
<body>
  <h1>RAGAS evaluation: {html.escape(first.service)}</h1>
  <p>Environment: {html.escape(first.env)} | Model: {html.escape(first.model)} |
     Prompt version: {html.escape(first.prompt_version)} | Questions: {len(records)}</p>
  <table>
    <thead><tr><th>Question</th>{metric_headers}</tr></thead>
    <tbody>
      <tr class="averages"><td>Run average</td>{average_cells}</tr>
      {"".join(result_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
