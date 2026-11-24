import asyncio
import json
from pathlib import Path

import pytest

from evals.ragas_metrics import RagasMetricScorer
from evals.runner import EvaluationCase, RagResponse, load_dataset, run_evaluation


class FakeMetricScorer:
    async def score(self, *, question: str, answer: str, contexts: list[str], reference: str) -> dict[str, float]:
        assert question == "What is the answer?"
        assert answer == "It is 42."
        assert contexts == ["The answer is 42."]
        assert reference == "The answer is 42."
        return {
            "faithfulness": 1.0,
            "answer_relevance": 0.9,
            "context_precision": 0.8,
            "context_recall": 0.7,
        }


def test_ragas_metrics_receive_only_their_required_inputs() -> None:
    class CaptureMetric:
        def __init__(self) -> None:
            self.inputs = None

        async def ascore(self, **kwargs):
            self.inputs = kwargs
            return type("MetricResult", (), {"value": 1.0})()

    scorer = object.__new__(RagasMetricScorer)
    scorer.metrics = {
        name: CaptureMetric()
        for name in (
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
        )
    }

    scores = asyncio.run(
        scorer.score(
            question="Question",
            answer="Answer",
            contexts=["Context"],
            reference="Reference",
        )
    )

    assert scores == {name: 1.0 for name in scorer.metrics}
    assert set(scorer.metrics["faithfulness"].inputs) == {
        "user_input",
        "response",
        "retrieved_contexts",
    }
    assert set(scorer.metrics["answer_relevance"].inputs) == {
        "user_input",
        "response",
    }
    assert set(scorer.metrics["context_precision"].inputs) == {
        "user_input",
        "reference",
        "retrieved_contexts",
    }
    assert set(scorer.metrics["context_recall"].inputs) == {
        "user_input",
        "retrieved_contexts",
        "reference",
    }


def test_default_dataset_has_ten_complete_question_pairs() -> None:
    dataset_path = Path(__file__).with_name("export_support.yaml")

    service, prompt_version, cases = load_dataset(dataset_path)

    assert service == "export_support"
    assert prompt_version == "export-support-v1"
    assert len(cases) == 10
    assert all(case.question and case.reference for case in cases)


def test_standalone_runner_configures_django_app_import_path() -> None:
    import redbox.setting_enums

    assert redbox.setting_enums.__file__.endswith("setting_enums.py")


def test_dataset_requires_ten_to_twenty_questions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.yaml"
    dataset_path.write_text(
        "service: test\nprompt_version: v1\nquestions:\n"
        "  - id: one\n    question: a question\n    reference: an answer\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="between 10 and 20"):
        load_dataset(dataset_path)


def test_run_persists_requested_schema_and_writes_html_report(tmp_path: Path) -> None:
    async def rag_runner(question: str) -> RagResponse:
        assert question == "What is the answer?"
        return RagResponse(
            answer="It is 42.",
            contexts=["The answer is 42."],
            model="test-model",
        )

    case = EvaluationCase(
        id="case-1",
        question="What is the answer?",
        reference="The answer is 42.",
    )
    output_path = tmp_path / "results.jsonl"
    report_path = tmp_path / "report.html"

    records = asyncio.run(
        run_evaluation(
            cases=[case],
            service="test-service",
            prompt_version="v1",
            env="TEST",
            rag_runner=rag_runner,
            metric_scorer=FakeMetricScorer(),
            output_path=output_path,
            report_path=report_path,
        )
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(row) == {
        "timestamp",
        "env",
        "service",
        "model",
        "prompt_version",
        "question",
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
    }
    assert row["env"] == "TEST"
    assert row["model"] == "test-model"
    assert row["faithfulness"] == 1.0
    assert len(records) == 1
    assert report_path.exists()
    assert "Run average" in report_path.read_text(encoding="utf-8")


def test_empty_contexts_fail_instead_of_reporting_scores(tmp_path: Path) -> None:
    async def rag_runner(question: str) -> RagResponse:
        return RagResponse(answer="No sources.", contexts=[], model="test-model")

    case = EvaluationCase(id="case-1", question="Question", reference="Answer")

    with pytest.raises(ValueError, match="no usable contexts"):
        asyncio.run(
            run_evaluation(
                cases=[case],
                service="test-service",
                prompt_version="v1",
                env="TEST",
                rag_runner=rag_runner,
                metric_scorer=FakeMetricScorer(),
                output_path=tmp_path / "results.jsonl",
                report_path=tmp_path / "report.html",
            )
        )
