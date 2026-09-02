"""
Report generation for RAG retrieval evaluation.

Writes a terminal table, a Markdown summary, an HTML report, and a structured
JSON file after each evaluation run. All outputs go to EVAL_REPORT_DIR (default:
tests/evaluation/reports/).
"""
from __future__ import annotations

import html as _html
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tests.evaluation.metrics.retrieval import AggregateScores, RetrievalScores, aggregate

REGRESSION_TOLERANCE = 0.05  # 5 pp drop vs baseline triggers a failure

# (display_name, agg_attribute, target_threshold, plain_english_description)
_METRIC_GLOSSARY = [
    ("Hit@1",        "hit_at_1",        0.50, "Was the single top result relevant? The strictest ranking test."),
    ("Hit@5",        "hit_at_5",        0.70, "Was a relevant chunk anywhere in the first 5 results? The primary pass/fail bar for easy questions."),
    ("Hit@10",       "hit_at_10",       0.80, "Was a relevant chunk in the first 10 results? The primary bar for hard questions."),
    ("Hit@30",       "hit_at_30",       0.90, "Was a relevant chunk anywhere in the top 30? A miss here means the content was never retrieved at all."),
    ("MRR",          "mrr",             0.50, "Mean Reciprocal Rank — 1 / rank, averaged over all questions. 1.0 = always first result. Higher is better."),
    ("Precision@5",  "precision_at_5",  0.30, "Of the first 5 results returned, what fraction were relevant? Measures list quality, not just presence."),
    ("Precision@10", "precision_at_10", 0.20, "Same as Precision@5 but across 10 results. Lower is expected — more results dilute precision."),
    ("NDCG@5",       "ndcg_at_5",       0.50, "Normalised Discounted Cumulative Gain at 5. Penalises relevant answers appearing lower in the list. 1.0 = perfect."),
    ("NDCG@10",      "ndcg_at_10",      0.60, "Best single number for overall ranking quality. Combines Hit@10 and position — a relevant answer at rank 3 scores higher than at rank 9."),
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _report_dir() -> Path:
    default = Path(__file__).parent.parent / "reports"
    return Path(os.environ.get("EVAL_REPORT_DIR", default))


def _score_class(value: float, threshold: float) -> str:
    if value >= threshold:
        return "good"
    if value >= threshold * 0.75:
        return "warn"
    return "poor"


def _rank_class(rank: int | None) -> str:
    if rank is None:
        return "poor"
    if rank <= 5:
        return "good"
    if rank <= 10:
        return "warn"
    return "poor"


@dataclass
class EvalReport:
    """Accumulates per-question scores during a test session and writes reports on close."""

    per_question: list[RetrievalScores] = field(default_factory=list)
    difficulties: dict[str, str] = field(default_factory=dict)
    questions: dict[str, str] = field(default_factory=dict)
    rag_params: dict = field(default_factory=dict)
    _started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def record(
        self,
        scores: RetrievalScores,
        difficulty: str = "unknown",
        question: str = "",
    ) -> None:
        self.per_question.append(scores)
        self.difficulties[scores.question_id] = difficulty
        if question:
            self.questions[scores.question_id] = question

    def aggregate(self) -> AggregateScores:
        return aggregate(self.per_question, self.difficulties)

    def write(self) -> Path:
        report_dir = _report_dir()
        report_dir.mkdir(parents=True, exist_ok=True)

        agg = self.aggregate()
        payload = self._build_payload(agg)
        meta = payload["meta"]

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        json_path = report_dir / f"eval_report_{timestamp}.json"

        json_path.write_text(json.dumps(payload, indent=2))
        (report_dir / "eval_report_latest.json").write_text(json.dumps(payload, indent=2))
        (report_dir / "eval_report_latest.md").write_text(self._build_markdown(agg, meta))
        (report_dir / "eval_report_latest.html").write_text(self._build_html(agg, meta))

        self._print_table(agg, meta)
        return json_path

    def _build_payload(self, agg: AggregateScores) -> dict:
        return {
            "meta": {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "started_at": self._started_at,
                "git_sha": _git_sha(),
                **self.rag_params,
            },
            "aggregate": agg.to_dict(),
            "per_question": [s.to_dict() for s in self.per_question],
        }

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _build_markdown(self, agg: AggregateScores, meta: dict) -> str:
        sha = meta.get("git_sha", "unknown")
        ts = meta.get("recorded_at", "")[:19].replace("T", " ")
        rag_k = meta.get("rag_k", "?")
        min_score = meta.get("min_score", "?")
        gauss = meta.get("rag_gauss_scale_size", "?")

        lines = [
            f"# RAG Retrieval Evaluation — {ts} UTC",
            "",
            f"**Git SHA**: `{sha}` | **rag_k**: {rag_k} | **min_score**: {min_score} | **gauss_scale**: {gauss}",
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Score | Target |",
            "|---|---|---|",
        ]
        for name, attr, target, _ in _METRIC_GLOSSARY:
            value = getattr(agg, attr)
            status = "OK" if value >= target else "LOW"
            lines.append(f"| {name} | {value:.3f} | {target:.2f} ({status}) |")
        lines += [
            f"| Questions | {agg.num_questions} | — |",
        ]

        if agg.by_difficulty:
            lines += ["", "## By Difficulty", "", "| Difficulty | N | Hit@5 | Hit@10 | MRR |", "|---|---|---|---|---|"]
            for level, d in agg.by_difficulty.items():
                lines.append(f"| {level} | {d['n']} | {d['hit_at_5']:.3f} | {d['hit_at_10']:.3f} | {d['mrr']:.3f} |")

        lines += [
            "",
            "## Per-Question Results",
            "",
            "| ID | Difficulty | Question | 1st Rank | Hit@5 | Hit@10 | MRR |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in sorted(self.per_question, key=lambda x: x.question_id):
            diff = self.difficulties.get(s.question_id, "?")
            rank_str = str(s.first_relevant_rank) if s.first_relevant_rank else "not found"
            q = self.questions.get(s.question_id, "")
            lines.append(
                f"| {s.question_id} | {diff} | {q} | {rank_str} "
                f"| {s.hit_at_5:.1f} | {s.hit_at_10:.1f} | {s.mrr:.2f} |"
            )

        lines += [
            "",
            "## What these metrics mean",
            "",
            "| Metric | Target | Plain English |",
            "|---|---|---|",
        ]
        for name, _attr, target, description in _METRIC_GLOSSARY:
            lines.append(f"| **{name}** | >{target:.2f} | {description} |")

        lines += [
            "",
            "---",
            "",
            "**Row colour guide (HTML report):**",
            "- Green: relevant chunk at rank 1–5",
            "- Amber: relevant chunk at rank 6–10",
            "- Red: relevant chunk at rank > 10 or not found",
        ]

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _build_html(self, agg: AggregateScores, meta: dict) -> str:
        sha = meta.get("git_sha", "unknown")
        ts = meta.get("recorded_at", "")[:19].replace("T", " ")
        rag_k = meta.get("rag_k", "?")
        gauss = meta.get("rag_gauss_scale_size", "?")
        gauss_decay = meta.get("rag_gauss_scale_decay", "?")
        n_candidates = meta.get("rag_num_candidates", "?")
        embedding = meta.get("embedding_model", "?")

        verdict = "PASS" if agg.hit_at_30 >= 0.9 else "NEEDS REVIEW"
        verdict_class = "pass" if verdict == "PASS" else "review"
        verdict_note = (
            "All questions findable in top-30."
            if agg.hit_at_30 >= 0.9
            else f"Hit@30 = {agg.hit_at_30:.2f} — some questions were not found in top-30 at all."
        )

        def card(label: str, value: float, threshold: float, note: str) -> str:
            cls = _score_class(value, threshold)
            return (
                f'<div class="card {cls}">'
                f'<div class="card-label">{label}</div>'
                f'<div class="card-value">{value:.3f}</div>'
                f'<div class="card-note">{note}</div>'
                f"</div>"
            )

        cards_html = (
            card("Hit@10",  agg.hit_at_10,  0.80, "Answer in first 10")
            + card("MRR",   agg.mrr,        0.50, "How early on average")
            + card("NDCG@10", agg.ndcg_at_10, 0.60, "Ranking quality")
            + card("Hit@30", agg.hit_at_30, 0.90, "Answer anywhere in top 30")
        )

        def agg_row(label: str, attr: str, target: float) -> str:
            value = getattr(agg, attr)
            cls = _score_class(value, target)
            return (
                f'<tr class="{cls}">'
                f"<td>{label}</td>"
                f'<td class="num">{value:.3f}</td>'
                f'<td class="target">&gt; {target:.2f}</td>'
                f"</tr>"
            )

        agg_rows = "".join(agg_row(name, attr, target) for name, attr, target, _ in _METRIC_GLOSSARY)
        agg_rows += f'<tr><td>Questions</td><td class="num">{agg.num_questions}</td><td class="target">—</td></tr>'

        glossary_rows = "".join(
            f"<tr>"
            f"<td><strong>{name}</strong></td>"
            f'<td class="target-col">&gt; {target:.2f}</td>'
            f"<td>{desc}</td>"
            f"</tr>"
            for name, _attr, target, desc in _METRIC_GLOSSARY
        )

        def pq_row(s: RetrievalScores) -> str:
            diff = self.difficulties.get(s.question_id, "?")
            rank = s.first_relevant_rank
            rank_str = str(rank) if rank else "not found"
            cls = _rank_class(rank)
            q_text = _html.escape(self.questions.get(s.question_id, ""))
            diff_badge = f'<span class="badge diff-{diff}">{diff}</span>'
            return (
                f'<tr class="{cls}">'
                f"<td>{_html.escape(s.question_id)}</td>"
                f"<td>{diff_badge}</td>"
                f'<td class="question-text">{q_text}</td>'
                f'<td class="num">{rank_str}</td>'
                f'<td class="num">{s.hit_at_5:.0f}</td>'
                f'<td class="num">{s.mrr:.2f}</td>'
                f'<td class="num">{s.ndcg_at_10:.2f}</td>'
                f"</tr>"
            )

        pq_rows = "".join(pq_row(s) for s in sorted(self.per_question, key=lambda x: x.question_id))

        diff_section = ""
        if agg.by_difficulty:
            diff_rows = "".join(
                f"<tr>"
                f'<td><span class="badge diff-{level}">{level}</span></td>'
                f"<td>{d['n']}</td>"
                f'<td class="num">{d["hit_at_5"]:.3f}</td>'
                f'<td class="num">{d["hit_at_10"]:.3f}</td>'
                f'<td class="num">{d["mrr"]:.3f}</td>'
                f"</tr>"
                for level, d in agg.by_difficulty.items()
            )
            diff_section = f"""
  <section>
    <h2>Results by Difficulty</h2>
    <table>
      <thead><tr><th>Difficulty</th><th>N</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th></tr></thead>
      <tbody>{diff_rows}</tbody>
    </table>
  </section>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Redbox Retrieval Eval — {_html.escape(ts)}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #f1f5f9; color: #1e293b; font-size: 14px; line-height: 1.5; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px; }}

    header {{ background: #1e293b; color: #f8fafc; border-radius: 8px;
              padding: 20px 24px; margin-bottom: 20px; }}
    header h1 {{ font-size: 20px; font-weight: 700; margin-bottom: 6px; }}
    .meta {{ font-size: 12px; color: #94a3b8; margin-bottom: 12px; }}
    .meta span {{ margin-right: 16px; }}
    .verdict {{ display: inline-block; padding: 4px 14px; border-radius: 20px;
                font-weight: 700; font-size: 13px; letter-spacing: .5px; }}
    .verdict.pass   {{ background: #22c55e; color: #fff; }}
    .verdict.review {{ background: #f59e0b; color: #fff; }}
    .verdict-note {{ font-size: 12px; color: #94a3b8; margin-left: 12px; }}

    .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
    .card {{ flex: 1 1 160px; background: #fff; border-radius: 8px; padding: 16px;
             border-left: 4px solid #cbd5e1; text-align: center; }}
    .card.good {{ border-left-color: #22c55e; }}
    .card.warn {{ border-left-color: #f59e0b; }}
    .card.poor {{ border-left-color: #ef4444; }}
    .card-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
                   color: #64748b; margin-bottom: 4px; }}
    .card-value {{ font-size: 28px; font-weight: 700; }}
    .card.good .card-value {{ color: #16a34a; }}
    .card.warn .card-value {{ color: #d97706; }}
    .card.poor .card-value {{ color: #dc2626; }}
    .card-note {{ font-size: 11px; color: #94a3b8; margin-top: 4px; }}

    section {{ background: #fff; border-radius: 8px; padding: 20px 24px; margin-bottom: 16px; }}
    h2 {{ font-size: 15px; font-weight: 700; margin-bottom: 14px; color: #0f172a; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid #e2e8f0;
          color: #475569; font-weight: 600; white-space: nowrap; }}
    td {{ padding: 7px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
    tr:last-child td {{ border-bottom: none; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .target {{ text-align: center; color: #94a3b8; font-size: 12px; }}
    .target-col {{ text-align: center; white-space: nowrap; color: #64748b; font-size: 12px; }}

    tr.good td {{ background: #f0fdf4; }}
    tr.warn td {{ background: #fffbeb; }}
    tr.poor td {{ background: #fef2f2; }}

    .question-text {{ max-width: 380px; color: #334155; font-size: 12px; }}

    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
              font-size: 11px; font-weight: 600; }}
    .diff-easy    {{ background: #dcfce7; color: #166534; }}
    .diff-medium  {{ background: #fef9c3; color: #713f12; }}
    .diff-hard    {{ background: #fee2e2; color: #991b1b; }}
    .diff-unknown {{ background: #f1f5f9; color: #475569; }}

    .legend {{ display: flex; gap: 16px; font-size: 12px; color: #64748b;
               margin-top: 12px; flex-wrap: wrap; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
            margin-right: 4px; vertical-align: middle; }}
    .dot.good {{ background: #22c55e; }}
    .dot.warn {{ background: #f59e0b; }}
    .dot.poor {{ background: #ef4444; }}

    .note {{ font-size: 11px; color: #94a3b8; margin-top: 8px; }}
  </style>
</head>
<body>
<div class="container">

  <header>
    <h1>Redbox RAG Retrieval Evaluation</h1>
    <div class="meta">
      <span>Run: {_html.escape(ts)} UTC</span>
      <span>Git: <code>{_html.escape(sha)}</code></span>
      <span>rag_k={rag_k}</span>
      <span>candidates={n_candidates}</span>
      <span>gauss_scale={gauss} / decay={gauss_decay}</span>
      <span>embedding: {_html.escape(str(embedding))}</span>
    </div>
    <span class="verdict {verdict_class}">{verdict}</span>
    <span class="verdict-note">{_html.escape(verdict_note)}</span>
  </header>

  <div class="cards">
    {cards_html}
  </div>

  <section>
    <h2>All Metrics</h2>
    <table>
      <thead><tr><th>Metric</th><th>Score</th><th>Target</th></tr></thead>
      <tbody>{agg_rows}</tbody>
    </table>
    <p class="note">
      Green = at or above target &nbsp;|&nbsp;
      Amber = within 25% below target &nbsp;|&nbsp;
      Red = below 75% of target
    </p>
  </section>

  <section>
    <h2>What these metrics mean</h2>
    <p style="font-size:13px;color:#475569;margin-bottom:12px;">
      All metrics are macro-averaged across {agg.num_questions} question(s).
      Relevance is determined by exact substring match against a gold snippet
      (case-insensitive). No LLM judge is used.
    </p>
    <table>
      <thead><tr><th>Metric</th><th>Target</th><th>Plain English</th></tr></thead>
      <tbody>{glossary_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Per-Question Results</h2>
    <div class="legend">
      <span><span class="dot good"></span>Rank 1–5 (top-5 pass)</span>
      <span><span class="dot warn"></span>Rank 6–10 (borderline)</span>
      <span><span class="dot poor"></span>Rank &gt;10 or not found</span>
    </div>
    <br>
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Difficulty</th><th>Question</th>
          <th>1st Rank</th><th>Hit@5</th><th>MRR</th><th>NDCG@10</th>
        </tr>
      </thead>
      <tbody>{pq_rows}</tbody>
    </table>
    <p class="note">
      "1st Rank" = position of the first relevant chunk in the returned list.
      Hit@5 is binary (1 or 0). Question text is populated when run via run_eval.py.
    </p>
  </section>
{diff_section}
</div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Terminal table
    # ------------------------------------------------------------------

    def _print_table(self, agg: AggregateScores, meta: dict) -> None:
        sha = meta.get("git_sha", "unknown")
        rag_k = meta.get("rag_k", "?")
        min_score = meta.get("min_score", "?")
        gauss = meta.get("rag_gauss_scale_size", "?")

        w = 70
        print("\n" + "=" * w)
        print("RAG Retrieval Evaluation Report")
        print(f"Git SHA: {sha}  |  rag_k={rag_k}  |  min_score={min_score}  |  gauss_scale={gauss}")
        print("=" * w)
        print(f"{'Metric':<22} {'Score':>8}  {'Target':>8}   {'Q':>4}")
        print("-" * w)
        for name, attr, target, _ in _METRIC_GLOSSARY:
            value = getattr(agg, attr)
            flag = "  " if value >= target else " !"
            n_col = f"{agg.num_questions:>4}" if name == "Hit@1" else "    "
            print(f"{name:<22} {value:>8.3f}  {target:>7.2f}{flag}{n_col}")
        print("=" * w)

        if agg.by_difficulty:
            print(f"\n{'Difficulty':<12} {'N':>4} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8}")
            print("-" * 44)
            for level, d in agg.by_difficulty.items():
                print(f"{level:<12} {d['n']:>4} {d['hit_at_5']:>8.3f} {d['hit_at_10']:>8.3f} {d['mrr']:>8.3f}")

        report_dir = _report_dir()
        print(f"\nReports saved to {report_dir}/")
        print(f"  eval_report_latest.json  — structured scores")
        print(f"  eval_report_latest.md    — markdown summary + glossary")
        print(f"  eval_report_latest.html  — stakeholder report (open in browser)")
        print()


def compare_to_baseline(
    agg: AggregateScores,
    baseline: dict,
    tolerance: float = REGRESSION_TOLERANCE,
) -> list[str]:
    """Return a list of regression messages (empty = no regressions)."""
    regressions = []
    baseline_agg = baseline.get("aggregate", {})

    for metric in ("hit_at_5", "hit_at_10", "hit_at_30", "mrr", "ndcg_at_5", "ndcg_at_10"):
        current = getattr(agg, metric)
        base = baseline_agg.get(metric)
        if base is not None and current < base - tolerance:
            regressions.append(
                f"{metric}: {current:.3f} < baseline {base:.3f} (drop {base - current:.3f} > tolerance {tolerance:.3f})"
            )

    return regressions
