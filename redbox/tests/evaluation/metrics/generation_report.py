"""
Report rendering for the generation evaluation.
Produces Markdown and HTML alongside the existing retrieval reports.
"""

from __future__ import annotations

import html as _html

from tests.evaluation.metrics.generation import AggregateGenerationScores, GenerationScores

_THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevance": 0.70,
    "context_precision": 0.40,
    "context_recall": 0.60,
}

_GLOSSARY = [
    (
        "Faithfulness",
        "faithfulness",
        "Are claims in the answer supported by retrieved chunks? Measures hallucination risk.",
    ),
    (
        "Answer Relevance",
        "answer_relevance",
        "Does the answer address the question? Measures whether the LLM stayed on topic.",
    ),
    (
        "Context Precision",
        "context_precision",
        "Of the retrieved chunks, how many were actually needed? Measures retrieval noise.",
    ),
    (
        "Context Recall",
        "context_recall",
        "Does the retrieved context contain all information from the expected answer? Measures coverage.",
    ),
]


def _cls(value: float, threshold: float) -> str:
    if value >= threshold:
        return "good"
    if value >= threshold * 0.75:
        return "warn"
    return "poor"


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def build_generation_markdown(
    scores: list[GenerationScores],
    agg: AggregateGenerationScores,
    meta: dict,
) -> str:
    ts = meta.get("recorded_at", "")[:19].replace("T", " ")
    sha = meta.get("git_sha", "unknown")

    lines = [
        f"# RAG Generation Evaluation — {ts} UTC",
        "",
        f"**Git SHA**: `{sha}`  |  **Questions**: {agg.num_questions}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Score | Target | Status |",
        "|---|---|---|---|",
    ]
    for label, attr, _ in _GLOSSARY:
        value = getattr(agg, attr)
        target = _THRESHOLDS[attr]
        status = "✓ OK" if value >= target else "✗ LOW"
        lines.append(f"| {label} | {value:.3f} | > {target:.2f} | {status} |")

    lines += [
        "",
        "## Per-Question Results",
        "",
        "| ID | Condensed Query | Faith. | Rel. | Prec. | Rec. | Words | Citations |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(scores, key=lambda x: x.question_id):
        cq = (s.condensed_query[:60] + "…") if len(s.condensed_query) > 60 else s.condensed_query
        lines.append(
            f"| {s.question_id} | {cq} "
            f"| {s.faithfulness:.2f} | {s.answer_relevance:.2f} "
            f"| {s.context_precision:.2f} | {s.context_recall:.2f} "
            f"| {s.answer_length_words} | {s.citation_count} |"
        )

    lines += [
        "",
        "## What these metrics mean",
        "",
        "| Metric | Target | Plain English |",
        "|---|---|---|",
    ]
    for label, attr, description in _GLOSSARY:
        target = _THRESHOLDS[attr]
        lines.append(f"| **{label}** | > {target:.2f} | {description} |")

    lines += [
        "",
        "---",
        "",
        "_All scores produced by Claude (Bedrock) acting as an LLM judge. "
        "Judge calls use the same model as production answer generation._",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def build_generation_html(
    scores: list[GenerationScores],
    agg: AggregateGenerationScores,
    meta: dict,
) -> str:
    ts = meta.get("recorded_at", "")[:19].replace("T", " ")
    sha = meta.get("git_sha", "unknown")

    verdict = "PASS" if all(getattr(agg, attr) >= _THRESHOLDS[attr] for _, attr, _ in _GLOSSARY) else "NEEDS REVIEW"
    verdict_class = "pass" if verdict == "PASS" else "review"

    def card(label: str, attr: str) -> str:
        value = getattr(agg, attr)
        t = _THRESHOLDS[attr]
        cls = _cls(value, t)
        return (
            f'<div class="card {cls}">'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-value">{value:.3f}</div>'
            f'<div class="card-note">target &gt; {t:.2f}</div>'
            f"</div>"
        )

    cards_html = "".join(card(label, attr) for label, attr, _ in _GLOSSARY)

    glossary_rows = "".join(
        f"<tr><td><strong>{label}</strong></td>"
        f'<td class="target-col">&gt; {_THRESHOLDS[attr]:.2f}</td>'
        f"<td>{desc}</td></tr>"
        for label, attr, desc in _GLOSSARY
    )

    def pq_row(s: GenerationScores) -> str:
        worst = min(s.faithfulness, s.answer_relevance, s.context_recall)
        cls = "good" if worst >= 0.70 else ("warn" if worst >= 0.50 else "poor")
        cq = _html.escape((s.condensed_query[:80] + "…") if len(s.condensed_query) > 80 else s.condensed_query)
        answer_short = _html.escape((s.answer[:120] + "…") if len(s.answer) > 120 else s.answer)
        return (
            f'<tr class="{cls}">'
            f"<td>{_html.escape(s.question_id)}</td>"
            f'<td class="question-text">{cq}</td>'
            f'<td class="question-text">{answer_short}</td>'
            f'<td class="num">{s.faithfulness:.2f}</td>'
            f'<td class="num">{s.answer_relevance:.2f}</td>'
            f'<td class="num">{s.context_precision:.2f}</td>'
            f'<td class="num">{s.context_recall:.2f}</td>'
            f'<td class="num">{s.citation_count}</td>'
            f"</tr>"
        )

    pq_rows = "".join(pq_row(s) for s in sorted(scores, key=lambda x: x.question_id))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Redbox Generation Eval — {_html.escape(ts)}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #f1f5f9; color: #1e293b; font-size: 14px; line-height: 1.5; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}
    header {{ background: #1e293b; color: #f8fafc; border-radius: 8px; padding: 20px 24px; margin-bottom: 20px; }}
    header h1 {{ font-size: 20px; font-weight: 700; margin-bottom: 6px; }}
    .meta {{ font-size: 12px; color: #94a3b8; margin-bottom: 12px; }}
    .meta span {{ margin-right: 16px; }}
    .verdict {{ display: inline-block; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 13px; }}
    .verdict.pass   {{ background: #22c55e; color: #fff; }}
    .verdict.review {{ background: #f59e0b; color: #fff; }}
    .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
    .card {{ flex: 1 1 160px; background: #fff; border-radius: 8px; padding: 16px; border-left: 4px solid #cbd5e1; text-align: center; }}
    .card.good {{ border-left-color: #22c55e; }}
    .card.warn {{ border-left-color: #f59e0b; }}
    .card.poor {{ border-left-color: #ef4444; }}
    .card-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .5px; color: #64748b; margin-bottom: 4px; }}
    .card-value {{ font-size: 28px; font-weight: 700; }}
    .card.good .card-value {{ color: #16a34a; }}
    .card.warn .card-value {{ color: #d97706; }}
    .card.poor .card-value {{ color: #dc2626; }}
    .card-note {{ font-size: 11px; color: #94a3b8; margin-top: 4px; }}
    section {{ background: #fff; border-radius: 8px; padding: 20px 24px; margin-bottom: 16px; }}
    h2 {{ font-size: 15px; font-weight: 700; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid #e2e8f0; color: #475569; font-weight: 600; }}
    td {{ padding: 7px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .target-col {{ text-align: center; font-size: 12px; color: #64748b; }}
    tr.good td {{ background: #f0fdf4; }}
    tr.warn td {{ background: #fffbeb; }}
    tr.poor td {{ background: #fef2f2; }}
    .question-text {{ max-width: 280px; font-size: 12px; color: #334155; }}
    .note {{ font-size: 11px; color: #94a3b8; margin-top: 8px; }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>Redbox RAG Generation Evaluation</h1>
    <div class="meta">
      <span>Run: {_html.escape(ts)} UTC</span>
      <span>Git: <code>{_html.escape(sha)}</code></span>
      <span>{agg.num_questions} questions</span>
    </div>
    <span class="verdict {verdict_class}">{verdict}</span>
  </header>
  <div class="cards">{cards_html}</div>
  <section>
    <h2>What these metrics mean</h2>
    <p style="font-size:13px;color:#475569;margin-bottom:12px;">
      All scores are produced by Claude (Bedrock) acting as an LLM judge.
      Scores range 0–1; higher is better.
    </p>
    <table>
      <thead><tr><th>Metric</th><th>Target</th><th>Plain English</th></tr></thead>
      <tbody>{glossary_rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Per-Question Results</h2>
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Condensed Query</th><th>Answer (preview)</th>
          <th>Faith.</th><th>Rel.</th><th>Prec.</th><th>Rec.</th><th>Cite</th>
        </tr>
      </thead>
      <tbody>{pq_rows}</tbody>
    </table>
    <p class="note">
      Green = all core metrics ≥ 0.70 &nbsp;|&nbsp; Amber = weakest metric ≥ 0.50 &nbsp;|&nbsp; Red = below 0.50
    </p>
  </section>
</div>
</body>
</html>"""
