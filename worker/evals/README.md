# RAGAS evaluation

This runner accompanies `tests/evaluation/`, which measures retrieval ranking only. It invokes the live Redbox graph for each curated question, evaluates its answer and the source contexts with RAGAS, appends one JSONL record per question, and writes an HTML report for the latest run.

## Setup and run

From the repository root, install the evaluation dependency group and configure Redbox's normal AWS and OpenSearch settings:

```bash
cd worker
poetry install --with evaluation
poetry run python -m evals.run_ragas
```

Or, after dependencies are installed:

```bash
make eval-ragas
```

Use `poetry run python -m evals.run_ragas --help` to see dataset, output, report, and ingestion options.

The runner uses the existing evaluation pipeline and corpus in `tests/evaluation/dataset/corpus/`. Place one or more evaluation PDFs there (the supplied question set currently targets the CPTPP Impact Assessment), and provide the normal OpenSearch and AWS access required for ingestion, embeddings, answer generation, and metric scoring. Evaluation model calls incur normal provider costs.

The default run ingests the corpus into the dedicated evaluation index and removes that index and its uploaded S3 objects afterward. `--skip-ingest` reuses an existing evaluation index, while `--no-cleanup` leaves resources created by the run in place.

The default question set is `export_support.yaml`. It contains ten reviewed question/reference-answer pairs aligned with the existing CPTPP evaluation set. Keep the set between 10 and 20 questions, and increment `prompt_version` whenever the evaluated prompt configuration changes. Each question must have a unique `id`, `question`, and gold `reference`.

By default, results are appended to `evals/results/ragas_results.jsonl` and the latest run's report is written to `evals/results/ragas_report_latest.html`. Both paths can be overridden with `--output` and `--report`. The dataset can be changed with `--dataset`. The generated results directory is git-ignored.

## Record schema

Each JSONL line contains exactly these fields:

| Field | Meaning |
|---|---|
| `timestamp` | UTC time at which this question was evaluated, in ISO 8601 format |
| `env` | Redbox environment from `ENVIRONMENT` (defaults to `LOCAL`) |
| `service` | Service identifier in the YAML dataset |
| `model` | Model recorded by Redbox for the response, or its configured chat model if call metadata is unavailable |
| `prompt_version` | Version declared in the YAML dataset |
| `question` | Evaluated user question |
| `faithfulness` | RAGAS faithfulness score |
| `answer_relevance` | RAGAS answer relevancy score |
| `context_precision` | RAGAS context precision score |
| `context_recall` | RAGAS context recall score |

Answers and contexts are passed to RAGAS at evaluation time but are not persisted in the export schema. The curated question and gold reference live in the version-controlled YAML. The JSONL output is suitable for later data workspace, dashboard, or Datadog ingestion (this repository does not send records to those systems.)

## Metric definitions and interpretation

All metrics are evaluated per question using RAGAS's current collections-based API. Faithfulness, context precision, and context recall are generally in the range 0–1, where higher is better. Answer relevance is based on cosine similarity and can fall outside 0–1. Higher scores indicate a response that addresses the question more directly.

| Metric | What it measures | Interpretation |
|---|---|---|
| **Context precision** | Whether relevant retrieved contexts are ranked ahead of irrelevant contexts, relative to the gold reference answer | Low values suggest noisy or poorly ordered retrieval |
| **Context recall** | Whether claims needed to support the gold reference answer can be found in retrieved contexts | Low values suggest missing relevant information |
| **Faithfulness** | The proportion of answer claims supported by retrieved contexts | Low values indicate unsupported claims or hallucination risk |
| **Answer relevance** | How directly the generated answer addresses the original question | Low values indicate incomplete, indirect, or off-topic answers - it does not measure factual correctness |

In RAGAS, context precision is average precision over the ranked contexts: relevant chunks appearing earlier contribute more. Context recall is the number of claims in the gold reference answer attributable to retrieved contexts divided by the total claims in that reference. Faithfulness is the number of generated-answer claims supported by retrieved contexts divided by the total generated-answer claims. Answer relevance reverse-generates likely questions from the answer and averages their embedding similarity to the user's question.

RAGAS metrics use an LLM and, for answer relevance, an embedding model. Scores are estimates rather than deterministic ground truth. Compare trends using the same evaluator configuration, question set, and prompt version. Context recall uses each question's gold reference answer as a proxy for reference contexts.

## Example interpretation
A representative evaluation run produced the following averages:

Metric	| Score
Faithfulness |	0.989
Answer Relevance |	0.880
Context Precision |	0.222
Context Recall |	1.000

These results indicate:

- Excellent retrieval coverage (Context Recall 1.000), with all information required to answer the evaluation questions retrieved successfully.
- Very high answer grounding (Faithfulness 0.989), suggesting a low risk of hallucinated content.
- Strong answer quality (Answer Relevance 0.880), with answers generally addressing the user's questions directly.
- Lower retrieval efficiency (Context Precision 0.222), indicating that retrieval returns additional context beyond that strictly required to answer the question.

This pattern suggests a retrieval strategy optimised for completeness (high recall) rather than selectivity (high precision). While answer quality remains high, future optimisation may focus on improving retrieval precision through chunking, ranking, or filtering changes.

## Reporting and follow-on integration

The generated HTML report presents per-question scores and run averages. The JSONL file is the stable record-level export for future reporting integrations. Datadog and data-workspace ingestion are intentionally not implemented here.
