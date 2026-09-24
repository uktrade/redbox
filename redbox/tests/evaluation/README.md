# Assit RAG Retrieval Evaluation Harness

A production-emulator test suite that measures the quality of Assist's retrieval pipeline end-to-end. Every test ingests real PDFs through the full production stack (Textract → chunking → Bedrock embeddings → OpenSearch) and retrieves against a curated set of known questions and answers.

---

## Prerequisites checklist

Before running any eval command, ensure the application is running. The application has been tested using the debugger locally. It is not guaranteed it will work when run in containers. 

Skip to step 4 if the application is already running locally with OpenSearch instance and AWS Creds all available. If not, check item 1 to 3.

### 1. OpenSearch running

```bash
docker compose up -d --wait opensearch
```

Default: `localhost:9200`, user `admin`, password `Opensearch2024^`. Override via environment variables (see below).

### 2. AWS credentials

The eval ingests PDFs to S3, calls Textract for extraction, and calls Bedrock for embeddings. You need credentials with access to all three services.

```bash
# Option A — AWS profile
export AWS_PROFILE=your-profile

# Option B — explicit keys (e.g. from an SSO session)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...   # required for SSO/assumed-role sessions
```

The Bedrock embedding model used is `us.amazon.nova-pro-v1` in region `us-east-1` by default. Ensure your credentials have `bedrock:InvokeModel` permission in that region.

### 3. Environment variables

Create a `.env` file in the repo root (or `redbox/.env`) with at minimum:

```env
BUCKET_NAME=your-s3-bucket-name
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_PASSWORD=Opensearch2024^
AWS_REGION=eu-west-2
```

The Makefile targets automatically inject `ENVIRONMENT=LOCAL` and `ENABLE_METADATA_EXTRACTION=true` — you do not need to set these manually.

### 4. Corpus PDF

Place the CPTPP Impact Assessment PDF at:

```
redbox/tests/evaluation/dataset/corpus/cptpp_impact_assessment.pdf
```

The eval will fail with a clear error if this directory is empty. Ideally, multiple pdfs can be placed in this directory.

### 5. Poetry dependencies installed

```bash
# From the repo root
poetry install
```

---

## Quick start

```bash
# From the repo root:

# 1. Start OpenSearch (if not already running)
docker compose up -d --wait opensearch

# 2. Run the full eval suite
make eval-retrieval

# 3. Open the stakeholder report in your browser
open redbox/tests/evaluation/reports/eval_report_latest.html
```

The first run takes 2–5 minutes (ingestion + embeddings). Subsequent runs with `--skip-ingest` take ~30 seconds.

---

## Command reference

| Command | What it does |
|---|---|
| `make eval-retrieval` | Run the full eval suite via pytest. Ingests corpus, runs all questions, writes reports, cleans up. |
| `make eval-update-baseline` | Promote the latest report to `baselines/baseline.json`. Run this after a verified improvement. |
| `make eval-ingest-corpus` | Ingest corpus PDFs into the inspect index for manual browsing. Does not run tests. |
| `make eval-query-index KEYWORD="trade"` | Query the inspect index for a keyword. Useful when authoring new Q&A pairs. |
| `make eval-query-index URI="eval-corpus/file.pdf"` | List all chunks from a specific document. |
| `make eval-generate-qa PDF=tests/evaluation/dataset/corpus/file.pdf` | Generate candidate Q&A pairs from a PDF using the LLM. Review and copy approved entries into `retrieval_eval_set.json`. |

To run the eval directly without Make:

```bash
cd redbox
poetry run python tests/evaluation/run_eval.py           # full run
poetry run python tests/evaluation/run_eval.py --skip-ingest  # skip ingestion
poetry run python tests/evaluation/run_eval.py --no-cleanup   # keep index after run
```

---

## What each test does

| Test | What it checks | Fails when |
|---|---|---|
| `test_retrieval_corpus_ingested` | Every PDF in `corpus/` was successfully extracted and indexed | A PDF produced no chunks |
| `test_retrieval_all_questions` | Every question in the dataset finds a relevant chunk somewhere in the top-30 results | Any question gets zero hits in top-30 — a complete miss |
| `test_retrieval_no_regression` | Key metrics have not dropped more than 5 percentage points vs the committed baseline | A metric drops more than the tolerance. *Skipped on the first run (no baseline yet).* |
| `test_retrieval_per_question[id]` | Each individual question finds a relevant chunk within the top-5 (easy/medium) or top-10 (hard) | The relevant chunk is ranked too low |
| `test_retrieval_ablation_gaussian` | Compares retrieval with and without the Gaussian re-ranking pass | Never fails — prints a comparison table to the terminal for review |

---

## Reading the results

After a run, three report files are written to `redbox/tests/evaluation/reports/`:

| File | Purpose |
|---|---|
| `eval_report_latest.html` | **Start here.** Stakeholder-friendly with colour coding, explanations, per-question breakdown. Open in a browser. |
| `eval_report_latest.md` | Markdown version — good for attaching to a PR or copying into Confluence. Includes a metric glossary. |
| `eval_report_latest.json` | Machine-readable scores. Used by the regression gate and baseline tooling. |
| `ingest_latest.log` | Full step-by-step ingestion log. Check this if a PDF fails to extract or chunk. |

### Metric glossary

| Metric | What it means | Target |
|---|---|---|
| **Hit@1** | Was the top result relevant? | > 0.50 |
| **Hit@5** | Was a relevant chunk in the first 5 results? | > 0.70 |
| **Hit@10** | Was a relevant chunk in the first 10 results? | > 0.80 |
| **Hit@30** | Was a relevant chunk anywhere in the top 30? | > 0.90 |
| **MRR** | Mean Reciprocal Rank — 1/rank averaged over all questions. 1.0 = always first. | > 0.50 |
| **Precision@5** | Of the first 5 results, what fraction were relevant? | > 0.30 |
| **NDCG@10** | Ranking quality — penalises relevant answers appearing lower down. Best single summary number. | > 0.60 |

**What to look at first:**

1. `Hit@30` — if this is below 1.0, the retriever is completely missing some questions. Investigate those IDs first; this is a bug.
2. `Hit@10` — the primary signal for whether retrieval quality is good enough for the RAG pipeline.
3. `MRR` / `NDCG@10` — ranking quality. Low values here mean the right content is being found but not ranked near the top.

**Colour coding in the HTML report:**
- Green row: relevant chunk at rank 1–5
- Amber row: relevant chunk at rank 6–10
- Red row: relevant chunk at rank > 10, or not found at all

---

## Updating the baseline

The baseline captures the "expected" scores for a given code and config state. The regression test (`test_retrieval_no_regression`) checks that future runs do not drop more than 5 percentage points below it.

**When to update:** After a code change that genuinely improves retrieval (not just a test change), and after you have reviewed the HTML report and are satisfied.

```bash
# 1. Run the eval and review the HTML report
make eval-retrieval
open redbox/tests/evaluation/reports/eval_report_latest.html

# 2. Promote to baseline
make eval-update-baseline

# 3. Commit
git add redbox/tests/evaluation/baselines/baseline.json
git commit -m "chore(eval): update retrieval baseline after <describe change>"
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `No PDFs found in tests/evaluation/dataset/corpus/` | Corpus PDF is missing | Copy `cptpp_impact_assessment.pdf` to that directory |
| `KeyError: 'ENVIRONMENT'` or `ImproperlyConfigured: Set the ENVIRONMENT` | `ENVIRONMENT` env var not set | Add `ENVIRONMENT=LOCAL` to your `.env`, or use `make eval-*` (it sets this automatically) |
| `pydantic ValidationError: enable_metadata_extraction` | `ENABLE_METADATA_EXTRACTION` env var not set | Add `ENABLE_METADATA_EXTRACTION=true` to your `.env`, or use `make eval-*` |
| `ModuleNotFoundError: No module named 'tests.evaluation'` | `PYTHONPATH` does not include the redbox package root | Use `make eval-*` or set `PYTHONPATH=$(PWD)/django_app:$(PWD)/redbox` |
| `django.core.exceptions.ImproperlyConfigured` pointing to `redbox_app.settings` | `pytest-dotenv` loaded `tests/.env.test` before the eval bootstrap | Expected — `run_eval.py` force-overrides `DJANGO_SETTINGS_MODULE`. Ensure `run_eval.py` is imported *before* any other `redbox.*` import. |
| Ablation table shows identical With/Without Gaussian columns | P0 bug: `metadata.file_name.keyword` in `queries.py:345` | Change to `metadata.uri.keyword` — see commit history for the fix |

---

## Adding new Q&A pairs

1. Check `retrieval_eval_set.json` and input questions following the format. Alternatively, use `make eval-generate-qa PDF=path/to/file.pdf` to generate candidates (TBD).
2. Review the output at `/tmp/candidate_qa.json`. Remove any pairs where the snippet is too generic or appears in many chunks.
3. Copy approved entries into `dataset/retrieval_eval_set.json`.
4. Assign a unique `id` (e.g. `cptpp_011`) and a `difficulty` (`easy`, `medium`, or `hard`).
5. Add the source PDF to `dataset/corpus/` if it is not already there.
6. Run `make eval-retrieval` and update the baseline if scores improve.
