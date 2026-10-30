Generation Evaluation:

The retrieval eval answers one question: *did the retriever return the right chunks?* The generation eval extends this to the full Classic Search RAG path. It condenses the question, retrieves chunks, generates an answer, and then has an LLM judge score the answer on four metrics. It reuses the retrieval eval's corpus, OpenSearch eval index, dataset and ingestion fixtures.

---

## 2. How the generation eval works

```
question ─► 1. condense ─► 2. retrieve per condensed query ─► 3. generate answer ─► 4. LLM judge ×4
            CONDENSE_*     ParameterisedElasticsearchRetriever   RETRIEVAL_* prompts   faithfulness, relevance,
            prompts        (Gaussian boost on), merge + dedup                          context precision, recall
```

| Metric | Score | Target |
|---|---|---|
| Faithfulness: are the answer's claims supported by the retrieved chunks? | supported / total claims | ≥ 0.70 |
| Answer relevance: does the answer address the question? | judge rating 0–5, divided by 5 | ≥ 0.70 |
| Context precision: are the useful chunks ranked near the top? | NDCG over chunks the judge marks as necessary | ≥ 0.40 |
| Context recall: do the chunks cover the `expected_answer`? | covered / total statements | ≥ 0.60 |

If the judge's reply can't be parsed, or the judge call fails, the metric scores 0.0 and the error is kept in `*_raw["error"]`.

**Tests** (`test_generation_eval.py`, marker `ai_generation`, 14 tests):

| Test | Fails when |
|---|---|
| `test_generation_corpus_ingested` | indexed document count ≠ corpus PDF count |
| `test_generation_query_condensation` | never. It compares Hit@10 for the raw and condensed queries and writes `reports/condensation_ablation.txt`. |
| `test_generation_all_questions` | any answer is empty. It also fills the `generation_report` accumulator. |
| `test_generation_no_regression` | any metric drops by more than 5 pp against `baselines/generation_baseline.json`. **Skipped** while the baseline is empty. |
| `test_generation_faithfulness[cptpp_001…010]` | faithfulness < 0.70 for that question |

When the session ends, the `generation_report` fixture writes `reports/generation_report_<ts>.json` and `generation_report_latest.{json,md,html}`.

The `ai` and `ai_generation` markers don't overlap: `make eval-retrieval` (`-m ai`) does not run the generation tests.

---

## 3. Running it

Prerequisites are the same as for the retrieval eval: AWS credentials (S3, Bedrock, Textract), OpenSearch on `localhost:9200`, and a bucket (`make eval_backend BUCKET_NAME=…`).

```bash
make eval-generation ## full suite. On the first run, no_regression is SKIPPED.
```
Review `redbox/tests/evaluation/reports/generation_report_latest.html`, then:

```bash
make eval-update-generation-baseline ## copies generation_report_latest.json into baselines/
```
Commit `baselines/generation_baseline.json` and re-run. no_regression should now PASS.

---
