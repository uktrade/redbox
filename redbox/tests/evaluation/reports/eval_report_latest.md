# RAG Retrieval Evaluation — 2026-09-01 15:55:13 UTC

**Git SHA**: `ebedfe0d` | **rag_k**: 30 | **min_score**: 0.6 | **gauss_scale**: 3

## Aggregate Metrics

| Metric | Score | Target |
|---|---|---|
| Hit@1 | 0.500 | 0.50 (OK) |
| Hit@5 | 0.700 | 0.70 (OK) |
| Hit@10 | 0.700 | 0.80 (LOW) |
| Hit@30 | 1.000 | 0.90 (OK) |
| MRR | 0.576 | 0.50 (OK) |
| Precision@5 | 0.180 | 0.30 (LOW) |
| Precision@10 | 0.110 | 0.20 (LOW) |
| NDCG@5 | 0.418 | 0.50 (LOW) |
| NDCG@10 | 0.442 | 0.60 (LOW) |
| Questions | 10 | — |

## By Difficulty

| Difficulty | N | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| easy | 4 | 0.500 | 0.500 | 0.542 |
| medium | 3 | 0.667 | 0.667 | 0.465 |
| hard | 3 | 1.000 | 1.000 | 0.733 |

## Per-Question Results

| ID | Difficulty | Question | 1st Rank | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|---|
| cptpp_001 | easy |  | 12 | 0.0 | 0.0 | 0.08 |
| cptpp_002 | easy |  | 1 | 1.0 | 1.0 | 1.00 |
| cptpp_003 | easy |  | 1 | 1.0 | 1.0 | 1.00 |
| cptpp_004 | easy |  | 12 | 0.0 | 0.0 | 0.08 |
| cptpp_005 | medium |  | 16 | 0.0 | 0.0 | 0.06 |
| cptpp_006 | medium |  | 1 | 1.0 | 1.0 | 1.00 |
| cptpp_007 | medium |  | 3 | 1.0 | 1.0 | 0.33 |
| cptpp_008 | hard |  | 5 | 1.0 | 1.0 | 0.20 |
| cptpp_009 | hard |  | 1 | 1.0 | 1.0 | 1.00 |
| cptpp_010 | hard |  | 1 | 1.0 | 1.0 | 1.00 |

## What these metrics mean

| Metric | Target | Plain English |
|---|---|---|
| **Hit@1** | >0.50 | Was the single top result relevant? The strictest ranking test. |
| **Hit@5** | >0.70 | Was a relevant chunk anywhere in the first 5 results? The primary pass/fail bar for easy questions. |
| **Hit@10** | >0.80 | Was a relevant chunk in the first 10 results? The primary bar for hard questions. |
| **Hit@30** | >0.90 | Was a relevant chunk anywhere in the top 30? A miss here means the content was never retrieved at all. |
| **MRR** | >0.50 | Mean Reciprocal Rank — 1 / rank, averaged over all questions. 1.0 = always first result. Higher is better. |
| **Precision@5** | >0.30 | Of the first 5 results returned, what fraction were relevant? Measures list quality, not just presence. |
| **Precision@10** | >0.20 | Same as Precision@5 but across 10 results. Lower is expected — more results dilute precision. |
| **NDCG@5** | >0.50 | Normalised Discounted Cumulative Gain at 5. Penalises relevant answers appearing lower in the list. 1.0 = perfect. |
| **NDCG@10** | >0.60 | Best single number for overall ranking quality. Combines Hit@10 and position — a relevant answer at rank 3 scores higher than at rank 9. |

---

**Row colour guide (HTML report):**
- Green: relevant chunk at rank 1–5
- Amber: relevant chunk at rank 6–10
- Red: relevant chunk at rank > 10 or not found
