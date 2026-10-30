# RAG Generation Evaluation — 2026-10-30 11:20:01 UTC

**Git SHA**: `5c4d8da9`  |  **Questions**: 10

## Aggregate Metrics

| Metric | Score | Target | Status |
|---|---|---|---|
| Faithfulness | 1.000 | > 0.70 | ✓ OK |
| Answer Relevance | 0.800 | > 0.70 | ✓ OK |
| Context Precision | 0.474 | > 0.40 | ✓ OK |
| Context Recall | 0.950 | > 0.60 | ✓ OK |

## Per-Question Results

| ID | Condensed Query | Faith. | Rel. | Prec. | Rec. | Words | Citations |
|---|---|---|---|---|---|---|---|
| cptpp_001 | What is the estimated long-run annual increase in UK GDP fro… | 1.00 | 0.00 | 0.77 | 1.00 | 0 | 0 |
| cptpp_002 | How many countries are currently members of the Comprehensiv… | 1.00 | 1.00 | 0.39 | 1.00 | 49 | 1 |
| cptpp_003 | What percentage of UK goods exports will be tariff-free unde… | 1.00 | 1.00 | 0.34 | 1.00 | 104 | 2 |
| cptpp_004 | What is the total combined GDP of all member countries in th… | 1.00 | 0.00 | 0.58 | 1.00 | 0 | 0 |
| cptpp_005 | How much is bilateral trade between the UK and CPTPP members… | 1.00 | 1.00 | 0.28 | 1.00 | 62 | 3 |
| cptpp_006 | What is the estimated impact of the Comprehensive and Progre… | 1.00 | 1.00 | 0.42 | 1.00 | 75 | 5 |
| cptpp_007 | What is the estimated annual reduction in tariff duties on U… | 1.00 | 1.00 | 0.76 | 1.00 | 109 | 1 |
| cptpp_008 | What is the projected increase in global greenhouse gas emis… | 1.00 | 1.00 | 0.46 | 1.00 | 135 | 2 |
| cptpp_009 | What tariff does Malaysia currently charge on UK whisky expo… | 1.00 | 1.00 | 0.41 | 1.00 | 78 | 2 |
| cptpp_010 | Which UK regions will see the largest percentage GDP increas… | 1.00 | 1.00 | 0.33 | 0.50 | 104 | 0 |

## What these metrics mean

| Metric | Target | Plain English |
|---|---|---|
| **Faithfulness** | > 0.70 | Are claims in the answer supported by retrieved chunks? Measures hallucination risk. |
| **Answer Relevance** | > 0.70 | Does the answer address the question? Measures whether the LLM stayed on topic. |
| **Context Precision** | > 0.40 | Of the retrieved chunks, how many were actually needed? Measures retrieval noise. |
| **Context Recall** | > 0.60 | Does the retrieved context contain all information from the expected answer? Measures coverage. |

---

_All scores produced by Claude (Bedrock) acting as an LLM judge. Judge calls use the same model as production answer generation._
