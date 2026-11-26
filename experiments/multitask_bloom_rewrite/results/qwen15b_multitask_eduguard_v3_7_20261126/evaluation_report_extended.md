# Evaluation report: calibrated additions (validation)

Generated 2026-11-26T20:36:38.104955+00:00. Original metrics are unchanged in `metrics.json` / `evaluation_report.md`; everything here is computed identically for the model outputs and for the human reference outputs.

## Bloom rewriting: model vs. reference under the same instruments

| Measure | Model | Human reference |
|---|---:|---:|
| Format check passed | 0.9971 | 0.956 |
| Classifier agrees with requested level | 0.8387 | 0.7977 |
| Topic preserved (bge-m3, threshold 0.5926) | 0.9032 | 0.9531 |
| Near-copy of source (ROUGE-L > 0.5946) | 0.0499 | 0.0381 |

- Agreement with the teacher-written rewrite for the same target level: ROUGE-L 0.3261 (cannot be raised by copying the source).
- Fully validated (calibrated): 0.7683 (classifier target match (conf >= 0.5) AND embedding topic preserved AND not trivial AND (format valid OR scenario-first Evaluate item flagged declarative)).

| Target level | N | Model accuracy | Reference accuracy (classifier ceiling) |
|---|---:|---:|---:|
| Remember | 61 | 0.2787 | 0.4262 |
| Understand | 59 | 0.9492 | 0.8644 |
| Apply | 52 | 0.9231 | 0.9423 |
| Analyze | 52 | 0.9808 | 0.7692 |
| Evaluate | 57 | 1.0 | 0.9825 |
| Create | 60 | 0.95 | 0.8333 |

Where the reference accuracy is low (the classifier misreads the human rewrite), a low model accuracy is a measurement limit, not a model failure.

## Closed-book QA

- bge-m3 similarity to the reference answer: 0.8217
- ROUGE-L with Markdown removed and the output cut to the reference length: 0.2808
- Median output 31.0 words vs reference 31.5.
- Factual correctness is not measured automatically; see `human_eval_qa.csv`.

## Summarization

- ROUGE-Lsum (summary-level): 0.3537
- bge-m3 similarity to the reference summary: 0.8321
- Median output 76.0 words vs reference 89.0.
- Faithfulness is not measured automatically; see `human_eval_summarization.csv`.

## Latency per task

| Task | N | Mean (s) | P50 (s) | P95 (s) |
|---|---:|---:|---:|---:|
| bloom_rewrite | 341 | 4.622 | 4.365 | 7.909 |
| qa | 146 | 6.251 | 5.95 | 8.86 |
| summarization | 146 | 18.525 | 13.979 | 39.825 |

## Human evaluation sheets

- `human_eval_qa.csv`: 62 items, stratified by qa_type, seed 42; model identity not shown.
- `human_eval_summarization.csv`: 41 items, stratified by source_type, seed 42; model identity not shown.

## Still required before claiming results

- Run the base model on the same split (`--condition base`) and report paired differences.
- Have at least one subject-matter rater complete the human sheets (two raters for agreement statistics).
- Report the test split once, for the final model only.
