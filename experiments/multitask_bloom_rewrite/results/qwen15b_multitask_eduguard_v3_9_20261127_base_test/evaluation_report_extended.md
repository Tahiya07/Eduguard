# Evaluation report: calibrated additions (test)

Generated 2026-11-27T19:57:43.261762+00:00. Original metrics are unchanged in `metrics.json` / `evaluation_report.md`; everything here is computed identically for the model outputs and for the human reference outputs.

## Bloom rewriting: model vs. reference under the same instruments

| Measure | Model | Human reference |
|---|---:|---:|
| Format check passed | 0.9943 | 0.9459 |
| Classifier agrees with requested level | 0.6439 | 0.8034 |
| Topic preserved (bge-m3, threshold 0.578) | 0.9943 | 0.9487 |
| Near-copy of source (ROUGE-L > 0.5806) | 0.3675 | 0.0655 |

- Agreement with the teacher-written rewrite for the same target level: ROUGE-L 0.3018 (cannot be raised by copying the source).
- BERTScore F1 (rescaled) to the teacher-written rewrite: 0.3806
- Fully validated (calibrated): 0.6182 (classifier target match (conf >= 0.5) AND embedding topic preserved AND not trivial AND (format valid OR scenario-first Evaluate item flagged declarative)).

| Target level | N | Model accuracy | Reference accuracy (classifier ceiling) |
|---|---:|---:|---:|
| Remember | 64 | 0.0312 | 0.4531 |
| Understand | 61 | 0.7213 | 0.9508 |
| Apply | 49 | 0.6939 | 0.8571 |
| Analyze | 53 | 0.5472 | 0.8679 |
| Evaluate | 62 | 0.9032 | 0.9032 |
| Create | 62 | 0.9839 | 0.8226 |

Where the reference accuracy is low (the classifier misreads the human rewrite), a low model accuracy is a measurement limit, not a model failure.

## Closed-book QA

- bge-m3 similarity to the reference answer: 0.7992
- BERTScore F1 (roberta-large, rescaled) to the reference answer: 0.126
- ROUGE-L with Markdown removed and the output cut to the reference length: 0.2069
- Median output 148 words vs reference 31.
- Factual correctness is not measured automatically; see `human_eval_qa.csv`.

## Summarization

- ROUGE-Lsum (summary-level): 0.2815
- bge-m3 similarity to the reference summary: 0.7883
- BERTScore F1 (roberta-large, rescaled; inputs over 512 tokens truncated) to the reference summary: -0.0091
- Median output 281 words vs reference 86.
- Faithfulness is not measured automatically; see `human_eval_summarization.csv`.

## Latency per task

| Task | N | Mean (s) | P50 (s) | P95 (s) |
|---|---:|---:|---:|---:|
| bloom_rewrite | 351 | 2.834 | 2.474 | 5.494 |
| qa | 147 | 14.27 | 13.671 | 18.512 |
| summarization | 147 | 32.549 | 30.779 | 48.914 |

## Human evaluation sheets

- `human_eval_qa.csv`: 61 items, stratified by qa_type, seed 42; model identity not shown.
- `human_eval_summarization.csv`: 41 items, stratified by source_type, seed 42; model identity not shown.

## Still required before claiming results

- Run the base model on the same split (`--condition base`) and report paired differences.
- Have at least one subject-matter rater complete the human sheets (two raters for agreement statistics).
- Report the test split once, for the final model only.
