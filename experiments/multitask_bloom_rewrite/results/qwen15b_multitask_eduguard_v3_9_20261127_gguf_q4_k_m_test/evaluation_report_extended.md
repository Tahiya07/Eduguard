# Evaluation report: calibrated additions (test)

Generated 2026-11-28T01:12:45.220751+00:00. Original metrics are unchanged in `metrics.json` / `evaluation_report.md`; everything here is computed identically for the model outputs and for the human reference outputs.

## Bloom rewriting: model vs. reference under the same instruments

| Measure | Model | Human reference |
|---|---:|---:|
| Format check passed | 1.0 | 0.9459 |
| Classifier agrees with requested level | 0.8917 | 0.8034 |
| Topic preserved (bge-m3, threshold 0.578) | 0.963 | 0.9487 |
| Near-copy of source (ROUGE-L > 0.5806) | 0.1054 | 0.0655 |

- Agreement with the teacher-written rewrite for the same target level: ROUGE-L 0.3748 (cannot be raised by copying the source).
- BERTScore F1 (rescaled) to the teacher-written rewrite: 0.433
- Fully validated (calibrated): 0.8462 (classifier target match (conf >= 0.5) AND embedding topic preserved AND not trivial AND (format valid OR scenario-first Evaluate item flagged declarative)).

| Target level | N | Model accuracy | Reference accuracy (classifier ceiling) |
|---|---:|---:|---:|
| Remember | 64 | 0.7031 | 0.4531 |
| Understand | 61 | 0.8361 | 0.9508 |
| Apply | 49 | 1.0 | 0.8571 |
| Analyze | 53 | 1.0 | 0.8679 |
| Evaluate | 62 | 1.0 | 0.9032 |
| Create | 62 | 0.8548 | 0.8226 |

Where the reference accuracy is low (the classifier misreads the human rewrite), a low model accuracy is a measurement limit, not a model failure.

## Closed-book QA

- bge-m3 similarity to the reference answer: 0.827
- BERTScore F1 (roberta-large, rescaled) to the reference answer: 0.3717
- ROUGE-L with Markdown removed and the output cut to the reference length: 0.3075
- Median output 34 words vs reference 31.
- Factual correctness is not measured automatically; see `human_eval_qa.csv`.

## Summarization

- ROUGE-Lsum (summary-level): 0.3645
- bge-m3 similarity to the reference summary: 0.8197
- BERTScore F1 (roberta-large, rescaled; inputs over 512 tokens truncated) to the reference summary: 0.2581
- Median output 82 words vs reference 86.
- Faithfulness is not measured automatically; see `human_eval_summarization.csv`.

## Latency per task

| Task | N | Mean (s) | P50 (s) | P95 (s) |
|---|---:|---:|---:|---:|
| bloom_rewrite | 351 | 2.852 | 2.815 | 3.976 |
| qa | 147 | 1.732 | 1.644 | 2.489 |
| summarization | 147 | 15.454 | 13.553 | 34.29 |

## Human evaluation sheets

- `human_eval_qa.csv`: 61 items, stratified by qa_type, seed 42; model identity not shown.
- `human_eval_summarization.csv`: 41 items, stratified by source_type, seed 42; model identity not shown.

## Still required before claiming results

- Run the base model on the same split (`--condition base`) and report paired differences.
- Have at least one subject-matter rater complete the human sheets (two raters for agreement statistics).
- Report the test split once, for the final model only.
