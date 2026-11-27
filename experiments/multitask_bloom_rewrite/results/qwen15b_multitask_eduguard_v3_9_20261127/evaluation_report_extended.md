# Evaluation report: calibrated additions (validation)

Generated 2026-11-27T11:10:28.357079+00:00. Original metrics are unchanged in `metrics.json` / `evaluation_report.md`; everything here is computed identically for the model outputs and for the human reference outputs.

## Bloom rewriting: model vs. reference under the same instruments

| Measure | Model | Human reference |
|---|---:|---:|
| Format check passed | 0.9971 | 0.956 |
| Classifier agrees with requested level | 0.9267 | 0.7977 |
| Topic preserved (bge-m3, threshold 0.578) | 0.8768 | 0.9619 |
| Near-copy of source (ROUGE-L > 0.5806) | 0.0323 | 0.0528 |

- Agreement with the teacher-written rewrite for the same target level: ROUGE-L 0.3062 (cannot be raised by copying the source).
- BERTScore F1 (rescaled) to the teacher-written rewrite: 0.3773
- Fully validated (calibrated): 0.8035 (classifier target match (conf >= 0.5) AND embedding topic preserved AND not trivial AND (format valid OR scenario-first Evaluate item flagged declarative)).

| Target level | N | Model accuracy | Reference accuracy (classifier ceiling) |
|---|---:|---:|---:|
| Remember | 61 | 0.8197 | 0.4426 |
| Understand | 59 | 0.9831 | 0.8475 |
| Apply | 52 | 0.8846 | 0.9231 |
| Analyze | 52 | 0.9808 | 0.7692 |
| Evaluate | 57 | 1.0 | 0.9825 |
| Create | 60 | 0.9 | 0.85 |

Where the reference accuracy is low (the classifier misreads the human rewrite), a low model accuracy is a measurement limit, not a model failure.

## Closed-book QA

- bge-m3 similarity to the reference answer: 0.8202
- BERTScore F1 (roberta-large, rescaled) to the reference answer: 0.3434
- ROUGE-L with Markdown removed and the output cut to the reference length: 0.2742
- Median output 35.5 words vs reference 31.5.
- Factual correctness is not measured automatically; see `human_eval_qa.csv`.

## Summarization

- ROUGE-Lsum (summary-level): 0.3571
- bge-m3 similarity to the reference summary: 0.8351
- BERTScore F1 (roberta-large, rescaled; inputs over 512 tokens truncated) to the reference summary: 0.2676
- Median output 77.5 words vs reference 89.0.
- Faithfulness is not measured automatically; see `human_eval_summarization.csv`.

## Latency per task

| Task | N | Mean (s) | P50 (s) | P95 (s) |
|---|---:|---:|---:|---:|
| bloom_rewrite | 341 | 4.408 | 4.176 | 7.982 |
| qa | 146 | 7.2 | 7.037 | 10.72 |
| summarization | 146 | 18.449 | 15.563 | 34.419 |

## Human evaluation sheets

- `human_eval_qa.csv`: 62 items, stratified by qa_type, seed 42; model identity not shown.
- `human_eval_summarization.csv`: 41 items, stratified by source_type, seed 42; model identity not shown.

## Still required before claiming results

- Run the base model on the same split (`--condition base`) and report paired differences.
- Have at least one subject-matter rater complete the human sheets (two raters for agreement statistics).
- Report the test split once, for the final model only.
