# Evaluation report: calibrated additions (validation)

Generated 2026-11-27T08:00:02.906968+00:00. Original metrics are unchanged in `metrics.json` / `evaluation_report.md`; everything here is computed identically for the model outputs and for the human reference outputs.

## Bloom rewriting: model vs. reference under the same instruments

| Measure | Model | Human reference |
|---|---:|---:|
| Format check passed | 0.9971 | 0.956 |
| Classifier agrees with requested level | 0.8299 | 0.8006 |
| Topic preserved (bge-m3, threshold 0.5913) | 0.9091 | 0.956 |
| Near-copy of source (ROUGE-L > 0.5926) | 0.0381 | 0.0381 |

- Agreement with the teacher-written rewrite for the same target level: ROUGE-L 0.3185 (cannot be raised by copying the source).
- BERTScore F1 (rescaled) to the teacher-written rewrite: 0.395
- Fully validated (calibrated): 0.7507 (classifier target match (conf >= 0.5) AND embedding topic preserved AND not trivial AND (format valid OR scenario-first Evaluate item flagged declarative)).

| Target level | N | Model accuracy | Reference accuracy (classifier ceiling) |
|---|---:|---:|---:|
| Remember | 61 | 0.2623 | 0.4426 |
| Understand | 59 | 0.9492 | 0.8475 |
| Apply | 52 | 0.9038 | 0.9231 |
| Analyze | 52 | 0.9423 | 0.7692 |
| Evaluate | 57 | 0.9825 | 0.9825 |
| Create | 60 | 0.9833 | 0.8667 |

Where the reference accuracy is low (the classifier misreads the human rewrite), a low model accuracy is a measurement limit, not a model failure.

## Closed-book QA

- bge-m3 similarity to the reference answer: 0.8238
- BERTScore F1 (roberta-large, rescaled) to the reference answer: 0.3452
- ROUGE-L with Markdown removed and the output cut to the reference length: 0.2761
- Median output 35.0 words vs reference 31.5.
- Factual correctness is not measured automatically; see `human_eval_qa.csv`.

## Summarization

- ROUGE-Lsum (summary-level): 0.3476
- bge-m3 similarity to the reference summary: 0.8334
- BERTScore F1 (roberta-large, rescaled; inputs over 512 tokens truncated) to the reference summary: 0.2545
- Median output 82.5 words vs reference 89.0.
- Faithfulness is not measured automatically; see `human_eval_summarization.csv`.

## Latency per task

| Task | N | Mean (s) | P50 (s) | P95 (s) |
|---|---:|---:|---:|---:|
| bloom_rewrite | 341 | 4.692 | 4.282 | 8.452 |
| qa | 146 | 7.003 | 6.624 | 10.793 |
| summarization | 146 | 19.595 | 15.903 | 31.664 |

## Human evaluation sheets

- `human_eval_qa.csv`: 62 items, stratified by qa_type, seed 42; model identity not shown.
- `human_eval_summarization.csv`: 41 items, stratified by source_type, seed 42; model identity not shown.

## Still required before claiming results

- Run the base model on the same split (`--condition base`) and report paired differences.
- Have at least one subject-matter rater complete the human sheets (two raters for agreement statistics).
- Report the test split once, for the final model only.
