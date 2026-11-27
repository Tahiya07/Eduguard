# Evaluation report: calibrated additions (validation)

Generated 2026-11-27T14:31:27.142088+00:00. Original metrics are unchanged in `metrics.json` / `evaluation_report.md`; everything here is computed identically for the model outputs and for the human reference outputs.

## Bloom rewriting: model vs. reference under the same instruments

| Measure | Model | Human reference |
|---|---:|---:|
| Format check passed | 0.9941 | 0.956 |
| Classifier agrees with requested level | 0.6158 | 0.7977 |
| Topic preserved (bge-m3, threshold 0.578) | 0.9941 | 0.9619 |
| Near-copy of source (ROUGE-L > 0.5806) | 0.3196 | 0.0528 |

- Agreement with the teacher-written rewrite for the same target level: ROUGE-L 0.295 (cannot be raised by copying the source).
- BERTScore F1 (rescaled) to the teacher-written rewrite: 0.3806
- Fully validated (calibrated): 0.6012 (classifier target match (conf >= 0.5) AND embedding topic preserved AND not trivial AND (format valid OR scenario-first Evaluate item flagged declarative)).

| Target level | N | Model accuracy | Reference accuracy (classifier ceiling) |
|---|---:|---:|---:|
| Remember | 61 | 0.0164 | 0.4426 |
| Understand | 59 | 0.7119 | 0.8475 |
| Apply | 52 | 0.7692 | 0.9231 |
| Analyze | 52 | 0.4038 | 0.7692 |
| Evaluate | 57 | 0.8246 | 0.9825 |
| Create | 60 | 0.9833 | 0.85 |

Where the reference accuracy is low (the classifier misreads the human rewrite), a low model accuracy is a measurement limit, not a model failure.

## Closed-book QA

- bge-m3 similarity to the reference answer: 0.803
- BERTScore F1 (roberta-large, rescaled) to the reference answer: 0.1277
- ROUGE-L with Markdown removed and the output cut to the reference length: 0.2156
- Median output 146.0 words vs reference 31.5.
- Factual correctness is not measured automatically; see `human_eval_qa.csv`.

## Summarization

- ROUGE-Lsum (summary-level): 0.2866
- bge-m3 similarity to the reference summary: 0.7974
- BERTScore F1 (roberta-large, rescaled; inputs over 512 tokens truncated) to the reference summary: -0.0013
- Median output 303.5 words vs reference 89.0.
- Faithfulness is not measured automatically; see `human_eval_summarization.csv`.

## Latency per task

| Task | N | Mean (s) | P50 (s) | P95 (s) |
|---|---:|---:|---:|---:|
| bloom_rewrite | 341 | 2.795 | 2.425 | 5.307 |
| qa | 146 | 14.305 | 13.746 | 20.946 |
| summarization | 146 | 35.268 | 34.124 | 52.198 |

## Human evaluation sheets

- `human_eval_qa.csv`: 62 items, stratified by qa_type, seed 42; model identity not shown.
- `human_eval_summarization.csv`: 41 items, stratified by source_type, seed 42; model identity not shown.

## Still required before claiming results

- Run the base model on the same split (`--condition base`) and report paired differences.
- Have at least one subject-matter rater complete the human sheets (two raters for agreement statistics).
- Report the test split once, for the final model only.
