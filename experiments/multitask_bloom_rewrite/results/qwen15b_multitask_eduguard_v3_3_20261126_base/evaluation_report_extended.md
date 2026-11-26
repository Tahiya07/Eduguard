# Evaluation report: calibrated additions (validation)

Generated 2026-11-26T11:51:00.090509+00:00. Original metrics are unchanged in `metrics.json` / `evaluation_report.md`; everything here is computed identically for the model outputs and for the human reference outputs.

## Bloom rewriting: model vs. reference under the same instruments

| Measure | Model | Human reference |
|---|---:|---:|
| Format check passed | 0.9966 | 0.9525 |
| Topic check passed (word overlap, original) | 0.8441 | 0.7085 |
| Cognitive check passed | 0.9966 | 0.9525 |
| Flagged declarative | 0.0034 | 0.0475 |
| Classifier agrees with requested level | 0.6102 | 0.7525 |
| Topic preserved (bge-m3, threshold 0.5836) | 0.9966 | 0.9458 |
| Near-copy of source (ROUGE-L > 0.5778) | 0.3288 | 0.0441 |

- Agreement with the teacher-written rewrite for the same target level: ROUGE-L 0.2899, bge-m3 similarity 0.7334. These cannot be raised by copying the source; the word-overlap topic check can, so it is not used to compare systems.
- Fully validated: original 0.4847 → calibrated 0.6 (classifier target match (conf >= 0.5) AND embedding topic preserved AND not trivial AND (format valid OR scenario-first Evaluate item flagged declarative)).
- 45 rewrites fail the word-overlap topic check but pass the embedding check.
- 1 'declarative' flags are scenario-first Evaluate items (a stated claim to judge), a valid item style in the dataset.

| Target level | N | Model accuracy | Reference accuracy (classifier ceiling) |
|---|---:|---:|---:|
| Remember | 59 | 0.0169 | 0.4407 |
| Understand | 49 | 0.6939 | 0.8776 |
| Apply | 42 | 0.7381 | 0.9286 |
| Analyze | 43 | 0.4651 | 0.4884 |
| Evaluate | 49 | 0.8571 | 0.9796 |
| Create | 53 | 0.9811 | 0.8491 |

Where the reference accuracy is low (the classifier misreads the human rewrite), a low model accuracy is a measurement limit, not a model failure.

## Closed-book QA: semantic agreement

- bge-m3 similarity to reference: mean 0.803, median 0.8119
- Key-point coverage (threshold 0.5033, calibrated on train references): mean 0.9199; all key points covered 0.7466; none covered 0.0 (n=146)
- Neither measure detects factual errors stated fluently; see `human_eval_qa.csv`.
- Length check: median output 146.0 words vs reference 31.5; 119 outputs use Markdown.
- Length-controlled (Markdown removed, cut to reference length): ROUGE-L 0.2156, ROUGE-2 0.0844. Recall (favours longer outputs): ROUGE-1 0.6362, ROUGE-2 0.1865.

## Summarization: semantic agreement

- bge-m3 similarity to reference: mean 0.7888, median 0.8127
  - arxiv_scientific_article: 0.7353 (n=33)
  - lecture_slide_text_style: 0.8621 (n=2)
  - original_academic_example: 0.8446 (n=29)
- ROUGE-Lsum (summary-level, comparable with arXiv literature): 0.3057
- Median length ratio (output/reference words): 3.324
- Outputs in lower-cased tokenized arXiv style: 0 of 64 (they mirror the arXiv references).
- Faithfulness is not measured automatically; see `human_eval_summarization.csv`.
- Length check: median output 282.5 words vs reference 73.5; 64 outputs use Markdown.
- Length-controlled (Markdown removed, cut to reference length): ROUGE-L 0.214, ROUGE-2 0.1025. Recall (favours longer outputs): ROUGE-1 0.6737, ROUGE-2 0.1942.

## Latency per task

| Task | N | Mean (s) | P50 (s) | P95 (s) |
|---|---:|---:|---:|---:|
| bloom_rewrite | 295 | 2.481 | 2.201 | 4.311 |
| qa | 146 | 12.56 | 12.671 | 12.796 |
| summarization | 64 | 29.288 | 29.641 | 36.839 |

## Human evaluation sheets

- `human_eval_qa.csv`: 62 items, stratified by qa_type, seed 42; model identity not shown.
- `human_eval_summarization.csv`: 40 items, stratified by source_type, seed 42; model identity not shown.

## Still required before claiming results

- Run the base model on the same split (`--condition base`) and report paired differences.
- Have at least one subject-matter rater complete the human sheets (two raters for agreement statistics).
- Report the test split once, for the final model only.
