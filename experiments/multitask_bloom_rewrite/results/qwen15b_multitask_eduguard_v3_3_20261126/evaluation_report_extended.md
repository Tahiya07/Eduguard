# Evaluation report: calibrated additions (validation)

Generated 2026-11-26T11:49:21.346563+00:00. Original metrics are unchanged in `metrics.json` / `evaluation_report.md`; everything here is computed identically for the model outputs and for the human reference outputs.

## Bloom rewriting: model vs. reference under the same instruments

| Measure | Model | Human reference |
|---|---:|---:|
| Format check passed | 0.8576 | 0.9525 |
| Topic check passed (word overlap, original) | 0.4847 | 0.7085 |
| Cognitive check passed | 0.8576 | 0.9525 |
| Flagged declarative | 0.1424 | 0.0475 |
| Classifier agrees with requested level | 0.8169 | 0.7525 |
| Topic preserved (bge-m3, threshold 0.5836) | 0.9254 | 0.9458 |
| Near-copy of source (ROUGE-L > 0.5778) | 0.0542 | 0.0441 |

- Agreement with the teacher-written rewrite for the same target level: ROUGE-L 0.3233, bge-m3 similarity 0.732. These cannot be raised by copying the source; the word-overlap topic check can, so it is not used to compare systems.
- Fully validated: original 0.3492 → calibrated 0.7627 (classifier target match (conf >= 0.5) AND embedding topic preserved AND not trivial AND (format valid OR scenario-first Evaluate item flagged declarative)).
- 131 rewrites fail the word-overlap topic check but pass the embedding check.
- 41 'declarative' flags are scenario-first Evaluate items (a stated claim to judge), a valid item style in the dataset.

| Target level | N | Model accuracy | Reference accuracy (classifier ceiling) |
|---|---:|---:|---:|
| Remember | 59 | 0.3051 | 0.4407 |
| Understand | 49 | 0.9796 | 0.8776 |
| Apply | 42 | 0.9286 | 0.9286 |
| Analyze | 43 | 0.8372 | 0.4884 |
| Evaluate | 49 | 0.9592 | 0.9796 |
| Create | 53 | 1.0 | 0.8491 |

Where the reference accuracy is low (the classifier misreads the human rewrite), a low model accuracy is a measurement limit, not a model failure.

## Closed-book QA: semantic agreement

- bge-m3 similarity to reference: mean 0.8177, median 0.8197
- Key-point coverage (threshold 0.5033, calibrated on train references): mean 0.8646; all key points covered 0.6644; none covered 0.0 (n=146)
- Neither measure detects factual errors stated fluently; see `human_eval_qa.csv`.
- Length check: median output 34.0 words vs reference 31.5; 0 outputs use Markdown.
- Length-controlled (Markdown removed, cut to reference length): ROUGE-L 0.279, ROUGE-2 0.1227. Recall (favours longer outputs): ROUGE-1 0.4016, ROUGE-2 0.1306.

## Summarization: semantic agreement

- bge-m3 similarity to reference: mean 0.8345, median 0.8537
  - arxiv_scientific_article: 0.7782 (n=33)
  - lecture_slide_text_style: 0.8532 (n=2)
  - original_academic_example: 0.8973 (n=29)
- ROUGE-Lsum (summary-level, comparable with arXiv literature): 0.4082
- Median length ratio (output/reference words): 0.946
- Outputs in lower-cased tokenized arXiv style: 33 of 64 (they mirror the arXiv references).
- Faithfulness is not measured automatically; see `human_eval_summarization.csv`.
- Length check: median output 87.5 words vs reference 73.5; 0 outputs use Markdown.
- Length-controlled (Markdown removed, cut to reference length): ROUGE-L 0.3032, ROUGE-2 0.1603. Recall (favours longer outputs): ROUGE-1 0.4628, ROUGE-2 0.1677.

## Latency per task

| Task | N | Mean (s) | P50 (s) | P95 (s) |
|---|---:|---:|---:|---:|
| bloom_rewrite | 295 | 4.474 | 4.111 | 8.014 |
| qa | 146 | 6.199 | 5.903 | 9.419 |
| summarization | 64 | 17.897 | 15.165 | 38.861 |

## Human evaluation sheets

- `human_eval_qa.csv`: 62 items, stratified by qa_type, seed 42; model identity not shown.
- `human_eval_summarization.csv`: 40 items, stratified by source_type, seed 42; model identity not shown.

## Still required before claiming results

- Run the base model on the same split (`--condition base`) and report paired differences.
- Have at least one subject-matter rater complete the human sheets (two raters for agreement statistics).
- Report the test split once, for the final model only.
