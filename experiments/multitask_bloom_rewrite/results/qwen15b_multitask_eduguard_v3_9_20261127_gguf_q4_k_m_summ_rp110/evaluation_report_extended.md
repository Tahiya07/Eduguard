# Evaluation report: calibrated additions (validation)

Generated 2026-11-28T06:39:04.279619+00:00. Original metrics are unchanged in `metrics.json` / `evaluation_report.md`; everything here is computed identically for the model outputs and for the human reference outputs.

## Summarization

- ROUGE-Lsum (summary-level): 0.3705
- bge-m3 similarity to the reference summary: 0.8391
- BERTScore F1 (roberta-large, rescaled; inputs over 512 tokens truncated) to the reference summary: 0.2531
- Median output 105.0 words vs reference 89.0.
- Faithfulness is not measured automatically; see `human_eval_summarization.csv`.

## Latency per task

| Task | N | Mean (s) | P50 (s) | P95 (s) |
|---|---:|---:|---:|---:|
| summarization | 146 | 14.321 | 12.262 | 30.466 |

## Human evaluation sheets

- `human_eval_summarization.csv`: 41 items, stratified by source_type, seed 42; model identity not shown.

## Still required before claiming results

- Run the base model on the same split (`--condition base`) and report paired differences.
- Have at least one subject-matter rater complete the human sheets (two raters for agreement statistics).
- Report the test split once, for the final model only.
