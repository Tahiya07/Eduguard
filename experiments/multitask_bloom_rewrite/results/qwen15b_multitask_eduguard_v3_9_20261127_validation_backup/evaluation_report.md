# Multi-task Evaluation Report

Generated (UTC): 2026-11-27T11:10:26.235289+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **633**
- Dataset manifest SHA-256: `10a9e969ff6f9e31a54cee956033e204a9926e260b1be5fe994811cb2b6b3040`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_9\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 341
- Target accuracy: 0.926686
- Macro-F1: 0.927232
- Format valid rate: 0.997067
- Trivial transform rate (lower is better): 0.0
- Topic preservation, near-copy rate, fully validated (calibrated), agreement with the reference rewrite and the classifier's accuracy on the reference rewrites: `evaluation_report_extended.md` (QA and summarization semantic similarity too).

| Pair type | N | Target accuracy | Format valid |
|---|---:|---:|---:|
| direct | 295 | 0.922034 | 0.99661 |
| pivot | 46 | 0.956522 | 1.0 |

## Closed-book academic QA (held-out validation)

- N: 146
- ROUGE-1 / ROUGE-2 / ROUGE-L: 0.378303 / 0.110625 / 0.265873

## Academic / lecture summarization (held-out validation)

- N: 146
- ROUGE-1: 0.394564
- ROUGE-2: 0.119967
- ROUGE-L: 0.245207
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| lecture_slide_text_style | 1 | 0.457912 | 0.108475 | 0.249158 |
| openstax_textbook_section | 116 | 0.368732 | 0.108643 | 0.222532 |
| original_academic_example | 29 | 0.495708 | 0.165659 | 0.335773 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 51 | 0.356275 | 0.104118 | 0.217218 |
| medium | 44 | 0.37896 | 0.115802 | 0.231518 |
| short | 29 | 0.495708 | 0.165659 | 0.335773 |
| very_long | 22 | 0.381208 | 0.104808 | 0.218089 |

## Output length

Hit cap = stopped by max_new_tokens without EOS. Loop stopped = output cut at a degenerate loop: a verbatim repeat of 24 generated tokens (cut to the last full sentence) or 4 consecutive sentences opening with the same 6 words (first kept); 0 = check off.

| Task | Cap | N | Hit cap | Loop stopped | Max tokens | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| bloom_rewrite | 160 | 341 | 0 | 2 | 125 | 27.98 |
| qa | 192 | 146 | 0 | 0 | 123 | 47.23 |
| summarization | 1104 | 146 | 1 | 6 | 1104 | 119.23 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 8.290201
- P95 latency (s): 20.683459

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_eduguard_v3_9_20261127\best_adapter`
