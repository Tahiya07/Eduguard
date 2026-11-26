# Multi-task Evaluation Report

Generated (UTC): 2026-11-26T20:36:36.480094+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **633**
- Dataset manifest SHA-256: `08eef49752b2f6ef3fbb2a968f47fe7947d22bfe8a0ed573b9c22f8367db3516`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_7\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 341
- Target accuracy: 0.83871
- Macro-F1: 0.828071
- Format valid rate: 0.997067
- Trivial transform rate (lower is better): 0.0
- Topic preservation, near-copy rate, fully validated (calibrated), agreement with the reference rewrite and the classifier's accuracy on the reference rewrites: `evaluation_report_extended.md` (QA and summarization semantic similarity too).

| Pair type | N | Target accuracy | Format valid |
|---|---:|---:|---:|
| direct | 295 | 0.823729 | 0.99661 |
| pivot | 46 | 0.934783 | 1.0 |

## Closed-book academic QA (held-out validation)

- N: 146
- ROUGE-1 / ROUGE-2 / ROUGE-L: 0.389251 / 0.124799 / 0.282418

## Academic / lecture summarization (held-out validation)

- N: 146
- ROUGE-1: 0.388773
- ROUGE-2: 0.122802
- ROUGE-L: 0.247302
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| lecture_slide_text_style | 1 | 0.531469 | 0.161972 | 0.251748 |
| openstax_textbook_section | 116 | 0.363987 | 0.110843 | 0.224384 |
| original_academic_example | 29 | 0.482999 | 0.16929 | 0.33882 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 51 | 0.365491 | 0.108447 | 0.223061 |
| medium | 44 | 0.359186 | 0.114958 | 0.229234 |
| short | 29 | 0.482999 | 0.16929 | 0.33882 |
| very_long | 22 | 0.377711 | 0.110489 | 0.218995 |

## Output length (generation stopped by max_new_tokens without EOS)

| Task | Cap | N | Hit cap | Max tokens | Mean tokens |
|---|---:|---:|---:|---:|---:|
| bloom_rewrite | 160 | 341 | 0 | 93 | 30.86 |
| qa | 192 | 146 | 0 | 110 | 42.23 |
| summarization | 1104 | 146 | 2 | 1104 | 125.39 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 8.204544
- P95 latency (s): 20.401592

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_eduguard_v3_7_20261126\best_adapter`
