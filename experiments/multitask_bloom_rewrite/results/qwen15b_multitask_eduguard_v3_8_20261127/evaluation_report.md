# Multi-task Evaluation Report

Generated (UTC): 2026-11-27T05:36:54.207317+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **633**
- Dataset manifest SHA-256: `66daa2ca3a02202d6aba1f8c7fc8f66cbb2111902983e61697cc7874ac4ddec8`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_8\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 341
- Target accuracy: 0.829912
- Macro-F1: 0.816476
- Format valid rate: 0.997067
- Trivial transform rate (lower is better): 0.0
- Topic preservation, near-copy rate, fully validated (calibrated), agreement with the reference rewrite and the classifier's accuracy on the reference rewrites: `evaluation_report_extended.md` (QA and summarization semantic similarity too).

| Pair type | N | Target accuracy | Format valid |
|---|---:|---:|---:|
| direct | 295 | 0.820339 | 0.99661 |
| pivot | 46 | 0.891304 | 1.0 |

## Closed-book academic QA (held-out validation)

- N: 146
- ROUGE-1 / ROUGE-2 / ROUGE-L: 0.380259 / 0.117843 / 0.272796

## Academic / lecture summarization (held-out validation)

- N: 146
- ROUGE-1: 0.384485
- ROUGE-2: 0.113661
- ROUGE-L: 0.237083
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| lecture_slide_text_style | 1 | 0.494624 | 0.137184 | 0.27957 |
| openstax_textbook_section | 116 | 0.362137 | 0.10513 | 0.217342 |
| original_academic_example | 29 | 0.47008 | 0.146976 | 0.314582 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 51 | 0.351539 | 0.099729 | 0.212043 |
| medium | 44 | 0.365641 | 0.110572 | 0.226749 |
| short | 29 | 0.47008 | 0.146976 | 0.314582 |
| very_long | 22 | 0.385715 | 0.108221 | 0.213644 |

## Output length

Hit cap = stopped by max_new_tokens without EOS. Loop stopped = a verbatim repeat of 24 generated tokens was cut (0 = guard off).

| Task | Cap | N | Hit cap | Loop stopped | Max tokens | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| bloom_rewrite | 160 | 341 | 0 | 0 | 86 | 30.88 |
| qa | 192 | 146 | 0 | 0 | 97 | 46.62 |
| summarization | 1104 | 146 | 1 | 3 | 1104 | 130.62 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 8.662378
- P95 latency (s): 22.037797

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_eduguard_v3_8_20261127\best_adapter`
