# Multi-task Evaluation Report

Generated (UTC): 2026-11-27T22:39:55.970528+00:00
Condition: **gguf**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **633**
- Dataset manifest SHA-256: `10a9e969ff6f9e31a54cee956033e204a9926e260b1be5fe994811cb2b6b3040`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_9\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 341
- Target accuracy: 0.903226
- Macro-F1: 0.90521
- Format valid rate: 1.0
- Trivial transform rate (lower is better): 0.002933
- Topic preservation, near-copy rate, fully validated (calibrated), agreement with the reference rewrite and the classifier's accuracy on the reference rewrites: `evaluation_report_extended.md` (QA and summarization semantic similarity too).

| Pair type | N | Target accuracy | Format valid |
|---|---:|---:|---:|
| direct | 295 | 0.891525 | 1.0 |
| pivot | 46 | 0.978261 | 1.0 |

## Closed-book academic QA (held-out validation)

- N: 146
- ROUGE-1 / ROUGE-2 / ROUGE-L: 0.407977 / 0.139828 / 0.301269

## Academic / lecture summarization (held-out validation)

- N: 146
- ROUGE-1: 0.413474
- ROUGE-2: 0.136166
- ROUGE-L: 0.263278
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| lecture_slide_text_style | 1 | 0.619048 | 0.410959 | 0.510204 |
| openstax_textbook_section | 116 | 0.381663 | 0.119224 | 0.232339 |
| original_academic_example | 29 | 0.53363 | 0.194458 | 0.378518 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 51 | 0.382902 | 0.127773 | 0.232927 |
| medium | 44 | 0.393262 | 0.126594 | 0.252415 |
| short | 29 | 0.53363 | 0.194458 | 0.378518 |
| very_long | 22 | 0.366381 | 0.097923 | 0.203455 |

## Output length

Hit cap = stopped by max_new_tokens without EOS. Loop stopped = output cut at a degenerate loop: a verbatim repeat of 24 generated tokens (cut to the last full sentence) or 4 consecutive sentences opening with the same 6 words (first kept); 0 = check off.

| Task | Cap | N | Hit cap | Loop stopped | Max tokens | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| bloom_rewrite | 160 | 341 | 0 | 0 | 79 | 26.72 |
| qa | 192 | 146 | 0 | 0 | 94 | 44.9 |
| summarization | 1104 | 146 | 1 | 27 | 1104 | 154.96 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 9.831158
- P95 latency (s): 37.767371

Checkpoint: `C:\Eduguard\models\gguf\qwen15b_multitask_eduguard_v3_9_q4_k_m.gguf`
