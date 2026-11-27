# Multi-task Evaluation Report

Generated (UTC): 2026-11-27T07:50:20.861800+00:00
Condition: **base**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **633**
- Dataset manifest SHA-256: `66daa2ca3a02202d6aba1f8c7fc8f66cbb2111902983e61697cc7874ac4ddec8`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_8\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 341
- Target accuracy: 0.615836
- Macro-F1: 0.567996
- Format valid rate: 0.994135
- Trivial transform rate (lower is better): 0.002933
- Topic preservation, near-copy rate, fully validated (calibrated), agreement with the reference rewrite and the classifier's accuracy on the reference rewrites: `evaluation_report_extended.md` (QA and summarization semantic similarity too).

| Pair type | N | Target accuracy | Format valid |
|---|---:|---:|---:|
| direct | 295 | 0.610169 | 0.99661 |
| pivot | 46 | 0.652174 | 0.978261 |

## Closed-book academic QA (held-out validation)

- N: 146
- ROUGE-1 / ROUGE-2 / ROUGE-L: 0.244264 / 0.070252 / 0.16001

## Academic / lecture summarization (held-out validation)

- N: 146
- ROUGE-1: 0.31162
- ROUGE-2: 0.079424
- ROUGE-L: 0.174392
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| lecture_slide_text_style | 1 | 0.371542 | 0.162698 | 0.217391 |
| openstax_textbook_section | 116 | 0.306294 | 0.0772 | 0.164784 |
| original_academic_example | 29 | 0.330857 | 0.085451 | 0.211341 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 51 | 0.299818 | 0.072003 | 0.155956 |
| medium | 44 | 0.297857 | 0.081761 | 0.17226 |
| short | 29 | 0.330857 | 0.085451 | 0.211341 |
| very_long | 22 | 0.341144 | 0.08401 | 0.172687 |

## Output length

Hit cap = stopped by max_new_tokens without EOS. Loop stopped = a verbatim repeat of 24 generated tokens was cut (0 = guard off).

| Task | Cap | N | Hit cap | Loop stopped | Max tokens | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| bloom_rewrite | 160 | 341 | 2 | 0 | 160 | 36.66 |
| qa | 192 | 146 | 137 | 1 | 192 | 189.99 |
| summarization | 1104 | 146 | 0 | 1 | 921 | 479.6 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 12.392208
- P95 latency (s): 40.829178

Checkpoint: `Qwen/Qwen2.5-1.5B-Instruct`
