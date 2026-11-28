# Multi-task Evaluation Report

Generated (UTC): 2026-11-28T01:12:44.069499+00:00
Condition: **gguf**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **test**
- Evaluated records: **645**
- Dataset manifest SHA-256: `10a9e969ff6f9e31a54cee956033e204a9926e260b1be5fe994811cb2b6b3040`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_9\dataset_manifest.json`

## Bloom rewrite (held-out test)

- N: 351
- Target accuracy: 0.891738
- Macro-F1: 0.892094
- Format valid rate: 1.0
- Trivial transform rate (lower is better): 0.0
- Topic preservation, near-copy rate, fully validated (calibrated), agreement with the reference rewrite and the classifier's accuracy on the reference rewrites: `evaluation_report_extended.md` (QA and summarization semantic similarity too).

| Pair type | N | Target accuracy | Format valid |
|---|---:|---:|---:|
| direct | 280 | 0.896429 | 1.0 |
| pivot | 71 | 0.873239 | 1.0 |

## Closed-book academic QA (held-out test)

- N: 147
- ROUGE-1 / ROUGE-2 / ROUGE-L: 0.413459 / 0.151077 / 0.308301

## Academic / lecture summarization (held-out test)

- N: 147
- ROUGE-1: 0.402734
- ROUGE-2: 0.134571
- ROUGE-L: 0.256091
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| lecture_slide_text_style | 2 | 0.421087 | 0.360496 | 0.378952 |
| openstax_textbook_section | 116 | 0.371932 | 0.11625 | 0.223603 |
| original_academic_example | 29 | 0.524674 | 0.192274 | 0.377569 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 52 | 0.386702 | 0.119321 | 0.232233 |
| medium | 44 | 0.355287 | 0.124792 | 0.226015 |
| short | 29 | 0.524674 | 0.192274 | 0.377569 |
| very_long | 22 | 0.374782 | 0.114111 | 0.212502 |

## Output length

Hit cap = stopped by max_new_tokens without EOS. Loop stopped = output cut at a degenerate loop: a verbatim repeat of 24 generated tokens (cut to the last full sentence) or 4 consecutive sentences opening with the same 6 words (first kept); 0 = check off.

| Task | Cap | N | Hit cap | Loop stopped | Max tokens | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| bloom_rewrite | 160 | 351 | 0 | 2 | 95 | 27.1 |
| qa | 192 | 147 | 0 | 0 | 140 | 44.28 |
| summarization | 1104 | 147 | 1 | 25 | 1104 | 151.74 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 5.468595
- P95 latency (s): 21.345051

Checkpoint: `C:\Eduguard\models\gguf\qwen15b_multitask_eduguard_v3_9_q4_k_m.gguf`
