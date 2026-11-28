# Multi-task Evaluation Report

Generated (UTC): 2026-11-27T16:39:56.179531+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **test**
- Evaluated records: **645**
- Dataset manifest SHA-256: `10a9e969ff6f9e31a54cee956033e204a9926e260b1be5fe994811cb2b6b3040`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_9\dataset_manifest.json`

## Bloom rewrite (held-out test)

- N: 351
- Target accuracy: 0.925926
- Macro-F1: 0.925991
- Format valid rate: 1.0
- Trivial transform rate (lower is better): 0.0
- Topic preservation, near-copy rate, fully validated (calibrated), agreement with the reference rewrite and the classifier's accuracy on the reference rewrites: `evaluation_report_extended.md` (QA and summarization semantic similarity too).

| Pair type | N | Target accuracy | Format valid |
|---|---:|---:|---:|
| direct | 280 | 0.925 | 1.0 |
| pivot | 71 | 0.929577 | 1.0 |

## Closed-book academic QA (held-out test)

- N: 147
- ROUGE-1 / ROUGE-2 / ROUGE-L: 0.387133 / 0.116065 / 0.267858

## Academic / lecture summarization (held-out test)

- N: 147
- ROUGE-1: 0.399425
- ROUGE-2: 0.121411
- ROUGE-L: 0.247822
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| lecture_slide_text_style | 2 | 0.412491 | 0.116777 | 0.226762 |
| openstax_textbook_section | 116 | 0.370758 | 0.106973 | 0.218189 |
| original_academic_example | 29 | 0.513196 | 0.179485 | 0.36781 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 52 | 0.393961 | 0.116338 | 0.226911 |
| medium | 44 | 0.355923 | 0.101237 | 0.218034 |
| short | 29 | 0.513196 | 0.179485 | 0.36781 |
| very_long | 22 | 0.349376 | 0.0972 | 0.198661 |

## Output length

Hit cap = stopped by max_new_tokens without EOS. Loop stopped = output cut at a degenerate loop: a verbatim repeat of 24 generated tokens (cut to the last full sentence) or 4 consecutive sentences opening with the same 6 words (first kept); 0 = check off.

| Task | Cap | N | Hit cap | Loop stopped | Max tokens | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| bloom_rewrite | 160 | 351 | 1 | 1 | 160 | 27.39 |
| qa | 192 | 147 | 0 | 0 | 118 | 47.59 |
| summarization | 1104 | 147 | 0 | 4 | 431 | 110.76 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 8.039276
- P95 latency (s): 20.267377

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_eduguard_v3_9_20261127\best_adapter`
