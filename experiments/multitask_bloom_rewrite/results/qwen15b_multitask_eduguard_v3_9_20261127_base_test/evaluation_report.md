# Multi-task Evaluation Report

Generated (UTC): 2026-11-27T19:57:40.953010+00:00
Condition: **base**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **test**
- Evaluated records: **645**
- Dataset manifest SHA-256: `10a9e969ff6f9e31a54cee956033e204a9926e260b1be5fe994811cb2b6b3040`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_9\dataset_manifest.json`

## Bloom rewrite (held-out test)

- N: 351
- Target accuracy: 0.643875
- Macro-F1: 0.600755
- Format valid rate: 0.994302
- Trivial transform rate (lower is better): 0.005698
- Topic preservation, near-copy rate, fully validated (calibrated), agreement with the reference rewrite and the classifier's accuracy on the reference rewrites: `evaluation_report_extended.md` (QA and summarization semantic similarity too).

| Pair type | N | Target accuracy | Format valid |
|---|---:|---:|---:|
| direct | 280 | 0.635714 | 0.992857 |
| pivot | 71 | 0.676056 | 1.0 |

## Closed-book academic QA (held-out test)

- N: 147
- ROUGE-1 / ROUGE-2 / ROUGE-L: 0.24708 / 0.066723 / 0.160627

## Academic / lecture summarization (held-out test)

- N: 147
- ROUGE-1: 0.306029
- ROUGE-2: 0.07856
- ROUGE-L: 0.17017
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| lecture_slide_text_style | 2 | 0.372302 | 0.146364 | 0.245747 |
| openstax_textbook_section | 116 | 0.295421 | 0.073275 | 0.15627 |
| original_academic_example | 29 | 0.343893 | 0.095024 | 0.220555 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 52 | 0.304719 | 0.078929 | 0.160611 |
| medium | 44 | 0.272486 | 0.071322 | 0.153188 |
| short | 29 | 0.343893 | 0.095024 | 0.220555 |
| very_long | 22 | 0.3263 | 0.070461 | 0.160309 |

## Output length

Hit cap = stopped by max_new_tokens without EOS. Loop stopped = output cut at a degenerate loop: a verbatim repeat of 24 generated tokens (cut to the last full sentence) or 4 consecutive sentences opening with the same 6 words (first kept); 0 = check off.

| Task | Cap | N | Hit cap | Loop stopped | Max tokens | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| bloom_rewrite | 160 | 351 | 0 | 0 | 116 | 35.51 |
| qa | 192 | 147 | 138 | 0 | 192 | 189.29 |
| summarization | 1104 | 147 | 0 | 2 | 867 | 463.66 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 12.212647
- P95 latency (s): 39.687396

Checkpoint: `Qwen/Qwen2.5-1.5B-Instruct`
