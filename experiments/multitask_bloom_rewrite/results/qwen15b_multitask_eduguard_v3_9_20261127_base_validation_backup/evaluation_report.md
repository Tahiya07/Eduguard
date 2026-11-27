# Multi-task Evaluation Report

Generated (UTC): 2026-11-27T14:31:24.736421+00:00
Condition: **base**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **633**
- Dataset manifest SHA-256: `10a9e969ff6f9e31a54cee956033e204a9926e260b1be5fe994811cb2b6b3040`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_9\dataset_manifest.json`

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
- ROUGE-1 / ROUGE-2 / ROUGE-L: 0.244338 / 0.070275 / 0.160062

## Academic / lecture summarization (held-out validation)

- N: 146
- ROUGE-1: 0.311628
- ROUGE-2: 0.079426
- ROUGE-L: 0.174396
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| lecture_slide_text_style | 1 | 0.371542 | 0.162698 | 0.217391 |
| openstax_textbook_section | 116 | 0.306304 | 0.077201 | 0.164789 |
| original_academic_example | 29 | 0.330857 | 0.085451 | 0.211341 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 51 | 0.299842 | 0.072007 | 0.155968 |
| medium | 44 | 0.297857 | 0.081761 | 0.17226 |
| short | 29 | 0.330857 | 0.085451 | 0.211341 |
| very_long | 22 | 0.341144 | 0.08401 | 0.172687 |

## Output length

Hit cap = stopped by max_new_tokens without EOS. Loop stopped = output cut at a degenerate loop: a verbatim repeat of 24 generated tokens (cut to the last full sentence) or 4 consecutive sentences opening with the same 6 words (first kept); 0 = check off.

| Task | Cap | N | Hit cap | Loop stopped | Max tokens | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| bloom_rewrite | 160 | 341 | 2 | 0 | 160 | 36.66 |
| qa | 192 | 146 | 137 | 1 | 192 | 189.94 |
| summarization | 1104 | 146 | 0 | 1 | 921 | 479.58 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 12.939662
- P95 latency (s): 42.739623

Checkpoint: `Qwen/Qwen2.5-1.5B-Instruct`
