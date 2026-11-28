# Multi-task Evaluation Report

Generated (UTC): 2026-11-28T06:39:00.952879+00:00
Condition: **gguf**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **633**
- Dataset manifest SHA-256: `10a9e969ff6f9e31a54cee956033e204a9926e260b1be5fe994811cb2b6b3040`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_9\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 0
- Target accuracy: 0.0
- Macro-F1: 0.0
- Format valid rate: 0.0
- Trivial transform rate (lower is better): 0.0
- Topic preservation, near-copy rate, fully validated (calibrated), agreement with the reference rewrite and the classifier's accuracy on the reference rewrites: `evaluation_report_extended.md` (QA and summarization semantic similarity too).

## Closed-book academic QA (held-out validation)

- N: 0
- ROUGE-1 / ROUGE-2 / ROUGE-L: None / None / None

## Academic / lecture summarization (held-out validation)

- N: 146
- ROUGE-1: 0.411836
- ROUGE-2: 0.124054
- ROUGE-L: 0.244876
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| lecture_slide_text_style | 1 | 0.492239 | 0.227171 | 0.31929 |
| openstax_textbook_section | 116 | 0.386074 | 0.11146 | 0.22089 |
| original_academic_example | 29 | 0.512111 | 0.170877 | 0.338252 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 51 | 0.376283 | 0.108198 | 0.207862 |
| medium | 44 | 0.395616 | 0.122028 | 0.245126 |
| short | 29 | 0.512111 | 0.170877 | 0.338252 |
| very_long | 22 | 0.39451 | 0.103145 | 0.207094 |

## Output length

Hit cap = stopped by max_new_tokens without EOS. Loop stopped = output cut at a degenerate loop: a verbatim repeat of 24 generated tokens (cut to the last full sentence) or 4 consecutive sentences opening with the same 6 words (first kept); 0 = check off.

| Task | Cap | N | Hit cap | Loop stopped | Max tokens | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| summarization | 1104 | 146 | 0 | 7 | 751 | 152.75 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 14.320784
- P95 latency (s): 30.465869

Checkpoint: `C:\Eduguard\models\gguf\qwen15b_multitask_eduguard_v3_9_q4_k_m.gguf`
