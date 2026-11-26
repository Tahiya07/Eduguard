# Multi-task Evaluation Report

Generated (UTC): 2026-11-26T11:39:35.251591+00:00
Condition: **base**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **505**
- Dataset manifest SHA-256: `ea9a2e954d655b69ae2faa6b1a5392bfc92b6e181e5a29544d65f40b5c163077`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_3\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 295
- Target accuracy: 0.610169
- Macro-F1: 0.570065
- Weighted-F1: 0.555321
- Fully validated rate: 0.484746
- Semantic preservation rate: 0.844068
- Cognitive validity rate: 0.99661
- Trivial transform rate: 0.0

## Closed-book academic QA (held-out validation)

- N: 146
- ROUGE-1 / ROUGE-2 / ROUGE-L: 0.243934 / 0.070151 / 0.159943
- BERTScore F1: not computed (bert_score not installed)
- Token F1: 0.223076
- Exact match: 0.0 (free-form references; expected near 0)

## Academic / lecture summarization (held-out validation)

- N: 64
- ROUGE-1: 0.333397
- ROUGE-2: 0.094427
- ROUGE-L: 0.190701
- BERTScore F1: not computed (bert_score not installed)
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| arxiv_scientific_article | 33 | 0.326217 | 0.094231 | 0.16583 |
| lecture_slide_text_style | 2 | 0.444011 | 0.220038 | 0.274788 |
| original_academic_example | 29 | 0.333939 | 0.085988 | 0.213204 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 9 | 0.311808 | 0.087754 | 0.161723 |
| medium | 4 | 0.38858 | 0.156441 | 0.21946 |
| short | 29 | 0.333939 | 0.085988 | 0.213204 |
| very_long | 22 | 0.331481 | 0.097008 | 0.167665 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 8.792371
- P50: 3.355667
- P95: 31.154088
- Model load time (s): 6.5975
- RSS (MB): 2252.86
- USS (MB): 1986.14
- GPU memory allocated (MB): 3906.99

## Deployment recommendation

**INCONCLUSIVE** — compare prompt-only, Bloom-only LoRA, the existing multitask LoRA, and task-balanced LoRA under identical split IDs, prompts, and decoding parameters.

Checkpoint: `Qwen/Qwen2.5-1.5B-Instruct`
