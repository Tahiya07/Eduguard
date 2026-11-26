# Multi-task Evaluation Report

Generated (UTC): 2026-11-26T10:04:54.313517+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **505**
- Dataset manifest SHA-256: `ea9a2e954d655b69ae2faa6b1a5392bfc92b6e181e5a29544d65f40b5c163077`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_3\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 295
- Target accuracy: 0.816949
- Macro-F1: 0.814095
- Weighted-F1: 0.801747
- Fully validated rate: 0.349153
- Semantic preservation rate: 0.484746
- Cognitive validity rate: 0.857627
- Trivial transform rate: 0.0

## Closed-book academic QA (held-out validation)

- N: 146
- ROUGE-1 / ROUGE-2 / ROUGE-L: 0.38545 / 0.121605 / 0.275394
- BERTScore F1: not computed (bert_score not installed)
- Token F1: 0.341543
- Exact match: 0.0 (free-form references; expected near 0)

## Academic / lecture summarization (held-out validation)

- N: 64
- ROUGE-1: 0.44896
- ROUGE-2: 0.160309
- ROUGE-L: 0.299383
- BERTScore F1: not computed (bert_score not installed)
- Empty predictions: 0

| Source type | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| arxiv_scientific_article | 33 | 0.429629 | 0.163951 | 0.269923 |
| lecture_slide_text_style | 2 | 0.514932 | 0.235115 | 0.359879 |
| original_academic_example | 29 | 0.466408 | 0.151006 | 0.328736 |

| Length category | N | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| long | 9 | 0.370343 | 0.135151 | 0.223262 |
| medium | 4 | 0.500311 | 0.178802 | 0.326404 |
| short | 29 | 0.466408 | 0.151006 | 0.328736 |
| very_long | 22 | 0.448785 | 0.179502 | 0.28692 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 6.674042
- P50: 5.189575
- P95: 17.455791
- Model load time (s): 10.14
- RSS (MB): 2294.79
- USS (MB): 2025.43
- GPU memory allocated (MB): 3977.42

## Deployment recommendation

**INCONCLUSIVE** — compare prompt-only, Bloom-only LoRA, the existing multitask LoRA, and task-balanced LoRA under identical split IDs, prompts, and decoding parameters.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_eduguard_v3_3_20261126\best_adapter`
