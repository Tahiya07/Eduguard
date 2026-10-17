# Multi-task Evaluation Report

Generated (UTC): 2026-10-17T01:57:04.565542+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **test**
- Evaluated records: **405**
- Dataset manifest SHA-256: `6217ec29cfb4586902002d7f75927af2cabf05c18f38f07354d643152fdea470`
- Manifest: `D:\Eduguard\data\multitask_bloom_rewrite_separate\dataset_manifest.json`

## Bloom rewrite (held-out test)

- N: 135
- Target accuracy: 0.785185
- Macro-F1: 0.773713
- Weighted-F1: 0.777893
- Fully validated rate: 0.637037
- Semantic preservation rate: 0.896296
- Cognitive validity rate: 0.977778
- Trivial transform rate: 0.0

## QA (SQuAD test)

- N: 135
- Exact Match: 0.592593
- Token F1: 0.775024

## Summarization (arXiv test)

- N: 135
- ROUGE-1: 0.432557
- ROUGE-2: 0.162559
- ROUGE-L: 0.2598

## Efficiency

- Mean latency (s): 2.747339
- P50: 1.091554
- P95: 7.760398
- Model load time (s): 5.221
- RSS (MB): 1973.71
- USS (MB): 1712.6
- GPU memory allocated (MB): 3977.11

## Deployment recommendation

**INCONCLUSIVE** — compare prompt-only, Bloom-only LoRA, the existing multitask LoRA, and task-balanced LoRA under identical split IDs, prompts, and decoding parameters.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_lora_separate_20261016\best_adapter`
