# Multi-task Evaluation Report

Generated (UTC): 2026-11-23T07:55:42.389221+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **test**
- Evaluated records: **405**
- Dataset manifest SHA-256: `959f86d61cce08f2da9dba317b1ad35d22858b09abee8f4b306a543b144f1568`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate\dataset_manifest.json`

## Bloom rewrite (held-out test)

- N: 135
- Target accuracy: 0.666667
- Macro-F1: 0.67236
- Weighted-F1: 0.663938
- Fully validated rate: 0.392593
- Semantic preservation rate: 0.703704
- Cognitive validity rate: 0.644444
- Trivial transform rate: 0.0

## QA (SQuAD test)

- N: 135
- Exact Match: 0.555556
- Token F1: 0.767323

## Summarization (arXiv test)

- N: 135
- ROUGE-1: 0.432103
- ROUGE-2: 0.160398
- ROUGE-L: 0.259092

## Efficiency

- Mean latency (s): 7.422833
- P50: 3.078254
- P95: 20.181295
- Model load time (s): 7.9549
- RSS (MB): 3184.52
- USS (MB): 2921.61
- GPU memory allocated (MB): 3976.61

## Deployment recommendation

**INCONCLUSIVE** — compare prompt-only, Bloom-only LoRA, the existing multitask LoRA, and task-balanced LoRA under identical split IDs, prompts, and decoding parameters.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_lora_separate_20261016\best_adapter`
