# Multi-task Evaluation Report

Generated (UTC): 2026-10-17T01:22:22.170092+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **636**
- Dataset manifest SHA-256: `6217ec29cfb4586902002d7f75927af2cabf05c18f38f07354d643152fdea470`
- Manifest: `D:\Eduguard\data\multitask_bloom_rewrite_separate\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 212
- Target accuracy: 0.811321
- Macro-F1: 0.803025
- Weighted-F1: 0.810248
- Fully validated rate: 0.646226
- Semantic preservation rate: 0.84434
- Cognitive validity rate: 1.0
- Trivial transform rate: 0.004717

## QA (SQuAD validation)

- N: 212
- Exact Match: 0.108491
- Token F1: 0.35182

## Summarization (arXiv validation)

- N: 212
- ROUGE-1: 0.332873
- ROUGE-2: 0.086756
- ROUGE-L: 0.181119

## Efficiency

- Mean latency (s): 2.913926
- P50: 1.322554
- P95: 6.950979
- Model load time (s): 5.2257
- RSS (MB): 8481.72
- USS (MB): 8248.37
- GPU memory allocated (MB): 3977.11

## Deployment recommendation

**INCONCLUSIVE** — compare prompt-only, Bloom-only LoRA, the existing multitask LoRA, and task-balanced LoRA under identical split IDs, prompts, and decoding parameters.

Checkpoint: `experiments\multitask_bloom_rewrite\models\experiment_B_bloom_rewrite_only\best_adapter`
