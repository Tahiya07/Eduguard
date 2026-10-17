# Multi-task Evaluation Report

Generated (UTC): 2026-10-17T00:06:12.340541+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **636**
- Dataset manifest SHA-256: `6217ec29cfb4586902002d7f75927af2cabf05c18f38f07354d643152fdea470`
- Manifest: `D:\Eduguard\data\multitask_bloom_rewrite_separate\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 212
- Target accuracy: 0.773585
- Macro-F1: 0.759148
- Weighted-F1: 0.763591
- Fully validated rate: 0.627358
- Semantic preservation rate: 0.863208
- Cognitive validity rate: 1.0
- Trivial transform rate: 0.0

## QA (SQuAD validation)

- N: 212
- Exact Match: 0.683962
- Token F1: 0.837897

## Summarization (arXiv validation)

- N: 212
- ROUGE-1: 0.43951
- ROUGE-2: 0.170411
- ROUGE-L: 0.266847

## Efficiency

- Mean latency (s): 2.764468
- P50: 1.040139
- P95: 7.067953
- Model load time (s): 7.0122
- RSS (MB): 8418.35
- USS (MB): 8184.93
- GPU memory allocated (MB): 3977.11

## Deployment recommendation

**INCONCLUSIVE** — compare prompt-only, Bloom-only LoRA, the existing multitask LoRA, and task-balanced LoRA under identical split IDs, prompts, and decoding parameters.

Checkpoint: `D:\Eduguard\experiments\multitask_bloom_rewrite\models\qwen15b_multitask_lora_separate_20261016\best_adapter`
