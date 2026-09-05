# Multi-task Evaluation Report

Generated (UTC): 2026-09-05T19:00:54.309623+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Test count: **8321**
- Corpus hash: `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite\test.jsonl`

## Bloom rewrite (held-out test)

- N: 1536
- Target accuracy: 0.555339
- Macro-F1: 0.49944
- Weighted-F1: 0.506866
- Fully validated rate: 0.525391
- Semantic preservation rate: 0.998698
- Cognitive validity rate: 1.0
- Trivial transform rate: 0.0

## QA (SQuAD held-out test half)

- N: 5285
- Exact Match: 0.607758
- Token F1: 0.802765

## Summarization (PubMed test)

- N: 1500
- ROUGE-1: 0.245588
- ROUGE-2: 0.055844
- ROUGE-L: 0.146661

## Efficiency

- Mean latency (s): 1.401479
- P50: 0.330946
- P95: 5.835658
- Model load time (s): 5.0304
- RSS (MB): 2020.76
- USS (MB): 1759.91
- GPU memory allocated (MB): 3977.11

## Deployment recommendation

**INCONCLUSIVE** — 0.5B base, 0.5B LoRA, 1.5B base, and 1.5B LoRA must be evaluated under the same protocol before model selection.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_lora_v3\best_adapter`
