# Multi-task Evaluation Report

Generated (UTC): 2026-09-05T23:30:32.870029+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Test count: **8321**
- Corpus hash: `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite\test.jsonl`

## Bloom rewrite (held-out test)

- N: 1536
- Target accuracy: 0.888021
- Macro-F1: 0.876835
- Weighted-F1: 0.890034
- Fully validated rate: 0.707031
- Semantic preservation rate: 0.824219
- Cognitive validity rate: 0.995443
- Trivial transform rate: 0.001302

## QA (SQuAD held-out test half)

- N: 5285
- Exact Match: 0.57843
- Token F1: 0.785403

## Summarization (PubMed test)

- N: 1500
- ROUGE-1: 0.249109
- ROUGE-2: 0.057348
- ROUGE-L: 0.149551

## Efficiency

- Mean latency (s): 1.469519
- P50: 0.387555
- P95: 6.350181
- Model load time (s): 4.7528
- RSS (MB): 2014.6
- USS (MB): 1754.52
- GPU memory allocated (MB): 3977.11

## Deployment recommendation

**INCONCLUSIVE** — 0.5B base, 0.5B LoRA, 1.5B base, and 1.5B LoRA must be evaluated under the same protocol before model selection.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_lora_v3\best_adapter`
