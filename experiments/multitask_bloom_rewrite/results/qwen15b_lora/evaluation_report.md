# Multi-task Evaluation Report

Generated (UTC): 2026-09-04T19:42:09.282870+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Test count: **8321**
- Corpus hash: `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite\test.jsonl`

## Bloom rewrite (held-out test)

- N: 1536
- Target accuracy: 0.794922
- Macro-F1: 0.761343
- Weighted-F1: 0.783357
- Fully validated rate: 0.421875
- Semantic preservation rate: 0.985026
- Cognitive validity rate: 0.570964
- Trivial transform rate: 0.0

## QA (SQuAD held-out test half)

- N: 5285
- Exact Match: 0.616272
- Token F1: 0.81299

## Summarization (PubMed test)

- N: 1500
- ROUGE-1: 0.244139
- ROUGE-2: 0.055728
- ROUGE-L: 0.146033

## Efficiency

- Mean latency (s): 1.677107
- P50: 0.426104
- P95: 7.670689
- Model load time (s): 12.3606
- RSS (MB): 2022.93
- USS (MB): 1762.32
- GPU memory allocated (MB): 3976.8

## Deployment recommendation

**INCONCLUSIVE** — 0.5B base, 0.5B LoRA, 1.5B base, and 1.5B LoRA must be evaluated under the same protocol before model selection.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_lora\best_adapter`
