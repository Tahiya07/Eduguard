# Multi-task Evaluation Report

Generated (UTC): 2026-09-04T23:18:18.828722+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-0.5B-Instruct`

## Dataset

- Test count: **8321**
- Corpus hash: `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite\test.jsonl`

## Bloom rewrite (held-out test)

- N: 1536
- Target accuracy: 0.763021
- Macro-F1: 0.719168
- Weighted-F1: 0.743839
- Fully validated rate: 0.366536
- Semantic preservation rate: 0.991536
- Cognitive validity rate: 0.525391
- Trivial transform rate: 0.0

## QA (SQuAD held-out test half)

- N: 5285
- Exact Match: 0.535667
- Token F1: 0.728535

## Summarization (PubMed test)

- N: 1500
- ROUGE-1: 0.251969
- ROUGE-2: 0.057809
- ROUGE-L: 0.155028

## Efficiency

- Mean latency (s): 1.47339
- P50: 0.277946
- P95: 6.490055
- Model load time (s): 6.3381
- RSS (MB): 2014.13
- USS (MB): 1753.78
- GPU memory allocated (MB): 1942.22

## Deployment recommendation

**INCONCLUSIVE** — 0.5B base, 0.5B LoRA, 1.5B base, and 1.5B LoRA must be evaluated under the same protocol before model selection.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen05b_multitask_lora\best_adapter`
