# Multi-task Evaluation Report

Generated (UTC): 2026-09-04T02:07:40.032559+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-0.5B-Instruct`

## Dataset

- Test count: **8321**
- Corpus hash: `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite\test.jsonl`

## Bloom rewrite (held-out test)

- N: 1536
- Target accuracy: 0.0
- Macro-F1: 0.0
- Weighted-F1: 0.0
- Fully validated rate: 0.0
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

- Mean latency (s): 1.505063
- P50: 0.349172
- P95: 6.378328
- Model load time (s): 7.209
- RSS (MB): 1849.57
- USS (MB): 1590.89
- GPU memory allocated (MB): 992.04

## Deployment recommendation

**INCONCLUSIVE** — 0.5B base, 0.5B LoRA, 1.5B base, and 1.5B LoRA must be evaluated under the same protocol before model selection.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen05b_multitask_lora\best_adapter`
