# Multi-task Evaluation Report

Generated (UTC): 2026-10-15T21:19:40.717418+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Test count: **14329**
- Corpus hash: `None`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite_final\test.jsonl`

## Bloom rewrite (held-out test)

- N: 135
- Target accuracy: 0.807407
- Macro-F1: 0.793853
- Weighted-F1: 0.797466
- Fully validated rate: 0.681481
- Semantic preservation rate: 0.948148
- Cognitive validity rate: 0.962963
- Trivial transform rate: 0.0

## QA (SQuAD held-out test half)

- N: 300
- Exact Match: 0.62
- Token F1: 0.817665

## Summarization (BillSum test)

- N: 300
- ROUGE-1: 0.441346
- ROUGE-2: 0.175111
- ROUGE-L: 0.271646

## Efficiency

- Mean latency (s): 3.235895
- P50: 1.067642
- P95: 6.870985
- Model load time (s): 6.7493
- RSS (MB): 14226.43
- USS (MB): 13990.91
- GPU memory allocated (MB): 3977.11

## Deployment recommendation

**INCONCLUSIVE** — 0.5B base, 0.5B LoRA, 1.5B base, and 1.5B LoRA must be evaluated under the same protocol before model selection.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_lora_oom_recovery_20261015\best_adapter`
