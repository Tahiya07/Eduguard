# Multi-task Evaluation Report

Generated (UTC): 2026-10-11T16:44:50.787495+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Test count: **13952**
- Corpus hash: `None`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite_final\test.jsonl`

## Bloom rewrite (held-out test)

- N: 114
- Target accuracy: 0.587719
- Macro-F1: 0.570993
- Weighted-F1: 0.570993
- Fully validated rate: 0.526316
- Semantic preservation rate: 0.991228
- Cognitive validity rate: 0.95614
- Trivial transform rate: 0.0

## QA (SQuAD held-out test half)

- N: 114
- Exact Match: 0.640351
- Token F1: 0.831071

## Summarization (BillSum test)

- N: 114
- ROUGE-1: 0.497957
- ROUGE-2: 0.312939
- ROUGE-L: 0.38741

## Efficiency

- Mean latency (s): 2.299943
- P50: 1.154823
- P95: 6.186941
- Model load time (s): 6.3193
- RSS (MB): 2186.13
- USS (MB): 1928.74
- GPU memory allocated (MB): 3977.11

## Deployment recommendation

**INCONCLUSIVE** — 0.5B base, 0.5B LoRA, 1.5B base, and 1.5B LoRA must be evaluated under the same protocol before model selection.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_lora_final\best_adapter`
