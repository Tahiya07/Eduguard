# Multi-task Evaluation Report

Generated (UTC): 2026-09-08T06:35:52.447610+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Test count: **8321**
- Corpus hash: `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite\test.jsonl`

## Bloom rewrite (held-out test)

- N: 1536
- Target accuracy: 0.951823
- Macro-F1: 0.944287
- Weighted-F1: 0.951474
- Fully validated rate: 0.76237
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

- Mean latency (s): 1.618745
- P50: 0.43601
- P95: 6.87914
- Model load time (s): 14.1715
- RSS (MB): 2033.03
- USS (MB): 1771.57
- GPU memory allocated (MB): 4044.25

## Deployment recommendation

**INCONCLUSIVE** — 0.5B base, 0.5B LoRA, 1.5B base, and 1.5B LoRA must be evaluated under the same protocol before model selection.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_lora_v3\best_adapter`
