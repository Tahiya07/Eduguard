# Multi-task Evaluation Report

Generated (UTC): 2026-09-06T16:10:01.532404+00:00
Condition: **lora**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Test count: **8321**
- Corpus hash: `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite\test.jsonl`

## Bloom rewrite (held-out test)

- N: 1536
- Target accuracy: 0.89974
- Macro-F1: 0.886912
- Weighted-F1: 0.898833
- Fully validated rate: 0.556641
- Semantic preservation rate: 0.832682
- Cognitive validity rate: 0.783854
- Trivial transform rate: 0.0

## QA (SQuAD held-out test half)

- N: 5285
- Exact Match: 0.620246
- Token F1: 0.80901

## Summarization (PubMed test)

- N: 1500
- ROUGE-1: 0.361013
- ROUGE-2: 0.111706
- ROUGE-L: 0.210515

## Efficiency

- Mean latency (s): 1.988162
- P50: 0.328683
- P95: 11.439686
- Model load time (s): 5.6599
- RSS (MB): 2031.6
- USS (MB): 1770.52
- GPU memory allocated (MB): 3977.11

## Deployment recommendation

**INCONCLUSIVE** — 0.5B base, 0.5B LoRA, 1.5B base, and 1.5B LoRA must be evaluated under the same protocol before model selection.

Checkpoint: `experiments\multitask_bloom_rewrite\models\qwen15b_multitask_lora_sumfix\best_adapter`
