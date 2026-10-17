# Multi-task Evaluation Report

Generated (UTC): 2026-10-17T00:41:48.994623+00:00
Condition: **base**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **636**
- Dataset manifest SHA-256: `6217ec29cfb4586902002d7f75927af2cabf05c18f38f07354d643152fdea470`
- Manifest: `D:\Eduguard\data\multitask_bloom_rewrite_separate\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 212
- Target accuracy: 0.160377
- Macro-F1: 0.153791
- Weighted-F1: 0.156363
- Fully validated rate: 0.113208
- Semantic preservation rate: 0.70283
- Cognitive validity rate: 0.985849
- Trivial transform rate: 0.014151

## QA (SQuAD validation)

- N: 212
- Exact Match: 0.099057
- Token F1: 0.320115

## Summarization (arXiv validation)

- N: 212
- ROUGE-1: 0.329744
- ROUGE-2: 0.079807
- ROUGE-L: 0.17796

## Efficiency

- Mean latency (s): 1.74318
- P50: 0.659908
- P95: 4.016528
- Model load time (s): 5.8185
- RSS (MB): 8347.79
- USS (MB): 8113.69
- GPU memory allocated (MB): 3906.36

## Deployment recommendation

**INCONCLUSIVE** — compare prompt-only, Bloom-only LoRA, the existing multitask LoRA, and task-balanced LoRA under identical split IDs, prompts, and decoding parameters.

Checkpoint: `Qwen/Qwen2.5-1.5B-Instruct`
