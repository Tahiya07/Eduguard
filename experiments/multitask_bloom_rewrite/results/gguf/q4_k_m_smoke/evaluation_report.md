# GGUF Multi-task Evaluation Report

Generated (UTC): 2026-09-07T07:21:56.922002+00:00
GGUF: `D:\Eduguard\models\gguf\qwen15b_multitask_v3_q4_k_m.gguf`

## Dataset

- Test count: **8321**
- Corpus hash: `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite\test.jsonl`

## Bloom rewrite

- N: 2
- Target accuracy: 0.5
- Macro-F1: 0.166667
- Weighted-F1: 0.5
- Fully validated rate: 0.5
- Semantic preservation rate: 1.0
- Cognitive validity rate: 1.0
- Trivial transform rate: 0.0

## QA

- N: 2
- Exact Match: 0.5
- Token F1: 0.833333

## Summarization

- N: 2
- ROUGE-1: 0.409761
- ROUGE-2: 0.135187
- ROUGE-L: 0.229031

## Efficiency

- Model load time (s): 0.581673
- Mean latency (s): 2.174906
- P50 latency (s): 0.579248
- P95 latency (s): 5.495216
- RSS (MB): 3820.55
- USS (MB): 2975.52

## Generation

- Temperature: 0.0
- Top-p: 1.0
- Top-k: 0
- Repeat penalty: 1.0
- Max tokens: 128
- Threads: 8
- Context: 1024
- GPU layers: 0
