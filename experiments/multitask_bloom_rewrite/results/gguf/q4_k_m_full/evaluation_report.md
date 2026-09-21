# GGUF Multi-task Evaluation Report

Generated (UTC): 2026-09-07T11:25:39.417844+00:00
GGUF: `D:\Eduguard\models\gguf\qwen15b_multitask_v3_q4_k_m.gguf`

## Dataset

- Test count: **8321**
- Corpus hash: `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite\test.jsonl`

## Bloom rewrite

- N: 1536
- Target accuracy: 0.889323
- Macro-F1: 0.874765
- Weighted-F1: 0.886423
- Fully validated rate: 0.746745
- Semantic preservation rate: 0.871094
- Cognitive validity rate: 0.992188
- Trivial transform rate: 0.000651

## QA

- N: 5285
- Exact Match: 0.577673
- Token F1: 0.782735

## Summarization

- N: 1500
- ROUGE-1: 0.325903
- ROUGE-2: 0.114891
- ROUGE-L: 0.201548

## Efficiency

- Model load time (s): 0.643711
- Mean latency (s): 1.734619
- P50 latency (s): 0.770284
- P95 latency (s): 6.315068
- RSS (MB): 3181.21
- USS (MB): 2335.93

## Generation

- Temperature: 0.0
- Top-p: 1.0
- Top-k: 0
- Repeat penalty: 1.0
- Max tokens: 128
- Threads: 8
- Context: 1024
- GPU layers: 0
