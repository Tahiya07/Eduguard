# GGUF Multi-task Evaluation Report

Generated (UTC): 2026-10-17T03:04:32.145658+00:00
GGUF: `D:\Eduguard\experiments\multitask_bloom_rewrite\models\experiment_C_multitask_capped\merged_model-Q4_K_M.gguf`

## Dataset

- Test count: **405**
- Corpus hash: `6217ec29cfb4586902002d7f75927af2cabf05c18f38f07354d643152fdea470`
- Test file: `D:\Eduguard\data\multitask_bloom_rewrite_separate\dataset_manifest.json`

## Bloom rewrite

- N: 135
- Target accuracy: 0.785185
- Macro-F1: 0.758716
- Weighted-F1: 0.762562
- Fully validated rate: 0.703704
- Semantic preservation rate: 0.962963
- Cognitive validity rate: 0.977778
- Trivial transform rate: 0.007407

## QA

- N: 135
- Exact Match: 0.607407
- Token F1: 0.795189

## Summarization

- N: 135
- ROUGE-1: 0.441568
- ROUGE-2: 0.17311
- ROUGE-L: 0.260554

## Efficiency

- Model load time (s): 3.374381
- Mean latency (s): 7.567015
- P50 latency (s): 0.918924
- P95 latency (s): 29.577746
- RSS (MB): 4003.02
- USS (MB): 3159.99

## Generation

- Temperature: 0.0
- Top-p: 1.0
- Top-k: 0
- Repeat penalty: 1.0
- Max tokens: 128
- Threads: 8
- Context: 8192
- GPU layers: 0
