# Multi-task Evaluation Report

Generated (UTC): 2026-11-27T16:42:28.015647+00:00
Condition: **gguf**
Model: `Qwen/Qwen2.5-1.5B-Instruct`

## Dataset

- Evaluated split: **validation**
- Evaluated records: **633**
- Dataset manifest SHA-256: `10a9e969ff6f9e31a54cee956033e204a9926e260b1be5fe994811cb2b6b3040`
- Manifest: `C:\Eduguard\data\multitask_bloom_rewrite_separate_eduguard_v3_9\dataset_manifest.json`

## Bloom rewrite (held-out validation)

- N: 341
- Target accuracy: 0.88563
- Macro-F1: 0.887659
- Format valid rate: 1.0
- Trivial transform rate (lower is better): 0.02346
- Topic preservation, near-copy rate, fully validated (calibrated), agreement with the reference rewrite and the classifier's accuracy on the reference rewrites: `evaluation_report_extended.md` (QA and summarization semantic similarity too).

| Pair type | N | Target accuracy | Format valid |
|---|---:|---:|---:|
| direct | 295 | 0.874576 | 1.0 |
| pivot | 46 | 0.956522 | 1.0 |

## Closed-book academic QA (held-out validation)

- N: 0
- ROUGE-1 / ROUGE-2 / ROUGE-L: None / None / None

## Academic / lecture summarization (held-out validation)

- N: 0
- ROUGE-1: None
- ROUGE-2: None
- ROUGE-L: None
- Empty predictions: 0

## Output length

Hit cap = stopped by max_new_tokens without EOS. Loop stopped = output cut at a degenerate loop: a verbatim repeat of 24 generated tokens (cut to the last full sentence) or 4 consecutive sentences opening with the same 6 words (first kept); 0 = check off.

| Task | Cap | N | Hit cap | Loop stopped | Max tokens | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| bloom_rewrite | 160 | 341 | 0 | 0 | 71 | 25.28 |

Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier, not human ground truth.

## Efficiency

- Mean latency (s): 6.107887
- P95 latency (s): 8.970413

Checkpoint: `C:\Eduguard\models\gguf\qwen25_1_5b_instruct_base_q4_k_m.gguf`
