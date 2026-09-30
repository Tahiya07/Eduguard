# Multi-task Bloom Rewrite Experiment (Qwen 0.5B vs 1.5B)

Isolated under `experiments/multitask_bloom_rewrite/`.  
**Does not modify** production app code or `models/qwen.gguf`.

## Current machine policy

Implement + audit + prepare + test + resource gate.  
**Do not train** if the resource gate fails.

## Quick start (safe steps)

See `reproduce/`.

## Training (only on a machine that passes the gate)

```powershell
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen05b_multitask.json
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask.json
```

Use a separate env from `requirements-train.txt` so production packages are not upgraded blindly.

## Evaluation (after training)

```powershell
python -m unittest experiments.multitask_bloom_rewrite.tests.test_evaluate_rewrite -v

python experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py `
  --config experiments/multitask_bloom_rewrite/configs/qwen05b_multitask.json `
  --condition lora

python experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py `
  --config experiments/multitask_bloom_rewrite/configs/qwen05b_multitask.json `
  --condition base
```

Results: `experiments/multitask_bloom_rewrite/results/qwen05b_lora/` (or `qwen05b_base/`).

Uses **only** `data/multitask_bloom_rewrite/test.jsonl` (8321 examples). Requires `models/qwen05b_multitask_lora/best_adapter/` on disk.

## v4 corrected Bloom supervision

The v4 branch uses a stricter teacher policy with level-specific cognitive
requirements, output cleanup for Qwen3 thinking tags, negative guards for
cross-level/procedural leakage, and up to three teacher attempts per example.

For the local Qwen3-14B GGUF used in Colab:

```bash
python experiments/bloom_rewrite/scripts/prepare_bloom_rewrite_v4.py \
  --teacher-mode llama_cpp \
  --teacher-model-path /content/qwen3_teacher/Qwen3-14B-Q4_K_M.gguf \
  --split train \
  --limit 100
```

After inspecting the 100-example pilot, generate the full train and validation
splits separately. Do not modify the frozen multitask test set.
