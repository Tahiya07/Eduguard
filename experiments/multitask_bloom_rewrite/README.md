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
