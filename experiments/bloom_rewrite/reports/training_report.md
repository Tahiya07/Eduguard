# Training report

**Status: TRAINING NOT STARTED — insufficient resources**

No LoRA run was launched. No adapter, no metrics, and no accuracy numbers exist yet. This file must not be cited as evidence that either model is better.

## Resource check (this development machine)

Measured 2026-08-26 with `scripts/check_training_resources.py`:

| Resource | Value |
|---|---|
| RAM total | 7.72 GB |
| RAM available | ~1.0 GB |
| CPU threads | 12 logical / 10 physical |
| CUDA | not available |
| GPU VRAM | n/a |
| torch | 2.11.0+cpu (Python 3.13) |
| peft | not installed on this interpreter |

| Model | Estimated CPU LoRA RAM | Feasible here |
|---|---|---|
| Qwen2.5-0.5B-Instruct | ~12 GB | No |
| Qwen2.5-1.5B-Instruct | ~28 GB | No |

JSON: `experiments/bloom_rewrite/results/resource_feasibility.json`

Additional blocker: this interpreter is missing `peft` and has a broken `transformers` install (`huggingface_hub` missing). Training extras are listed in `experiments/bloom_rewrite/requirements-train.txt`. Production `requirements.txt` was not changed.

## Prepared but not executed

Identical configs:

- `experiments/bloom_rewrite/configs/qwen05b_lora.json`
- `experiments/bloom_rewrite/configs/qwen15b_lora.json`

Seed 42, 3 epochs, lr 2e-4, LoRA r=16 α=32 dropout=0.05, max length 512, batch 2, grad accum 8, cosine schedule, early stopping patience 2.

On a suitable machine, runs write timestamped directories under `results/qwen05b/` and `results/qwen15b/` so previous results are not overwritten.
