# First Execution Status (safe steps only)

Measured on the development machine. No training was started.

## Resource gate

```
TRAINING NOT STARTED — INSUFFICIENT RESOURCES
```

| Metric | Measured |
|--------|----------|
| RAM total | 7.72 GB |
| RAM available | ~0.35–1.9 GB (varies) |
| CUDA | false |
| Est. 0.5B CPU LoRA | 12 GB |
| Est. 1.5B CPU LoRA | 28 GB |

## Dataset versions / hashes

| Dataset | Detail |
|---------|--------|
| Figshare Bloom | 2330 raw / 2330 usable; `data/figshare_bloom_v1.csv` |
| Bloom rewrite synth | `bloom_rewrite_synth_v2`; hash `b3725b77862868dcd3d7ad07f1d2e15ae41d6d9887e8510d5396de8c4e790bae`; train 6975 / val 1491 / test 1536 (total 10002) |
| SQuAD 1.1 | Official JSON (HF `rajpurkar/squad` failed on local `datasets==2.19.1`); train 87599; validation/dev 10570; file SHA256 recorded in audit |
| PubMed summarization | `ccdv/pubmed-summarization`; train 119924 (usable 117108); val 6633; test 6658 |

## Multi-task corpus (Mix A, seed 42)

Corpus hash: `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4`

| Split | Total | Bloom | QA | Summarization |
|-------|------:|------:|---:|--------------:|
| train | 17437 | 6975 | 5231 | 5231 |
| validation | 8276 | 1491 | 5285 | 1500 |
| test | 8321 | 1536 | 5285 | 1500 |

Train proportions (actual): Bloom **0.40** / QA **0.30** / Sum **0.30**

## Leakage

Source-level overlaps: **0** for all tasks (train∩val, train∩test, val∩test).  
Source Bloom markers in prompts: **0**.  
Identical full SFT strings across splits: **12** (synthetic template overlap warning only).

## Training configs

`qwen05b_multitask.json` and `qwen15b_multitask.json` differ only in `model_id` / output paths. Decision rule frozen in `configs/decision_rule.json`.

## Tests

`unittest` suite: **12/12 OK**

## Train commands (for a machine that passes the gate)

```powershell
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen05b_multitask.json
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask.json
```

Use `requirements-train.txt` in a separate environment. Do not overwrite `models/qwen.gguf`.
