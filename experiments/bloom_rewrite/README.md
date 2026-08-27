# Bloom target-rewrite experiment

Controlled comparison of **Qwen2.5-0.5B-Instruct** vs **Qwen2.5-1.5B-Instruct** as EduGuard’s target-level Bloom question rewrite generator.

This tree is **independent of production**. Do not replace `models/qwen.gguf` / `slm_bloom_moderation.gguf` until the experiment is finished and a model is selected from measured evidence.

## Status (Phases 1–8)

| Phase | Status |
|---|---|
| 1 Repository + Figshare inspection | Done. Figshare is classification-only. |
| 2 Public dataset audit | Done. No suitable transformation-pair corpus. CogBench rejected. |
| 3 Dataset strategy | **B**: Figshare source pool + deterministic synthetic transforms |
| 4 Leakage-controlled splits | Done. Group split; zero normalized overlap |
| 5 Synthetic pairs | Done. Qwen 0.5B/1.5B were **not** used as teachers |
| 6 Quality diagnostics | Done. See `data/bloom_rewrite/dataset_statistics.json` |
| 7 Identical LoRA configs | Done. `configs/qwen05b_lora.json`, `configs/qwen15b_lora.json` |
| 8 Resource check | **TRAINING NOT STARTED — insufficient resources** |
| 9–17 Train / eval / GGUF / recommendation | Blocked until a suitable machine is available |

Current machine (2026-08-26): **7.72 GB RAM**, **~1 GB available**, **no CUDA**, CPU-only torch. Estimated CPU LoRA need is ~12 GB (0.5B) and ~28 GB (1.5B).

## Generator task (production-aligned)

**BEFORE (v1, incorrect for production):** `question + source Bloom + target Bloom → rewrite`

**AFTER (v2):** `question + target Bloom → rewrite`

`source_bloom_level` remains in dataset **metadata** for stratification and the source→target matrix. It is **never** in the generator prompt.

This matches EduGuard production: user selects a target level; the rewrite GGUF receives the question and that target (see `build_targeted_rewrite_prompt` — no source level). The Bloom classifier stays a separate validator.

## Dataset

Version: `bloom_rewrite_synth_v2`  
Hash: `b3725b77862868dcd3d7ad07f1d2e15ae41d6d9887e8510d5396de8c4e790bae`  
Seed: `42`

| Split | Examples |
|---|---|
| Train | 6975 |
| Validation | 1491 |
| Test | 1536 |
| Total usable | 10002 |

All 30 non-self source→target cells are present. Understand is over-represented because Figshare is Understand-heavy.

v1 (3-input prompts) is archived under `data/bloom_rewrite_versions/bloom_rewrite_synth_v1/`.

Rebuild / validate:

```text
python experiments/bloom_rewrite/scripts/prepare_dataset.py --new-version
python experiments/bloom_rewrite/scripts/prepare_dataset.py --validate-only
python experiments/bloom_rewrite/scripts/test_prompt_contract.py
```

## Commands (from repo root)

Audit:

```text
python experiments/bloom_rewrite/scripts/audit_figshare.py
```

Prepare / validate (refuses silent overwrite):

```text
python experiments/bloom_rewrite/scripts/prepare_dataset.py
```

Resource check:

```text
python experiments/bloom_rewrite/scripts/benchmark_resources.py
```

Train 0.5B (only if the resource check passes):

```text
python experiments/bloom_rewrite/scripts/train_bloom_rewrite_lora.py --config experiments/bloom_rewrite/configs/qwen05b_lora.json
```

Train 1.5B:

```text
python experiments/bloom_rewrite/scripts/train_bloom_rewrite_lora.py --config experiments/bloom_rewrite/configs/qwen15b_lora.json
```

Evaluate:

```text
python experiments/bloom_rewrite/scripts/evaluate_rewrite.py --model-key 0.5b --role lora --generator-path <adapter-or-merged>
python experiments/bloom_rewrite/scripts/evaluate_rewrite.py --model-key 1.5b --role lora --generator-path <adapter-or-merged>
```

Compare:

```text
python experiments/bloom_rewrite/scripts/compare_models.py --pred-a experiments/bloom_rewrite/results/qwen05b/evaluation/predictions.jsonl --pred-b experiments/bloom_rewrite/results/qwen15b/evaluation/predictions.jsonl --metrics-a experiments/bloom_rewrite/results/qwen05b/evaluation/metrics.json --metrics-b experiments/bloom_rewrite/results/qwen15b/evaluation/metrics.json
```

GGUF benchmark:

```text
python experiments/bloom_rewrite/scripts/benchmark_gguf.py --gguf <path.gguf> --model-key 0.5b
```

Training extras (do not add these to production `requirements.txt`):

```text
pip install -r experiments/bloom_rewrite/requirements-train.txt
```

## Training format (identical for both models)

```text
SYSTEM:
You are an expert academic assessment editor.
Rewrite an academic question so that the student's required cognitive
task matches the requested Bloom level.
Preserve the original topic, important technical concepts, and academic intent.
Do not merely replace verbs.
Output only one student-facing exam question.

USER:
Original question:
{source_question}

Target Bloom level:
{target_bloom_level}

ASSISTANT:
{target_rewrite}
```

Assert: `"Original Bloom level:"` / `"Source Bloom level:"` must not appear in any SFT prompt.

Production note: live `build_targeted_rewrite_prompt()` also injects per-level task policy and examples; both paths are 2-input (`question` + `target`). Policy/example wrapping can be re-aligned at deployment without changing the trained task.

## Hyperparameters (identical)

| Setting | Value |
|---|---|
| Seed | 42 |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Max seq length | 512 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Batch size | 2 |
| Grad accumulation | 8 |
| Decoding for eval | greedy, max 96 new tokens |

## Anti-cheating rules enforced here

- Figshare is not represented as rewrite supervision.
- No Qwen 0.5B/1.5B self-training on its own generations.
- Splits are by question-family group, not row.
- The classifier must stay **fixed** during 0.5B vs 1.5B comparison.
- Do not train GGUF directly.
- Do not overwrite previous run directories.
- Do not claim a winner without measured paired evaluation.
