# EduGuard Federated Privacy Research — Audit Report

**Date:** 2026-08-28  
**Scope:** Repository state vs approved research architecture (audit-only + implementation status)  
**Repositories:** EduGuard (`C:\Users\tahiy\PycharmProjects\Eduguard`), Framework FL (`C:\Users\tahiy\PycharmProjects\Framework\federated\`)

---

## 1. Executive summary

EduGuard is a deployable offline academic assistant with Bloom classification (0.5B HF), local generation (1.5B GGUF), RAG, and layered inference privacy. The federated research stack has been **implemented** under `training/` and `experiments/federated/` with production runtime isolation preserved.

| Component | Status |
|-----------|--------|
| FL infrastructure (FedAvg, FedProx, IID, Dirichlet) | **IMPLEMENTED**, smoke-testable |
| Full FL reproduction on 8 clients × 5 rounds | **GPU-READY / NOT EXECUTED** on laptop |
| Phase 2A DP validation | **FAILED** (per-sample LoRA gradients); gate **CLOSED** |
| Federated DP | **BLOCKED** (stub refuses without lock) |
| SecAgg simulator | **IMPLEMENTED** + unit tested |
| Master experiment runner | **IMPLEMENTED** |
| Production backend regression | **No training imports** in `backend/service.py` |

---

## 2. Base models and runtime paths

| Role | Model | Runtime |
|------|-------|---------|
| Bloom classifier (FL target) | `Qwen/Qwen2.5-0.5B-Instruct` + LoRA r=32, α=64, dropout=0.1, `modules_to_save=["score"]` | HF / PEFT |
| Generator (separate) | `Qwen2.5-1.5B-Instruct` GGUF Q4_K_M | llama.cpp CPU |

Prompt construction: `predict_bloom.build_prompt()`.

---

## 3. Centralized training (`train_qwen_bloom.py`)

Verified configuration:

- 6 Bloom labels, `AutoModelForSequenceClassification`
- LoRA targets: q/k/v/o/gate/up/down_proj
- Optimizer: AdamW, lr=1e-4, weight_decay=0.01, cosine schedule
- batch_size=2, gradient_accumulation_steps=8
- label_smoothing=0.05, balanced class weights (max 3.0)
- Custom weighted Trainer with class weights

Missing at audit start (now added under `training/centralized/`): `merge_model.py`, `evaluate_bloom.py`, `quantize_bloom.py`, `train_bloom_lora.py` wrapper.

---

## 4. Framework FL comparison

| Aspect | Framework | EduGuard `training/federated/` |
|--------|-----------|--------------------------------|
| Algorithm | FedAvg, FedProx (μ=0.01) | Same |
| Partition | IID, Dirichlet | Same (`partition.py`) |
| Trainable state | LoRA + score head | Same (`extract_trainable_state`) |
| Transport | XOR `secure_bundle.py` | **Rejected** — JSON+SHA256 integrity only |
| DP noise | `add_dp_noise` | **Rejected** — not formal DP |
| Reference result | ~50.3% test acc (FedAvg IID) | Not yet reproduced on GPU |

---

## 5. Production isolation

`backend/service.py` (`FrameworkService`) does **not** import `training`, `peft`, `opacus`, or federated modules. Production may consume exported federated artifacts only.

---

## 6. DP validation (Phase 2A)

Command: `python -m training.centralized.validate_dp_bloom`

| Gate | Result (local CPU, Opacus 1.6.0) |
|------|----------------------------------|
| opacus_import | PASS |
| accounting_monotonicity | PASS |
| clipping | PASS |
| loss_variants | PASS |
| per_sample_gradients | **FAIL** — LoRA grad_sample ≠ manual per-example grads |

**No** `artifacts/privacy/dp_bloom_validated_v1.json` written (correct).

Implication: Phase 8 federated DP is **BLOCKED**. Follow failure ladder (uniform CE, Opacus-native loop, score-head-only, grad_accum=1, FFA-LoRA).

---

## 7. Directory structure vs plan

```
training/centralized/     ✓ validate_dp_bloom, merge, evaluate, quantize, train_bloom_lora wrapper
training/federated/       ✓ config, partition, client, server, aggregation, simulation, checkpoint, tasks, transport, secure_aggregation, dp (gated stub)
experiments/federated/    ✓ run_all_experiments.py, configs/, scripts/, GPU_RUN_GUIDE.md
artifacts/federated/      ✓ results/, models/ paths
artifacts/privacy/        ✓ dp_validation_failed_latest.json
tests/federated_training/ ✓ partition, aggregation, transport, secagg, dp_validation, data_locality, runner
```

Not yet implemented: full privacy attack models, federated multitask, task-skew partitions, literature review artifact.

---

## 8. Gaps and risks

1. **DP per-sample gradients on LoRA** — primary blocker for formal DP claims.
2. **Framework parity** — must be verified on GPU before publishing FL results.
3. **Privacy attacks** — script exists but returns `NOT_EXECUTED` until models trained.
4. **Multitask FL** — intentionally deferred pending centralized multitask success.

---

## 9. Recommended GPU execution order

1. `python experiments/federated/run_all_experiments.py --no-laptop-mode --allow-gpu --new-run`
2. Verify Phase 3 smoke + Phase 4 `fedavg_iid` against Framework JSON
3. Run Phase 6 DP validation; only proceed to federated DP if lock passes
4. Resume after any interruption with `--resume`

See `experiments/federated/GPU_RUN_GUIDE.md`.
