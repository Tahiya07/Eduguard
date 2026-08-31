# EduGuard GPU Readiness Report

**Generated:** 2026-08-28 (pre-GPU hardening pass)  
**Purpose:** Honest readiness assessment before university GPU execution.

---

## Summary

| Area | Status |
|------|--------|
| Environment readiness | **READY WITH LIMITATION** (CUDA required on GPU PC) |
| Code readiness | **READY** |
| Test readiness | **READY** (lightweight tests pass locally) |
| Checkpoint readiness | **READY WITH LIMITATION** (round-level FL resume; no HF Trainer mid-epoch resume) |
| FL readiness | **READY** (not executed at full scale) |
| DP readiness | **FAILED** (per-sample LoRA gradients) |
| SecAgg readiness | **READY** (simulator + tests) |
| Privacy attack readiness | **NOT IMPLEMENTED** (placeholder exits 2) |
| Multitask readiness | **NOT IMPLEMENTED** (deferred) |
| Deployment readiness | **READY WITH LIMITATION** (export path exists; not validated on GPU) |

---

## Verified from repository

- **0.5B Bloom classifier** with LoRA r=32, α=64, `modules_to_save=["score"]` — `training/federated/config.py`, `client.py`
- **FedAvg / FedProx** — `aggregation.py`, `client.py`, `server.py`
- **IID / Dirichlet** — `partition.py`
- **Client-local training** — `client.py`; raw CSV stays local; transport sends LoRA state only
- **DP gate** — `validate_dp_bloom.py`; federated DP blocked in `training/federated/dp.py`
- **SecAgg simulator** — `secure_aggregation.py` + tests
- **Master runner** — `run_all_experiments.py` with resume, status, dry-run, laptop mode
- **1.5B GGUF separation** — production path unchanged; no FL in `backend/service.py`

---

## Hardening applied this pass

1. **Stale result prevention** — results must include matching `run_id`; auto-skip disabled for mismatched artifacts
2. **Laptop mode** — blocks all `GPU_REQUIRED` / `GPU_RECOMMENDED` by default
3. **DP validation experiment** — no longer incorrectly gated behind itself
4. **Privacy attacks** — exit code 2 → runner reports `NOT_EXECUTED`, not `COMPLETE`
5. **Round-level FL checkpoint** — `artifacts/federated/runs/<tag>/round_checkpoint.json`
6. **Result metadata** — `run_id`, `config_hash`, `dataset_hashes`, `git_revision` in simulation JSON
7. **Communication bytes** — upload/download tracked per round and total (not zero-filled)
8. **Score-head-only DP diagnostic** — non-blocking diagnostic in `validate_dp_bloom.py`
9. **`check_gpu_environment.py`** and **`preflight.py`** — pre-flight tooling

---

## Blocking issues (must resolve on GPU PC)

1. **Phase 2A DP validation FAILED** — LoRA per-sample gradients do not match manual reference
2. **Framework FL parity NOT VERIFIED** — full `fedavg_iid` not executed
3. **Privacy attacks NOT IMPLEMENTED** — placeholder only
4. **Federated DP training loop NOT WIRED** — gate stub only
5. **Multitask FL DEFERRED**

---

## University GPU start sequence

```bash
cd /path/to/Eduguard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-federated.txt

python experiments/federated/check_gpu_environment.py
python experiments/federated/preflight.py
python experiments/federated/run_all_experiments.py --dry-run --new-run --no-laptop-mode

python experiments/federated/run_all_experiments.py --no-laptop-mode --allow-gpu --new-run
```

After shutdown:

```bash
python experiments/federated/run_all_experiments.py --resume --no-laptop-mode --allow-gpu
python experiments/federated/run_all_experiments.py --status
```

**First expensive experiment on GPU:** `fedavg_iid` (8 clients × 5 rounds × 3 local epochs), after CPU phases 1–3 complete.

---

## Scientific limitations (unchanged)

- No formal DP claim until `validation_gate_passed: true`
- SecAgg is a research simulator, not production MPC
- No FL convergence claim until GPU execution completes
- No privacy-attack claim until real attacks run
- No multitask FL until centralized multitask succeeds
