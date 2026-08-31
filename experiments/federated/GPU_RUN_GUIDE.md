# EduGuard Federated Research — GPU Execution Guide

This guide describes how to run the full federated-privacy research pipeline on the university GPU machine. The laptop development environment only runs **CPU_SMOKE** experiments by default.

## 1. Environment setup

```bash
cd /path/to/Eduguard
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-federated.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Dataset and models

Ensure these exist in the repository root:

- `data/figshare_bloom_v1_train.csv`
- `data/figshare_bloom_v1_val.csv`
- `data/figshare_bloom_v1_test.csv`

Hugging Face will download `Qwen/Qwen2.5-0.5B-Instruct` on first training run. Baseline centralized adapter (optional parity check): `models/qwen_bloom_trained0.5B/`.

## 3. Pre-flight checks (run before any GPU experiment)

```bash
python experiments/federated/check_gpu_environment.py
python experiments/federated/preflight.py
```

`check_gpu_environment.py` exits 0 only when CUDA, datasets, and core packages are ready.  
`preflight.py` additionally validates configs, output directories, gate state, and production import isolation.

On the laptop (no CUDA), preflight will report **NOT READY** — that is expected.

## 4. Core GPU run (one command)

```bash
python experiments/federated/preflight.py
python experiments/federated/run_all_experiments.py --no-laptop-mode --allow-gpu --new-run --profile core
```

Resume after shutdown:

```bash
python experiments/federated/run_all_experiments.py --resume --no-laptop-mode --allow-gpu
```

Status:

```bash
python experiments/federated/run_all_experiments.py --status
```

Dry run (no execution):

```bash
python experiments/federated/run_all_experiments.py --dry-run --no-laptop-mode
```

Single phase:

```bash
python experiments/federated/run_all_experiments.py --phase 4 --no-laptop-mode --allow-gpu
```

Single experiment:

```bash
python experiments/federated/run_all_experiments.py --experiment fedavg_iid --no-laptop-mode --allow-gpu
```

## 4. Experiment phases

| Phase | Content | Resource |
|------:|---------|----------|
| 1 | Environment + unit tests | CPU |
| 2 | Production runtime import check | CPU |
| 3 | FL smoke + Framework reproduction | CPU smoke / GPU full |
| 4 | FedAvg/FedProx IID + Dirichlet | GPU |
| 5 | Utility gap report | GPU recommended |
| 6 | **Phase 2A DP validation** (scientific gate) | GPU |
| 7 | DP procedure lock (only if Phase 6 passes) | GPU |
| 8 | Federated DP (blocked until lock) | GPU |
| 9 | SecAgg simulator verification | CPU |
| 11 | Privacy attacks | GPU |
| 14 | Model export (merge adapter) | GPU recommended |
| 16 | Final manifest | CPU |

### DP gate

- Phase 6 (`dp_validation`) runs Opacus gates. If per-sample LoRA gradients fail, **no DP lock is written**.
- Phase 8 (`federated_dp`) is **BLOCKED** until `artifacts/privacy/dp_bloom_validated_v1.json` has `validation_gate_passed: true`.
- Do not weaken tolerances or claim DP without passing all gates.

### Multitask gate

- Federated multitask is **BLOCKED** until `artifacts/federated/results/centralized_multitask_success.json` exists with `"success": true`.

### Privacy attacks

- `run_privacy_attacks.py` is a placeholder. It exits code **2** and the runner reports **NOT_EXECUTED** — not SUCCESS.

### Checkpoint / shutdown recovery

- FL simulation saves `artifacts/federated/runs/<tag>/round_checkpoint.json` after each round.
- Resume FL with `python -m training.federated.simulation --resume ...` (same config).
- Master runner resume: `--resume` loads `experiments/federated/state/run_state.json`, skips completed experiments with matching `run_id` artifacts, continues from first incomplete step.
- `Ctrl+C` marks current experiment **INTERRUPTED**; `--resume` retries it.

### Backup before long runs

- `experiments/federated/state/run_state.json`
- `experiments/federated/results/runs/`
- `artifacts/federated/models/`
- `artifacts/privacy/`

## 6. Experiment phases

### DP gate (Phase 6 → 8)

Federated DP experiments are **BLOCKED** until:

```text
artifacts/privacy/dp_bloom_validated_v1.json
  validation_gate_passed: true
```

Run manually:

```bash
python -m training.centralized.validate_dp_bloom --output artifacts/privacy/dp_bloom_validated_v1.json
```

If validation fails, see `artifacts/privacy/dp_validation_failed_latest.json`. Do **not** claim formal DP until all gates pass.

### Multitask gate (Phase 13)

Federated multitask is blocked until centralized multitask succeeds (`artifacts/federated/results/centralized_multitask_success.json`).

## 6. Expected artifacts

| Path | Description |
|------|-------------|
| `artifacts/federated/results/federated_lora_fedavg_iid.json` | Full FedAvg+IID metrics |
| `artifacts/federated/models/qwen_bloom_federated0.5B_fedavg_iid/` | Global LoRA adapter |
| `artifacts/privacy/dp_bloom_validated_v1.json` | DP lock (only if passed) |
| `experiments/federated/state/run_state.json` | Resume state |
| `experiments/federated/logs/master_run.log` | Master log |
| `experiments/federated/results/runs/<run_id>/run_manifest.json` | Immutable run record |

## 7. Framework parity

Compare GPU FedAvg+IID output against Framework reference:

```text
Framework/results/federated_lora_fedavg_iid1.json  (~50.3% test accuracy)
```

If discrepancy is large (>5% absolute on same split), **stop and diagnose** before continuing.

## 8. Troubleshooting

| Issue | Action |
|-------|--------|
| CUDA not available | Install CUDA-matched PyTorch; use `--allow-gpu` |
| Experiment BLOCKED | Check `--status` and gate artifacts |
| OOM during FL | Reduce `batch_size` in config or use fewer clients for debug |
| DP per-sample grad fails | See failure ladder in master prompt; try uniform CE, grad_accum=1 |
| Resume reruns completed work | State auto-skips when outputs exist and are valid JSON |

## 9. Safe shutdown

Press `Ctrl+C` once. The runner records `INTERRUPTED` in `run_state.json`. Resume with `--resume`.

## 10. Back up before long runs

- `experiments/federated/state/run_state.json`
- `experiments/federated/results/runs/`
- `artifacts/federated/models/`
- `artifacts/privacy/`

## 11. Production deployment note

Federated training does **not** run in `backend/service.py`. After export, point the runtime Bloom classifier to a validated merged artifact under `artifacts/federated/models/`.
