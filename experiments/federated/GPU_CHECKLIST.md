# EduGuard GPU Week — Checklist

## Before leaving laptop

- [ ] `python -m pytest tests/federated_training/ -q` passes
- [ ] `experiments/federated/dataset_lock.json` committed
- [ ] `experiments/federated/COMMAND_MANIFEST.md` reviewed
- [ ] Repository copied to GPU machine (git clone or zip)

## On GPU machine — setup

- [ ] Clone/copy repository
- [ ] `python -m venv .venv` and activate
- [ ] `pip install -r requirements-federated.txt`
- [ ] `python experiments/federated/generate_environment_lock.py`
- [ ] `python experiments/federated/check_gpu_environment.py` → READY
- [ ] `python experiments/federated/preflight.py` → READY
- [ ] `python experiments/federated/run_all_experiments.py --dry-run --profile core --no-laptop-mode`

## Core research run

- [ ] `python experiments/federated/run_all_experiments.py --no-laptop-mode --allow-gpu --new-run --profile core`
- [ ] Verify Framework parity (`artifacts/evaluation/framework_parity_gate.json`)
- [ ] Verify FL utility (`artifacts/evaluation/utility_gap_report.json`)

## DP gate

- [ ] `dp_validation` completes (pass or fail recorded honestly)
- [ ] If passed: `artifacts/privacy/dp_bloom_validated_v1.json` exists
- [ ] If failed: `artifacts/privacy/dp_validation_failure.json` — federated DP remains BLOCKED

## Privacy + non-IID

- [ ] SecAgg verification passes
- [ ] Non-IID α=0.1 and α=1.0 complete
- [ ] Privacy attacks attempted (may be NOT_IMPLEMENTED until inference loop wired)

## Export + deployment

- [ ] `export_federated_artifact` produces merged model under `artifacts/federated/global/`
- [ ] `deployment_regression.json` recorded

## Backup (before leaving GPU lab)

- [ ] `experiments/federated/state/run_state.json`
- [ ] `experiments/federated/results/runs/`
- [ ] `artifacts/federated/models/` and `artifacts/federated/global/`
- [ ] `artifacts/privacy/`
- [ ] `artifacts/evaluation/`
- [ ] `experiments/federated/logs/`

## Resume after any shutdown

```bash
python experiments/federated/run_all_experiments.py --resume --no-laptop-mode --allow-gpu
python experiments/federated/run_all_experiments.py --status
```
