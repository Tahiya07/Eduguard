# EduGuard — One-Week GPU Execution Plan

Dependency-ordered stages for the university GPU machine. Do not skip scientific gates.

## Stage A — Environment + parity (Day 1)

| Item | Experiment ID | Output |
|------|---------------|--------|
| Preflight + env lock | `env_validation`, `generate_environment_lock` | `preflight_report.json`, `environment_lock.json` |
| Unit tests | `unit_tests_federated` | pass |
| FL smoke | `fl_smoke_fedavg_iid` | smoke JSON |
| **FedAvg IID** | `fedavg_iid` | `federated_lora_fedavg_iid.json` |
| Parity gate | `framework_parity_gate` | `framework_parity_gate.json` |

**Stop if:** parity gap > 5% absolute accuracy — diagnose before Stage B.

**Resume point:** `fedavg_iid` round checkpoint at `artifacts/federated/runs/fedavg_iid/round_checkpoint.json`

## Stage B — FL baseline matrix (Days 1–3)

| Experiment | ID |
|------------|-----|
| FedProx IID | `fedprox_iid` |
| FedAvg non-IID α=0.5 | `fedavg_noniid_a05` |
| FedProx non-IID α=0.5 | `fedprox_noniid_a05` |
| Utility gap | `utility_gap_analysis` |

## Stage C — DP validation (Day 3–4)

| Experiment | ID | Gate |
|------------|-----|------|
| Centralized DP | `dp_validation` | Must pass for federated DP |
| Federated DP | `federated_dp` | **BLOCKED** until lock exists |

**If DP fails:** continue independent non-DP experiments (SecAgg, non-IID α=0.1/1.0, attacks).

## Stage D — Privacy mechanisms (Day 4)

| Experiment | ID |
|------------|-----|
| SecAgg verify | `secagg_verification` |
| Non-IID α=0.1 | `fedavg_noniid_a01` |
| Non-IID α=1.0 | `fedavg_noniid_a10` |

## Stage E — Attacks (Day 5)

| Experiment | ID | Note |
|------------|-----|------|
| Privacy attacks | `privacy_attacks` | NOT_IMPLEMENTED until GPU models exist |

## Stage F — Export + deployment (Day 5–6)

| Experiment | ID |
|------------|-----|
| Merge adapter | `export_federated_artifact` |
| Deployment regression | `deployment_regression` |
| Final manifest | `final_research_manifest` |

## Stage H — Optional (remaining time only)

Run with `--profile extended` after all core experiments complete:

```bash
python experiments/federated/run_all_experiments.py --resume --no-laptop-mode --allow-gpu --profile extended
```

## Recommended stopping points (safe shutdown)

- After any experiment shows `COMPLETE` in `--status`
- After a FL simulation round completes (round checkpoint written)
- Never during an active client training subprocess if avoidable

## Multitask

**DEFERRED** unless core experiments finish early. Do not sacrifice Stage B/C for multitask.
