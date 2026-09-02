# EduGuard Federated Research — Command Manifest

All commands run from repository root after `pip install -r requirements-federated.txt`.

## Environment

```bash
python experiments/federated/generate_environment_lock.py
python experiments/federated/generate_dataset_lock.py
```

## Verification

```bash
python experiments/federated/check_gpu_environment.py
python experiments/federated/preflight.py
```

Laptop (no CUDA expected to fail GPU check):

```bash
python experiments/federated/preflight.py --allow-no-gpu
```

## Dry run

```bash
python experiments/federated/run_all_experiments.py --dry-run --profile core --no-laptop-mode
```

## Core GPU run (new)

```bash
python experiments/federated/run_all_experiments.py --no-laptop-mode --allow-gpu --new-run --profile core
```

## Resume after shutdown

```bash
python experiments/federated/run_all_experiments.py --resume --no-laptop-mode --allow-gpu
```

## Status

```bash
python experiments/federated/run_all_experiments.py --status
```

## Individual experiments

```bash
python experiments/federated/run_all_experiments.py --experiment fedavg_iid --no-laptop-mode --allow-gpu
python experiments/federated/run_all_experiments.py --experiment dp_validation --no-laptop-mode --allow-gpu
```

## Compare 5-round vs 20-round FedAvg (deployment pick)

```bash
python experiments/federated/scripts/compare_fedavg_5r_vs_r20.py
# PowerShell helper (merge + deployment hints):
powershell -File scripts/deploy_federated_winner.ps1
```

Outputs: `artifacts/evaluation/fedavg_5r_vs_r20_comparison.md`, `deployment_recommendation.json`

## FedProx IID 20 rounds

```bash
python experiments/federated/run_all_experiments.py --experiment fedprox_iid_r20 --no-laptop-mode --allow-gpu --profile extended

python -m training.federated.simulation --clients 8 --rounds 20 --local-epochs 3 --algorithm fedprox --prox-mu 0.01 --partition iid --seed 42 --experiment-tag fedprox_iid_r20 --global-adapter artifacts/federated/models/qwen_bloom_federated0.5B_fedprox_iid_r20 --results-json artifacts/federated/results/federated_lora_fedprox_iid_r20.json --fresh
```

## DP validation (standalone)

```bash
python -m training.centralized.validate_dp_bloom --output artifacts/privacy/dp_bloom_validated_v1.json
```

## Export (after fedavg_iid)

```bash
python training/centralized/merge_model.py --model-size 0.5b --lora-dir artifacts/federated/models/qwen_bloom_federated0.5B_fedavg_iid --output-dir artifacts/federated/global/qwen_bloom_federated0.5B_fedavg_iid_merged --force
```

## Deployment test

```bash
python experiments/federated/scripts/deployment_regression.py
```

## Extended profile (optional, after core complete)

```bash
python experiments/federated/run_all_experiments.py --resume --no-laptop-mode --allow-gpu --profile extended
```

## Time budget hint (144 hours = 1 week)

```bash
python experiments/federated/run_all_experiments.py --resume --no-laptop-mode --allow-gpu --profile all --max-hours 144
```

## Unit tests (laptop)

```bash
python -m pytest tests/federated_training/ -q
```
