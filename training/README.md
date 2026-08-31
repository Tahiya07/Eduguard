# EduGuard Training & Federated Research

Training code is **isolated from production runtime**. The FastAPI backend does not import this package.

## Layout

```
training/
  centralized/     merge, evaluate, quantize, DP validation gate
  federated/       FedAvg/FedProx simulation, transport, SecAgg simulator
artifacts/
  federated/       bundles, global adapters, simulation JSON reports
  privacy/         DP validation artifacts (only after Phase 2A passes)
experiments/federated/scripts/   matrix runners, parity reports
```

## Phase 1 — Federated Bloom LoRA (no DP claim)

```powershell
cd C:\Users\tahiy\PycharmProjects\Eduguard
py -3.11 -m pip install -r requirements-training.txt
py -3.11 -m training.federated.simulation --clients 8 --rounds 5 --algorithm fedavg --partition iid
```

Compare with Framework baseline: `experiments/federated/scripts/run_fl_matrix.py`

## Phase 2A — DP validation gate (required before any DP label)

```powershell
py -3.11 -m training.centralized.validate_dp_bloom
```

On success writes: `artifacts/privacy/dp_bloom_validated_v1.json`  
On failure writes: `artifacts/privacy/dp_validation_failed_latest.json` — **do not claim DP**.

## Tests

```powershell
py -3.11 -m pytest tests/federated_training/ -q
```

## Production runtime

Unchanged. Set `BLOOM_MODEL_DIR` to a merged federated checkpoint only after manual evaluation and merge:

```powershell
py -3.11 training/centralized/merge_model.py --lora-dir artifacts/federated/models/qwen_bloom_federated0.5B_fedavg_iid --output-dir models/qwen_bloom_federated_merged0.5B --force
```
