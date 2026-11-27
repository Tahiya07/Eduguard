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

## Live federated rounds on teacher-confirmed labels

The deployed classifier (FedProx r20, merged) is frozen at runtime; it never trains on requests.
Teachers can confirm or correct a predicted Bloom level in Teacher Studio ("Confirm level").
The backend appends that label to `data/federated_local/confirmed_labels.jsonl` on the same
machine (`backend/label_store.py`; disable with `FL_COLLECT_LABELS=false`). Notes and session ids
are not stored with it. Rounds are run deliberately, e.g. once per term:

```powershell
# coordinator, once: shared round-0 adapter on top of the deployed classifier; copy the folder to every site
.venv\Scripts\python.exe -m training.federated.local_round init

# each site: how many confirmed labels it has (needs >= 30, at least two levels)
.venv\Scripts\python.exe -m training.federated.local_round status
# each site: one local FedProx round -> data/federated_local/rounds/round_01/<site>_round01.json
.venv\Scripts\python.exe -m training.federated.local_round train --site-id school_a --round 1
#   send ONLY the bundle .json (adapter weights + SHA-256); site_labels.csv stays on the site

# coordinator: aggregate the received bundles (>= 2 sites) into the global adapter
.venv\Scripts\python.exe -m training.federated.local_round aggregate --round 1 --bundles school_a_round01.json school_b_round01.json
# coordinator: merge, evaluate on the Figshare validation split, accept only if accuracy does not drop > 1 point
.venv\Scripts\python.exe -m training.federated.local_round promote
#   accepted -> each site sets BLOOM_MODEL_DIR to the printed folder; next round repeats train/aggregate/promote
```

Privacy scope: raw questions never leave a site. Bundles are integrity-protected but not
encrypted, and there is no differential-privacy guarantee (DP validation failed; see Phase 2A).

## Production runtime

Unchanged. Set `BLOOM_MODEL_DIR` to a merged federated checkpoint only after manual evaluation and merge:

```powershell
py -3.11 training/centralized/merge_model.py --lora-dir artifacts/federated/models/qwen_bloom_federated0.5B_fedavg_iid --output-dir models/qwen_bloom_federated_merged0.5B --force
```
