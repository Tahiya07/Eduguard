# FedAvg IID: 5 rounds vs 20 rounds

Generated: 2026-09-02T22:09:16.537131+00:00

## Summary

- **Winner (test metrics):** `fedavg_iid_r20`
- **Fair comparison:** False
- **Note:** r20 uses 4x optimizer steps (6240 vs 1560); not budget-matched.

## Test metrics

| Metric | fedavg_iid (5r) | fedavg_iid_r20 | Delta (r20 - 5r) |
|---|---:|---:|---:|
| Accuracy | 0.6657 | 0.7971 | 0.1314 |
| Macro-F1 | 0.6158 | 0.779 | 0.1632 |
| QWK | 0.6678 | 0.8037 | 0.1359 |
| Within-1 | 0.8229 | 0.8829 | 0.06 |
| Severe err | 0.0857 | 0.0743 | -0.0114 |
| ECE | 0.1299 | 0.0619 | -0.068 |

## Training budget

- 5r optimizer steps: 1560
- r20 optimizer steps: 6240
- 5r runtime (s): 1906.2761
- r20 runtime (s): 7526.2142

## Deployment

- Recommended `BLOOM_MODEL_DIR`: `C:\Users\tahiy\PycharmProjects\Eduguard\artifacts\federated\global\qwen_bloom_federated0.5B_fedavg_iid_r20_merged`
- Merge status: SKIPPED
