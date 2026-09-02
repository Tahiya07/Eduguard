# FL Baseline Diagnosis

**DP status:** NOT VALIDATED

## Experiment comparison

| Experiment | Status | Accuracy | Macro-F1 | QWK | Within-1 | Severe err | ECE | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| framework_reference | DOCUMENTED_ONLY | 0.503 | None | None | None | None | None | None |
| fedavg_iid | EXECUTED | 0.6657 | 0.6158 | 0.6678 | 0.8229 | 0.0857 | 0.1299 | 1906.2761 |
| fedavg_iid_r20 | EXECUTED | 0.7971 | 0.779 | 0.8037 | 0.8829 | 0.0743 | 0.0619 | 7526.2142 |
| fedavg_iid_localepoch1 | EXECUTED | 0.3629 | 0.1376 | 0.1947 | 0.6514 | 0.24 | 0.098 | 1192.5343 |
| fedprox_iid | EXECUTED | 0.6829 | 0.6382 | 0.6954 | 0.84 | 0.0914 | 0.12 | 2984.7894 |

## Best observed configuration

- Configuration: fedavg_iid_r20
- Reason: Highest test accuracy among executed targeted experiments (not claimed optimal).
- Fair comparison: False
- Test set used for selection: False

## Immutable baseline

- fedavg_iid accuracy: 0.5686
- Lock file: `D:\Eduguard\artifacts\evaluation\fedavg_iid_baseline_lock.json`
