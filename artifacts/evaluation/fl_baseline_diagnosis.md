# FL Baseline Diagnosis

**DP status:** NOT VALIDATED

## Experiment comparison

| Experiment | Status | Accuracy | Macro-F1 | QWK | Within-1 | Severe err | ECE | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| framework_reference | UNKNOWN | 0.5029 | 0.4262 | 0.4454 | 0.6943 | 0.1743 | 0.1698 | None |
| fedavg_iid | IMMUTABLE_BASELINE | 0.5686 | 0.4962 | 0.6124 | 0.7571 | 0.1286 | 0.1682 | 1659.72 |
| fedavg_iid_r20 | NOT_EXECUTED | None | None | None | None | None | None | None |
| fedavg_iid_localepoch1 | NOT_EXECUTED | None | None | None | None | None | None | None |
| fedprox_iid | NOT_EXECUTED | None | None | None | None | None | None | None |

## Best observed configuration

- Configuration: fedavg_iid
- Reason: Highest test accuracy among executed targeted experiments (not claimed optimal).
- Fair comparison: False
- Test set used for selection: False

## Immutable baseline

- fedavg_iid accuracy: 0.5686
- Lock file: `C:\Users\tahiy\PycharmProjects\Eduguard\artifacts\evaluation\fedavg_iid_baseline_lock.json`
