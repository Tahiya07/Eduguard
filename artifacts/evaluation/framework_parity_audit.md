# Framework Parity Audit

**Exact parity established:** False
**DP status:** NOT VALIDATED

## Metrics

- Framework test accuracy: 0.5029
- EduGuard test accuracy: 0.5686
- Absolute difference (EduGuard − Framework): 0.0657
- Within 5pp threshold: False

## Configuration comparisons

| Field | Framework | EduGuard | Same | Potential impact |
|---|---|---|:---:|---|
| aggregation | fedavg | None | False | configuration mismatch may shift convergence or parity |
| base_model | Qwen/Qwen2.5-0.5B-Instruct | Qwen/Qwen2.5-0.5B-Instruct | True | none (matched) |
| batch_size | 2 | 2 | True | none (matched) |
| clients | 8 | 8 | True | none (matched) |
| dataset_hashes | None | None | True | none (matched) |
| gradient_accumulation | 8 | 8 | True | none (matched) |
| label_smoothing | None | None | True | none (matched) |
| learning_rate | 0.0001 | 0.0001 | True | none (matched) |
| local_epochs | 3.0 | 3.0 | True | none (matched) |
| lora_alpha | 64 | None | False | configuration mismatch may shift convergence or parity |
| lora_dropout | 0.1 | None | False | configuration mismatch may shift convergence or parity |
| lora_r | 32 | 32 | True | none (matched) |
| modules_to_save | ['score'] | ['score'] | True | none (matched) |
| partition | iid | iid | True | none (matched) |
| rounds | 5 | 5 | True | none (matched) |
| seed | 42 | 42 | True | none (matched) |
| warmup_ratio | None | None | True | none (matched) |
| weight_decay | None | None | True | none (matched) |

## Partition

- EduGuard partition artifact: `C:\Users\tahiy\PycharmProjects\Eduguard\artifacts\federated\runs\fedavg_iid\partition.json`
- Framework partition verified: False
- Note: Framework partition artifact not present in this repo; IID procedure documented only.

## Conclusion

EduGuard FedAvg+IID baseline (56.86%) exceeds historical Framework (~50.3%) by 0.0657 absolute accuracy. Exact parity is NOT established without Framework partition/config artifacts. Diagnose before claiming migration fidelity.
