# Framework Parity Audit

**Exact parity established:** False
**DP status:** NOT VALIDATED

## Metrics

- Framework test accuracy: 0.503
- EduGuard test accuracy: 0.5686
- Absolute difference (EduGuard − Framework): 0.0656
- Within 5pp threshold: False

## Configuration comparisons

| Field | Framework | EduGuard | Same | Potential impact |
|---|---|---|:---:|---|
| aggregation | FedAvg weighted by client sample count | fedavg_sample_weighted | False | configuration mismatch may shift convergence or parity |
| base_model | Qwen/Qwen2.5-0.5B-Instruct | Qwen/Qwen2.5-0.5B-Instruct | True | none (matched) |
| batch_size | 2 | 2 | True | none (matched) |
| clients | 8 | 8 | True | none (matched) |
| dataset_hashes | None | {'data/figshare_bloom_v1_train.csv': 'd5cd8c28c4b6cbfc6b6803830c64185df6fb591466d1abf145d84f6b24f9fc4d', 'data/figshare_bloom_v1_val.csv': '14cd9304434d97d9add2ed13a30ed9bf3adb09373b4f6d3f03c6efd141fe4ef0', 'data/figshare_bloom_v1_test.csv': '7ee161c5f006a8e939df97d7528a1ec64e877701ac74c56e1afd63d0a262eb97'} | False | configuration mismatch may shift convergence or parity |
| gradient_accumulation | 8 | 8 | True | none (matched) |
| label_smoothing | None | 0.05 | False | configuration mismatch may shift convergence or parity |
| learning_rate | 0.0001 | 0.0001 | True | none (matched) |
| local_epochs | 3.0 | 3.0 | True | none (matched) |
| lora_alpha | 64 | 64 | True | none (matched) |
| lora_dropout | 0.1 | 0.1 | True | none (matched) |
| lora_r | 32 | 32 | True | none (matched) |
| modules_to_save | ['score'] | ['score'] | True | none (matched) |
| partition | iid | iid | True | none (matched) |
| rounds | 5 | 5 | True | none (matched) |
| seed | 42 | 42 | True | none (matched) |
| warmup_ratio | None | 0.1 | False | configuration mismatch may shift convergence or parity |
| weight_decay | None | 0.01 | False | configuration mismatch may shift convergence or parity |

## Partition

- EduGuard partition artifact: `D:\Eduguard\artifacts\federated\runs\fedavg_iid\partition.json`
- Framework partition verified: False
- Note: Framework partition artifact not present in this repo; IID procedure documented only.

## Conclusion

EduGuard FedAvg+IID baseline (56.86%) exceeds historical Framework (~50.3%) by 0.0656 absolute accuracy. Exact parity is NOT established without Framework partition/config artifacts. Diagnose before claiming migration fidelity.
