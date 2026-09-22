# Proof ④ — Protected-content evaluation

- status: `FROM_ARTIFACT`
- source: `artifacts/model train results/privacy_guard_eval.json`
- paper_ready (artifact present): `True`

## Headline rates

| Metric | Value | Count |
| --- | ---: | ---: |
| attack_block_rate | 1.0 | 104/104 |
| leakage_rate | 0.0 | 0/104 |
| false_block_rate | 1.0 | 15/15 |
| benign_allow_rate | 0.0 | 0/15 |

## Limitations

- Empirical screening only — not a cryptographic security proof.
- Formal differential privacy is out of scope for this proof.
- High false_block_rate indicates over-blocking / utility collapse.

## Ablation

```json
{
  "status": "OK",
  "source": "artifacts/model train results/privacy_benchmark_baselines.json",
  "n_cases": 1416,
  "n_attack_cases": 1344,
  "n_benign_cases": 64,
  "baselines": [
    {
      "name": "no_guard",
      "attack_block_rate": 0.0,
      "leakage_rate": 1.0,
      "false_block_rate": 0.0,
      "benign_allow_rate": 1.0
    },
    {
      "name": "role_only_no_output_guard",
      "attack_block_rate": 0.0,
      "leakage_rate": 1.0,
      "false_block_rate": 0.0,
      "benign_allow_rate": 1.0
    },
    {
      "name": "regex_only",
      "attack_block_rate": 0.5238095238095238,
      "leakage_rate": 0.47619047619047616,
      "false_block_rate": 0.0,
      "benign_allow_rate": 1.0
    },
    {
      "name": "federated_dp_only",
      "attack_block_rate": 1.0,
      "leakage_rate": 0.0,
      "false_block_rate": 1.0,
      "benign_allow_rate": 0.0
    },
    {
      "name": "learned_plus_overlap",
      "attack_block_rate": 1.0,
      "leakage_rate": 0.0,
      "false_block_rate": 1.0,
      "benign_allow_rate": 0.0
    },
    {
      "name": "full_hybrid_guard",
      "attack_block_rate": 1.0,
      "leakage_rate": 0.0,
      "false_block_rate": 1.0,
      "benign_allow_rate": 0.0
    }
  ]
}
```
