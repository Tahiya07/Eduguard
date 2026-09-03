# Paper results — deployable FL Bloom model

Generated: 2026-09-03T19:26:57.200867+00:00

- Winner: `fedprox_iid_r20` best round `20`
- Selection: max validation accuracy; ties by macro_f1 then quadratic_weighted_kappa
- Model dir: `D:\Eduguard\artifacts\federated\global\qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged`
- Live eval: True

## Headline test metrics

- Accuracy: 0.8057
- Macro-F1: 0.7982
- QWK: 0.8493
- Within-1: 0.9086
- Severe err: 0.0429
- ECE: 0.0537

## Insert into manuscript

- Tables: `artifacts\evaluation\paper/table1_main_metrics.tex`, `table2_algorithm_comparison.tex`, `table3_per_class.tex`
- Figures: `artifacts\evaluation\paper/fig_*.png` (also `.pdf`)

## Honesty notes

- Selection used **validation** accuracy; reported headline metrics are **held-out test**.
- Best-round ≠ last-round when curves overshoot.
- Do not claim deployable round-N weights without `_best` adapter on disk.
