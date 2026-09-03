# Paper results — deployable FL Bloom model

Generated: 2026-09-03T12:45:00.322837+00:00

- Winner: `fedprox_iid_r20` best round `17`
- Selection: max validation accuracy; ties by macro_f1 then quadratic_weighted_kappa
- Model dir: `C:\Users\tahiy\PycharmProjects\Eduguard\artifacts\federated\global\qwen_bloom_federated0.5B_fedprox_iid_r20_best_r17_merged`
- Live eval: False

## Headline test metrics

- Not available yet (run after best-checkpoint merge).

## Insert into manuscript

- Tables: `C:\Users\tahiy\PycharmProjects\Eduguard\artifacts\evaluation\paper/table1_main_metrics.tex`, `table2_algorithm_comparison.tex`, `table3_per_class.tex`
- Figures: `C:\Users\tahiy\PycharmProjects\Eduguard\artifacts\evaluation\paper/fig_*.png` (also `.pdf`)

## Honesty notes

- Selection used **validation** accuracy; reported headline metrics are **held-out test**.
- Best-round ≠ last-round when curves overshoot.
- Do not claim deployable round-N weights without `_best` adapter on disk.
