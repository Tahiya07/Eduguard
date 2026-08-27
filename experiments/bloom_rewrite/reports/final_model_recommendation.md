# Final model recommendation

**No model is selected.** Training and evaluation have not been run.

Do **not** choose 1.5B because it is larger. Do **not** choose 0.5B because it is smaller.

Decision rule (to be applied only after paired measurements):

1. Check whether each model reaches an acceptable target-rewrite quality threshold on the held-out test set (fixed classifier + validators + human ratings when available).
2. Exclude models that fail that threshold.
3. Among acceptable models, prefer the smaller/faster model if quality is statistically and practically comparable (McNemar + bootstrap CI on paired items).
4. Select 1.5B only if it provides a substantial quality advantage **and** still satisfies CPU/offline/~1 GB measured-memory constraints.
5. Select 0.5B if quality is comparable with significantly lower resource cost.

Placeholder comparison table (all cells pending):

| Metric | Qwen 0.5B | Qwen 1.5B |
|---|---|---|
| Target accuracy | pending | pending |
| Macro F1 | pending | pending |
| C1–C6 accuracy | pending | pending |
| Topic preservation | pending | pending |
| Question validity | pending | pending |
| Trivial rewrite rate | pending | pending |
| Meta-output rate | pending | pending |
| Generation latency | pending | pending |
| GGUF size | pending | pending |
| Peak memory / USS | pending | pending |
| CPU feasibility | pending | pending |
