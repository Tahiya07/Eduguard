# Role separation (architectural) and PrivacyGuard empirical evaluation

| Item | Public | Protected | Bloom tools / rate | Notes |
| --- | --- | --- | --- | --- |
| Student | Yes | No (403 if scope=protected) | No exam classification tools | Query+output screening |
| Teacher | Yes | Authorized protected workflows (code) | Yes | Output copy screening of protected wording |
| PrivacyGuard attacks blocked | 104/104 | rate=1.0 | — | privacy_guard_eval.json |
| Benign student allow | 15 prompts | rate=0.0 | — | Utility collapse under current suite |
| Ablation attacks | 1344 | full_hybrid block=1.0 | benign allow=0.0 | privacy_benchmark_baselines.json |
