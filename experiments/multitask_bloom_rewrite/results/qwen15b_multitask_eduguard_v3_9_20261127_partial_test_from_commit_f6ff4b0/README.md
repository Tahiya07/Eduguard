# Partial test-split run (not validation)

These files were committed in `f6ff4b0` inside the v3.9 validation results folder while the
one-time **test** evaluation was still running (550 of 645 test items: 351 rewrites, 147 QA,
52 summaries). They are kept here for provenance only.

- Complete test results: `../qwen15b_multitask_eduguard_v3_9_20261127_test/`
- Validation metrics (correct, unchanged): `../qwen15b_multitask_eduguard_v3_9_20261127/`
- The per-item bf16 **validation** predictions were overwritten by this run and are not retained;
  the 341 validation rewrite texts survive in
  `../qwen15b_multitask_eduguard_v3_9_20261127/human_eval_predictions.jsonl`.
