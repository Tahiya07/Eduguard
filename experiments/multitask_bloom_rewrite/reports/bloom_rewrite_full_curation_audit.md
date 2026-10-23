# Bloom rewrite semantic curation audit

Date: 2026-10-23

## Scope and edits

Reviewed 1422 prepared records against their source questions and target labels across train, validation, and test. The task-specific JSONL files contain 919 rewritten rows compared with the preserved pre-curation backup; 503 rows were retained unchanged. IDs, source questions, target Bloom labels, schema, row counts, and split boundaries were preserved. Derived SFT completions were synchronized.

| Split | Before | After | Corrected | Retained |
|---|---:|---:|---:|---:|
| Train | 1075 | 1075 | 692 | 383 |
| Validation | 212 | 212 | 148 | 64 |
| Test | 135 | 135 | 79 | 56 |

The machine-readable audit lists every corrected and retained ID, the old and new rewrites, and 26 source-context uncertainties. The existing split backup directory was preserved.

## Integrity and diagnostics

- Canonical source mappings: 0 errors.
- Source/group split leakage: 0 IDs.
- Duplicate IDs / exact duplicate rewrite strings: 0 / 0.
- Derived SFT target mismatches: 0.
- Rule-guided diagnostic scan: 975 passed heuristic checks; 447 rows received at least one heuristic flag. Flags include false positives from lexical topic-overlap and imperative-format checks and are not semantic error counts.

## Tests and limitations

- Targeted curation tests: 4 passed.
- Full suite: 50 tests; 12 failures, 3 errors, 6 skipped using a workspace-local temp directory. Failures are prompt-contract expectations for the literal “actual student task”; errors are one missing `EXPECTED_TEST_TOTAL` import and two references to missing `qwen05b_multitask.json`.
- Dataset validator: structural counts and hashes pass; it exits 1 because `transformers` is unavailable for tokenizer QC.
- Review was single-agent and rule-guided; no independent second annotator or model-based verification was performed. Ambiguous source prompts are itemized in the JSON audit.

## Final pre-training review (2026-10-23)

All 26 previously itemized uncertain records were re-opened against their canonical source rows and current prepared examples. One Understand record (`bloomshift_final_train_0452`) had a clear monetary-policy source and source-specific explanation; its ambiguity flag is resolved without changing the rewrite. One Apply record (`bloomshift_expansion_exp038_figshare_row_248_apply`) was corrected to preserve the source request for three advantages of zymogen activation, applied to the named digestive-protease or blood-clotting cascade; its derived SFT completion and manifest/QC hashes were updated. The remaining 24 prompts omit a referent, problem, supplied specifications/output, or another necessary case detail. Their most defensible conditional rewrites remain, and no missing facts were invented.

Counts remain 1,075 / 212 / 135. The review is source-grounded and single-annotator; it is not independent model or dual-annotator validation. See `final_pretraining_review` in the JSON audit for the complete unresolved ID list and dispositions.

## Final validator and test-suite results

After the one row correction, `validate_separate_dataset` passed all counts, hashes, source mappings, and split-isolation checks; all 1,422 Bloom SFT targets match their rewrite text. The repository validator passed structural/count/hash checks but exited 1 at tokenizer QC because `transformers` is unavailable. The full test suite ran 50 tests: 12 duplicated exact-string prompt-contract failures, 3 errors (missing `EXPECTED_TEST_TOTAL` export once; missing legacy 0.5B config twice), and 6 skips for absent legacy merged-root fixtures. Four task-separated loader tests and four curation tests passed. The active separate 1.5B config parses and references the current task-separated data. No tests, application code, or config were modified. See `pretraining_readiness_20261023.md` for the final readiness decision.

## Reassessment of remaining source-context flags (2026-10-23)

All 24 outstanding entries were compared to canonical source records. Nine are sufficiently specified as open-ended tasks because the rewrite explicitly lets the learner choose the omitted scenario/example, or because the Understand operation does not depend on a missing instance; those rows were retained unchanged. Fifteen remain unresolved: five unidentified “this” referents, two unspecified term pairs, three missing specifications, two missing genetic-algorithm problem instances for instance-dependent operations, one missing case/outcomes, and two missing nested-loop output patterns. Item-level decisions are in `final_pretraining_review` in the JSON audit.

The project `.venv` has Transformers 5.17.0 and tokenizers 0.23.2, outside the training pins (4.41.0 and 0.19.1); the selected 1.5B tokenizer files are not cached. The validator passed structural/count/hash checks but could not run tokenizer QC offline. The active separate 1.5B config validation passes. The full suite passes 60/60 after updating brittle/obsolete assertions to the current prompt and task-separated interface. No Bloom rewrite rows changed in this reassessment.

Count clarification: 10 of the original uncertainties were resolved without changing their rewrite (one in the previous pass and nine in this pass); one other uncertainty was resolved by the earlier zymogen rewrite correction. The 15 remaining uncertainties are distributed 10 train / 3 validation / 2 test.
