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
