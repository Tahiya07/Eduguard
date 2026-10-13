# Batch 0.3.9 reconciliation addendum

- New file: `candidate_batch_0.3.9_assistant_curated.jsonl`
- Added: 50 draft transformations from 10 source IDs (two sources each from original Remember, Apply, Analyze, Evaluate, and Create levels).
- Target counts: Remember 8; Understand 10; Apply 8; Analyze 8; Evaluate 8; Create 8.
- All rows remain `pending_human_validation`; all have `training_eligible=false`.
- Combined candidate inventory: 483 raw records across 196 unique source IDs, assuming no overlap between the ten new source IDs and the previously inventoried IDs.

These counts describe files, not valid or training-ready examples. This batch uses topic- and constraint-conditioned templates and can be too broad or fail to preserve the source's intended assessment purpose. Do not train on it. Review provenance and source eligibility, semantic preservation, cognitive operation, constraints, and duplicates; then obtain independent annotation and adjudication before release or splitting.