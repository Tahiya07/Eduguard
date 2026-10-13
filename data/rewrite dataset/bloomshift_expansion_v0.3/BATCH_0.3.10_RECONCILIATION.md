# Batch 0.3.10 reconciliation addendum

- Added `candidate_batch_0.3.10_assistant_curated.jsonl`: 30 draft transformations from 6 source IDs.
- The six source IDs were checked against the previously inventoried candidate files and were not present there.
- Combined raw candidate inventory: 513 records across 202 unique source IDs, assuming no unobserved overlap beyond the checked inventory.
- Every row remains `pending_human_validation` and `training_eligible=false`.

These are inventory counts, not validated-example counts. Do not use this batch for training. Review source eligibility and provenance, semantic/content preservation, target-level cognitive demand, technical accuracy, and duplicates; then obtain independent human annotation and adjudication.
