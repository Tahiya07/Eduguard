# BloomShift expansion candidate batch 0.3.7 — assistant-curated pilot

## Contents
- 30 records across 10 source questions from `source_candidates_for_rewrite.csv`.
- 21 draft rewrites marked feasible for review.
- 9 explicit not-feasible decisions where the source lacks context or parameters needed for a grounded transformation.
- Source groups are retained; all records remain `training_eligible=false`.

## Why this batch exists
This is a small, manually reasoned pilot intended to compare direct curation against local-model generations. It deliberately does not force every source into every target level. Rewrite drafts aim to preserve the semantic core while stating when an original demand was removed or reframed to fit the target cognitive operation.

## Limitations
- Source rows are still marked `pending` and `not_reviewed`; this batch does not approve source eligibility.
- The rewrites were authored as drafts and have not been independently human-validated.
- `feasible` means a plausible draft was written, not that semantic equivalence or Bloom alignment has been established.
- Not-feasible records are retained for auditability and must not be used as training targets.
- The source pool's exact-match and near-duplicate screening is not proof of global uniqueness. Reconcile all inventories and provenance/licensing before release.
- Review technical claims and domain-specific assumptions, especially put-call parity, monetary-policy mechanisms, FISH workflow, and Pondy's model terminology.
- Split by `source_group_id`; all rewrites derived from the same source must stay in a single partition.

## Required next steps
1. Review source eligibility and source quality.
2. Independently annotate meaning preservation, target cognitive operation, answerability, and factual/domain accuracy.
3. Resolve or reject candidates with reviewer disagreement.
4. Deduplicate against all existing candidate files and historical datasets.
5. Build source-group-disjoint train/validation/test partitions only after approval.
6. Recompute task/level counts and rerun structural and semantic audits before training.
