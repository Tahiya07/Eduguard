# EduGuard BloomShift — Expanded Candidate v3

The canonical split files contain the expanded dataset transferred from `fix/arxiv-multitask-training`.

| Split | Records |
|---|---:|
| Train | 1075 |
| Validation | 212 |
| Test | 135 |
| **Total** | **1422** |

The three split files are group-disjoint; see `expanded_split_audit_v3.json` for the computed overlap checks, target-level counts, and review-status counts.

## Human-review status

The source split files retain their per-record review metadata. Only records explicitly marked `review_status=human_validated` and `training_eligible=true` should be treated as human-validated and eligible for training. Missing or unmarked review status is **not** equivalent to validation.

The prior `audit.json` documents the earlier 888-record candidate and its quality screens; it is retained as historical audit information, not as the count manifest for this expanded version. The older `expanded_split_audit_v2.json` also describes an earlier 1,024-record state and is retained for history. Use `expanded_split_audit_v3.json` for the current counts.

## Research cautions

- Run the dataset validator and inspect review-status filtering before training or evaluation.
- Because source groups were reassigned and the test set changed, previous metrics are not directly comparable to results on this version.
- Automated source/content and language checks are screening aids; they do not independently establish semantic preservation, Bloom alignment, or pedagogical validity.
