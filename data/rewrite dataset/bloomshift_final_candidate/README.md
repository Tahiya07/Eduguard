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

The source split files retain per-record review metadata. Only records explicitly marked `review_status=human_validated` and `training_eligible=true` are eligible for inclusion. Missing or unmarked review status is **not** equivalent to validation.

The final multitask builder also requires distinct `reviewer_1.csv` and `reviewer_2.csv` files with complete ratings for every eligible validation/test row. Metadata flags alone do not prove independent annotation. If these files are identical, empty, or incomplete, the builder intentionally stops instead of silently treating the held-out labels as human-validated.

The prior `audit.json` documents the earlier 888-record candidate and its quality screens; it is retained as historical audit information, not as the count manifest for this expanded version. The older `expanded_split_audit_v2.json` also describes an earlier 1,024-record state and is retained for history. Use `expanded_split_audit_v3.json` for the current counts.

## Research cautions

- Run `python experiments/multitask_bloom_rewrite/scripts/audit_bloomshift_dataset.py` before training. The audit writes an eligibility report and a CSV queue for unreviewed rows.
- Complete and preserve independent reviewer ratings before rebuilding the final multitask dataset; the builder now fails closed if held-out annotation evidence is not auditable.
- Because source groups were reassigned and the test set changed, previous metrics are not directly comparable to results on this version.
- Automated source/content and language checks are screening aids; they do not independently establish semantic preservation, Bloom alignment, or pedagogical validity.
