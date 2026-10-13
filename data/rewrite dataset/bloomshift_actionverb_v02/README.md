# BloomShift action-verb revision v0.2.0

This directory mirrors the v0.2.0 candidate from the standalone BloomShift repository for EduGuard multitask training.

- Train / validation / test: 691 / 138 / 138 (967 rows total).
- 258 unique source IDs.
- Target rewrites begin with an action verb aligned to the target Bloom level.
- Source groups are disjoint across splits.
- The supervisor annotation worksheet contains all 276 held-out validation/test examples with blank annotation fields.

This is a synthetic silver candidate, not a human-certified gold dataset. Fresh human review is required before claiming semantic preservation, context preservation, Bloom alignment, or pedagogical validity. The original v0.1.0 candidate remains unchanged.

Source of record: https://github.com/Tahiya07/BloomShift/tree/fix/action-verb-rewrites-expand-splits/dataset/v0.2.0
