# Pending expansion candidates (combined batches 0.3.1–0.3.5)

This file consolidates the five reviewed-for-structure candidate batches from `bloomshift_expansion_v0.3` into one pool alongside the existing BloomShift candidate dataset.

## Contents

- Combined candidate records: 136
- Batch sizes: 0.3.1 = 25; 0.3.2 = 27; 0.3.3 = 24; 0.3.4 = 30; 0.3.5 = 30
- Unique candidate IDs: 136
- Unique source IDs within this combined pool: 136
- Records pending human validation: 136
- Records marked training-eligible: 0

## Target-level counts

| Target Bloom level | Candidates |
|---|---:|
| Remember | 18 |
| Understand | 27 |
| Apply | 18 |
| Analyze | 29 |
| Evaluate | 18 |
| Create | 26 |
| **Total** | **136** |

## How this relates to train/validation/test

The existing `train.json`, `validation.json`, and `test.json` files are intentionally unchanged. This consolidated file is a **pending-review pool**, not an automatically appended training split. Every candidate currently has `training_eligible=false`; adding unvalidated rewrites directly to training could contaminate the experiments and undermine semantic-fidelity/Bloom-alignment evaluation.

Before any candidate is promoted into a split:
1. Review source-meaning preservation and whether the target cognitive operation is actually achieved.
2. Resolve or reject candidates that rely on missing figures, expressions, data, cases, or course-specific context.
3. Deduplicate against existing data using normalized text and source/group relationships, not only IDs.
4. Assign source groups to splits before expanding records, to prevent group leakage.
5. Record reviewer decisions and only then build a versioned expanded train/validation/test set.

The counts above are structural checks only. They are not evidence of human validation, semantic correctness, or Bloom-level validity.
