# EduGuard BloomShift v4

A source-anchored training dataset for transforming an educational question from one Revised Bloom cognitive level to another while preserving the original question's substantive content.

## Design
- Built from EduGuard's 893 locked source questions; current v3 target rewrites are not reused.
- Only transformable question families are admitted: programming, calculation, comparison, design, evaluation, and process.
- Same-level identity copies are excluded.
- Training uses two surface variants per selected source-target pair; validation and test use one.
- Target labels are balanced across Remember, Understand, Apply, Analyze, Evaluate, and Create.
- Train/validation/test remain source-group disjoint.
- The transformation changes the cognitive operation, not the underlying topic, named entities, quantities, or stated requirements.

## Important status
This is a **silver training candidate**, not a human-certified gold benchmark. Validation/test should receive independent educator review before the dataset is used as a publication-quality human-validated benchmark.

## Files
- train.json
- validation.json
- test.json
- audit.json

The previous dataset under `data/rewrite dataset/train.json`, `validation.json`, and `test.json` is intentionally untouched.
