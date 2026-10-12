# BloomShift expansion candidate batch 0.3.3

This batch adds 24 manually drafted candidate rewrites selected from `source_candidates_for_rewrite.csv`.

## Coverage

Four candidate records target each of the six revised Bloom taxonomy levels:

- Remember: 4
- Understand: 4
- Apply: 4
- Analyze: 4
- Evaluate: 4
- Create: 4

The records use 23 distinct opening verbs. Each row stores the source question, source group ID, target level, opening verb, level-specific verb bank, preserved elements, preliminary feasibility, semantic risks, and review status.

## Validation status

- Human validation remains pending for every record.
- `training_eligible` is `false` for every record.
- Some sources refer to missing figures, data, recordings, case details, or assumptions. Those limitations are called out in `semantic_risks`; candidates requiring absent context must not be treated as independently answerable.
- A source question's original Bloom label is not itself proof that the rewritten version matches the target level.
- The Create examples are grounded in sources originally classified as Create, but still need review for added constraints and task fidelity.
- This is an additional candidate batch, not a validated replacement for the existing train/validation/test datasets and not a complete rewrite of the 2,007-question source pool.

Before training, review meaning preservation, intended cognitive demand, source context, action-verb appropriateness, near duplicates, and source-group leakage. Keep all rewrites from a source group in the same split.
