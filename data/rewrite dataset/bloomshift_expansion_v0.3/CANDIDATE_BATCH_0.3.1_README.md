# BloomShift expansion candidate batch 0.3.1

This file contains 25 manually drafted rewrite candidates based on source questions in `source_candidates_for_rewrite.csv`.

## Action-verb diversity

The candidate JSONL now stores a level-specific action-verb reference bank for each record, plus the selected opening verb and a diversity-policy field. The current batch uses 18 distinct opening verbs across its five represented target levels (Remember, Understand, Apply, Analyze, Evaluate). It has no Create-target record yet; do not treat the reference bank as evidence that all six levels are represented in this batch.

Action verbs should vary within each Bloom level where the actual cognitive operation permits. A verb is not proof of Bloom alignment: review the whole task, context, and expected reasoning.

## Validation status

- Human validation: pending for every record.
- Training eligibility: false for every record.
- Some transformations are conditional and carry explicit semantic/context risks.
- This batch is not a replacement for the existing train/validation/test sets and is not the complete expansion of the 2,007-source pool.
- Before training, review source fidelity, target Bloom alignment, feasibility, duplicates, and grouped splitting.

Every record retains the original source question and source group identifier. The opening verb is included for auditing; it does not itself prove semantic preservation or cognitive-level alignment.
