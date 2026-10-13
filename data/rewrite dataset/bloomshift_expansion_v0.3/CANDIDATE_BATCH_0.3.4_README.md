# BloomShift expansion candidate batch 0.3.4

This batch adds 30 draft rewrite candidates selected from `source_candidates_for_rewrite.csv`.

## Coverage

Five candidates target each revised Bloom taxonomy level:

- Remember: 5
- Understand: 5
- Apply: 5
- Analyze: 5
- Evaluate: 5
- Create: 5

The records use 25 distinct opening verbs. Each record includes its original source question and source-group identifier, target Bloom level, opening verb, level-specific verb bank, preserved elements, preliminary feasibility, semantic risks, and validation status.

## Validation and limitations

- All 30 records remain pending human validation and have `training_eligible=false`.
- Source questions are drawn from the current candidate pool and were checked against the earlier candidate batches by source ID before inclusion. This does not establish global novelty against every historical dataset or repository.
- Some questions depend on omitted expressions, data, cases, or course-specific frameworks; these are flagged in `semantic_risks`. A candidate cannot be considered answerable or meaning-preserving without the missing context.
- The source Bloom label is not proof that the rewrite meets its target cognitive level.
- A varied opening verb is an aid to wording diversity, not a substitute for checking the actual cognitive operation.
- This batch is not a replacement for the existing split datasets and is not training-ready.

Before training, conduct semantic-fidelity and Bloom-alignment review, resolve missing-context cases, inspect near-duplicates, and enforce source-group-disjoint splits.
