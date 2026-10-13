# BloomShift expansion candidate batch 0.3.5

Adds 30 draft rewrite candidates selected from the Figshare-derived source candidate pool.

## Coverage

Five candidates target each of the six Bloom levels: Remember, Understand, Apply, Analyze, Evaluate, and Create. The batch uses 23 distinct opening verbs.

Each record retains the source question and source-group ID, documents the intended cognitive operation and preserved elements, records context/semantic risks, and remains pending human validation.

## Validation status and limitations

- All candidates have `training_eligible=false`; this is a draft candidate batch, not a validated training dataset.
- Source IDs were checked against candidate batches 0.3.1–0.3.4 and the user-approved candidate JSONL before selection. The sources are unique within this batch and were not used in those fetched files.
- Structural checks do not establish semantic preservation, intended Bloom-level alignment, answerability, or quality.
- Several source prompts depend on omitted figures, data, course frameworks, or context. These dependencies are noted in each candidate's `semantic_risks` field.
- A varied opening verb does not itself prove a cognitive-level transformation.
- Before training, review meaning preservation and target cognitive demand, resolve missing-context cases, inspect near-duplicates, and maintain source-group-disjoint data splits.
