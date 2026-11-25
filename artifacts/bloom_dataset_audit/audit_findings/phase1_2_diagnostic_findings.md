# EduGuard dataset audit — Phase 1 and diagnostic screening

## Scope and source handling

- Authoritative input: the user-named DOCX copied byte-for-byte into `original_source`. The supplied master prompt was treated as audit instructions, not dataset content.
- Exactly 600 numbered examples are present, IDs 1–600 in order with no gaps or extra examples.
- The DOCX body has 1,882 paragraphs and 605 tables. Of these, 600 are Bloom transformation tables and five are embedded question-data tables.
- The five embedded tables belong to Examples 428, 442, 475, 477, and 548. They were kept with their associated examples.
- 599 transformation tables have the expected five data rows. Example 105's table has four data rows; its Create prompt follows as a paragraph outside the table. The extraction preserves it as the fifth transformation and records its malformed placement. No question was rewritten.
- Every example has a source question and original Bloom label. The extraction preserves additional continuation/metadata paragraphs so the document is not silently flattened. The source includes 40 consecutive proposed CLO/PO contribution notes (Examples 131–170); these are source text, not independently validated mappings.
- DOCX contains 1,933 Office Math XML nodes (536 math text nodes); equation layout and symbols need visual/subject review.

## Structural results

- Extracted transformations: 3,000 across 600 examples after preserving Example 105's orphaned Create row.
- All examples have five distinct target labels and the expected set excluding the original label once the orphaned row is included.
- Original label counts: Remember 9; Understand 74; Apply 212; Analyze 175; Evaluate 75; Create 55.
- Target counts: Remember 591; Understand 526; Apply 388; Analyze 425; Evaluate 525; Create 545. Counts are derived from the source labels and transformation headings; they do not establish semantic correctness.
- Exact normalized duplicate original-question groups detected: 0. Semantic near-duplicate review is pending.


## Directed source-to-target transition counts

These are counts from the source's declared labels and target headings, not semantic validation.

| Source → target | Count | Source → target | Count | Source → target | Count |
|---|---:|---|---:|---|---:|
| Remember → Understand | 9 | Remember → Apply | 9 | Remember → Analyze | 9 |
| Remember → Evaluate | 9 | Remember → Create | 9 | Understand → Remember | 74 |
| Understand → Apply | 74 | Understand → Analyze | 74 | Understand → Evaluate | 74 |
| Understand → Create | 74 | Apply → Remember | 212 | Apply → Understand | 212 |
| Apply → Analyze | 212 | Apply → Evaluate | 212 | Apply → Create | 212 |
| Analyze → Remember | 175 | Analyze → Understand | 175 | Analyze → Apply | 175 |
| Analyze → Evaluate | 175 | Analyze → Create | 175 | Evaluate → Remember | 75 |
| Evaluate → Understand | 75 | Evaluate → Apply | 75 | Evaluate → Analyze | 75 |
| Evaluate → Create | 75 | Create → Remember | 55 | Create → Understand | 55 |
| Create → Apply | 55 | Create → Analyze | 55 | Create → Evaluate | 55 |


## Initial semantic/technical screening candidates

Examples 23, 48, 54, 58, 105, 440, 442, 445, 475, and 477 are listed in `semantic_review_candidates.csv` with the reason each deserves a closer look. These are diagnostic flags, not final adjudications. Notable cases include the unspecified series in Example 58, missing proposed FIRST/FOLLOW values in Example 445's Evaluate prompt, and the likely Apply-versus-Analyze issue in Example 475.

## Audit boundary

This is a Phase 1 structural audit and Phase 2 diagnostic screening. It is not yet a completed question-by-question academic audit: original Bloom decisions, all 3,000 transformation levels, content preservation, technical correctness, near duplicates, and transformation-style patterns still require semantic review. No examples have been corrected, rejected, or removed. No training-ready files or splits have been generated.
