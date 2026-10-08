# EduGuard BloomShift — Final Candidate

Recommended candidate for supervisor validation before the next multitask retraining run.

Train: 660 records
Validation: 114 records
Test: 114 records
Total: 888 records

Each split is target-balanced. Same-level identity copies are excluded. Source groups remain disjoint across train, validation, and test.

The dataset was rebuilt from the original EduGuard source questions. Previous generated target rewrites were not reused. Incomplete/self-referential source questions and source/task families that could not support a reasonable content-preserving transformation were excluded.

This is a silver candidate, not a human-certified gold dataset. Use supervisor_annotation.csv to review every validation/test example before freezing the dataset.

A small set of mechanically malformed held-out rewrites identified during the final audit was corrected without regenerating the candidate dataset. The supervisor annotation file was updated to remain synchronized with those corrections. Semantic, Bloom-level, and pedagogical validity of the held-out examples still requires human review.

The rewrite forms were also diversified to reduce dependence on repeated WH-question openings. Action-oriented forms such as *analyze*, *apply*, *assess*, *design*, *construct*, *identify*, and *state* are now used alongside interrogative forms. This is intended to reduce superficial wording regularity; it does not establish Bloom-level validity, which remains subject to supervisor review.

## Supervisor annotation protocol

The held-out validation and test examples are intended for human review. The annotation schema separates content preservation from context preservation so reviewers can assess whether the original subject matter, entities, conditions, constraints, and task setting remain intact while cognitive demand changes. Reviewers should also record Bloom alignment, their independently judged Bloom level, meaningful transformation, pedagogical validity, language quality, and a final decision. Blank annotation fields are intentional until human review is performed.

### Recommended controlled annotation values

For reproducible human annotation, use the following controlled values:

- `bloom_aligned`: Yes / No
- `reviewer_bloom_level`: Remember / Understand / Apply / Analyze / Evaluate / Create / Unclear
- `content_preserved`: Yes / No / Partial
- `context_preserved`: Yes / No / Partial
- `meaningful_transformation`: Yes / No / Partial
- `pedagogically_valid`: Yes / No / Partial
- `clear_and_grammatical`: Yes / No / Partial
- `final_decision`: Accept / Revise / Reject
- `corrected_question`: enter only when revision is needed; otherwise leave blank
- `reason`: brief evidence-based justification for No/Partial/Revise/Reject decisions

These are annotation instructions, not pre-filled labels. Reviewers should independently judge each held-out rewrite before the dataset is treated as human-validated.


## Quality audits

A deterministic content-preservation audit was performed across all 888 records. It screened source-anchor coverage and checked preservation of source numerals, quoted material, and explicit negation cues. Two held-out marketing-strategy rewrites were corrected to restore source-specific product/model context, and the annotation file was synchronized.

A separate language-quality audit screened all 888 target rewrites for mechanically malformed template phrasing. A limited set of duplicated words, source-imperative insertion errors, duplicated distinction verbs, and malformed evaluation endings was corrected without intentionally changing the source topic or target cognitive operation. These automated checks are screening controls only; they do not establish semantic equivalence, Bloom alignment, or pedagogical validity. Human review remains required before gold release.


### Bloom cognitive-validity audit

A Bloom-level screening was performed across all 888 target rewrites using action-verb cues together with structural inspection of repeated transformation templates. This confirmed that lexical cues alone are not sufficient to certify Bloom level: valid transformations may express the intended operation structurally, while action verbs can occur in mixed cognitive contexts. Clearly weak or malformed transformations were corrected, including evaluation-derived Understand rewrites that asked the reader to identify what was being judged rather than understand or explain the underlying content. The corrected candidate still requires human Bloom-level annotation before it can be treated as a gold benchmark.

### Semantic/content-preservation audit

A deterministic content-preservation screen was run across all 888 source/target pairs. It checked content-bearing lexical overlap as a risk indicator and separately checked source numerals, quoted material, and explicit source identifiers. The overlap screen flagged 32 cases (16 train, 6 validation, 10 test) for manual review, but this is not evidence of semantic failure: Bloom transformation necessarily introduces cognitive-operation language and inflectional changes. No source numerals or quoted material were lost in the screen, and no automatic corrections were applied because semantic similarity cannot be established safely from lexical overlap alone.

The 32 flagged cases should receive priority during human validation. This audit is therefore a screening step, not a claim of semantic equivalence or human certification.
