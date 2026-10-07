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
