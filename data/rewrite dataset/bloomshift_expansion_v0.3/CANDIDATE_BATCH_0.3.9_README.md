# Candidate batch 0.3.9 — template-assisted drafts

- Records: 50 transformations
- Distinct source questions: 10
- Source coverage: two each originally labelled Remember, Apply, Analyze, Evaluate, and Create.
- Target counts: Remember 8; Understand 10; Apply 8; Analyze 8; Evaluate 8; Create 8.
- Status: every record is `pending_human_validation` and `training_eligible=false`.

These are expansion drafts, not gold data. Topic- and constraint-conditioned templates were used to broaden coverage; they may fail to preserve the original assessment purpose or create tasks that are too broad. An unflagged source is not automatically eligible. Do not train on this batch. Subject-matter reviewers must confirm source eligibility, provenance, semantic/content preservation, genuine target-level cognitive operation, and retention of counts, entities, technical details, and format. Resolve duplicates and obtain independent human labels/adjudication before making a training split.