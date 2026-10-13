# BloomShift expansion candidate batch 0.3.8 — assistant-curated drafts

## Contents
- 160 draft transformations across 32 previously unused source questions.
- Five non-identity target-level rewrites per source question.
- Target-level openings use imperative action verbs.
- Every record is marked `pending_human_validation` and `training_eligible=false`.

## Source reconciliation
The 32 source IDs were checked against `source_candidates_for_rewrite.csv` and the other candidate files available in `data/rewrite dataset/bloomshift_expansion_v0.3/`. The source text in each record matches the source-pool row exactly, and no source ID in this batch appears in the other enumerated candidate files. This is an exact-ID/text check, not a semantic duplicate audit.

## Important quality limits
- These are assistant-curated **draft candidates**, not human-validated data.
- A verb-led rewrite does not prove that the target cognitive operation is correctly represented.
- Transformations that introduce a diagram, plan, checklist, evaluation criteria or applied scenario change the task format or add assumptions; these are explicitly noted as changed demands or semantic risks.
- Some source questions refer to course-specific taxonomies or cited materials. Their correctness and answerability require a subject-matter reviewer.
- Source questions and rewritten text may be derived from third-party exam material. Verify provenance, exact version, attribution and redistribution rights before public release.
- Do not train on this batch or include it in the paper's validated-example count until the human-review protocol is complete.

## Recommended review
For each source-target pair, have independent reviewers score: (1) source-topic/content preservation, (2) context preservation, (3) target cognitive operation, (4) answerability, (5) factual/domain accuracy, (6) clarity, and (7) final accept/revise/reject decision. Record disagreements and adjudication. Split by source group only after accepted examples are frozen.
