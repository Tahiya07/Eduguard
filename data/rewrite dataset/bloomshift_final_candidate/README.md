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

### Human annotation and gold-release protocol

The 228 validation/test examples should be reviewed independently by at least two annotators before any claim of human validation. Reviewers should not see another reviewer's labels before submitting their own decision. A short calibration batch of 20 examples should be annotated first and discussed to resolve interpretation differences; the calibration batch is for protocol alignment and should not be used as the reported agreement estimate.

For the main annotation pass, each reviewer records all controlled fields independently. `reviewer_bloom_level` is the reviewer's own judgment, not an attempt to reproduce the target label. `content_preserved` assesses whether the substantive topic, entities, facts, and requested task content remain intact; `context_preserved` assesses conditions, scenario, constraints, audience, and other contextual qualifiers. `bloom_aligned` should be Yes only when the cognitive demand is consistent with the target level, not merely because a target-level verb appears. `meaningful_transformation` should be judged against the source task, and `pedagogically_valid` should consider whether the resulting question is answerable and educationally coherent. Reviewers should use `reason` to cite the specific evidence for any No/Partial/Revise/Reject judgment.

For agreement reporting, calculate Cohen's kappa for categorical fields when exactly two annotators are used, and use a multi-rater agreement statistic such as Krippendorff's alpha if more than two annotators contribute labels. Report agreement separately for Bloom alignment, reviewer Bloom level, content preservation, context preservation, meaningful transformation, pedagogical validity, language quality, and final decision. Do not collapse these into a single unsupported quality score. Agreement should be reported before adjudication.

Disagreements should be adjudicated by a senior reviewer or supervisor using the original source question, target question, and both rationales. The adjudicator may Accept, Revise, or Reject the example and should record the corrected question and reason where applicable. Adjudication changes must be applied to the dataset and annotation file together. The adjudicated result is the release label; raw independent labels should be retained separately for agreement analysis.


### Annotation infrastructure

The repository now includes a reproducible annotation workspace:

- `annotation_calibration.csv`: 20-example calibration batch covering all six target Bloom levels. It contains no reviewer labels and is excluded from reported agreement.
- `reviewer_1.csv` and `reviewer_2.csv`: identical blank worksheets for independent annotation of all 228 validation/test examples. Human fields are intentionally empty.
- `annotation_agreement_template.csv`: pre-adjudication agreement-reporting template for each annotation field.
- `annotation_manifest.json`: records the annotation version, calibration IDs, reviewer count, priority-review count, and file roles.

The calibration set contains two validation and one test example per target level, plus the two previously corrected held-out marketing-strategy examples. The latter are included to ensure the calibration process explicitly exercises examples that previously required content-preservation correction. No reviewer decision is prefilled.


### Proposed gold-release criteria

BloomShift should not be called a human-validated gold dataset until: (1) all 228 validation/test examples have completed independent review; (2) every example has a resolved final decision; (3) examples marked Reject are excluded from the released gold split; (4) Revise cases are corrected and re-reviewed; (5) agreement statistics are reported for the main annotation pass; (6) the 32 cases flagged by the semantic/content-preservation screen receive explicit review; and (7) the final released files, annotation table, and audit record are synchronized. Any threshold for acceptable agreement should be declared in the paper before results are interpreted rather than selected after seeing the outcomes.

Until these conditions are met, the repository should continue to describe the dataset as a silver candidate.


### Source-to-target transition coverage

The candidate contains 28 of the 30 possible non-identity source-to-target Bloom transitions. The two absent directions are **Remember → Evaluate** and **Remember → Create**, reflecting the very small number of Remember-level source questions in the underlying source dataset. Target levels are balanced within each split, but the source-to-target transition matrix is therefore not fully balanced.

These missing transitions should not be filled by synthetic source questions solely to improve matrix balance, because doing so would alter source provenance and require a new curation cycle. Benchmark reporting should therefore describe target-level balance and explicitly disclose the incomplete source-to-target transition coverage.
### Agreement analysis

After both reviewers complete the 228-example main annotation pass, run the reproducible agreement checker from the repository root:

```bash
python scripts/analyze_annotation_agreement.py \\
  --reviewer1 "data/rewrite dataset/bloomshift_final_candidate/reviewer_1.csv" \\
  --reviewer2 "data/rewrite dataset/bloomshift_final_candidate/reviewer_2.csv" \\
  --output "data/rewrite dataset/bloomshift_final_candidate/annotation_agreement.json"
```

The script refuses to compute agreement while any controlled human-label field is blank, checks reviewer alignment and duplicate IDs, excludes the separate calibration batch, and reports Cohen's kappa separately for each annotation dimension. It does not generate or alter reviewer labels.
