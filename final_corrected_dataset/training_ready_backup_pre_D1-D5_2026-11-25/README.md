# EduGuard Bloom Transformation Dataset — Training-Ready Export

This directory holds training exports derived from the corrected 600-example master dataset in the parent folder (`../corrected_dataset.json`). The master and its correction history are **not modified**: `integrity_report.json` records the SHA-256 of every master file before and after the build, and they match.

> **Scope of this release.** The checks here establish **data integrity and training eligibility only**: structure, labels, exclusions, leakage, and reproducible splits. They say nothing about how well a model trained on this data will perform. No model has been trained or evaluated on these splits.

## What each example contains

Each example has one source question with its Bloom level, and five rewritten questions on the same content, one for each of the other five levels of the revised Bloom's taxonomy (Remember, Understand, Apply, Analyze, Evaluate, Create). Levels were judged by the cognitive operation required, not by the leading verb.

## Files

| File | Contents |
|---|---|
| `usable_examples.jsonl` | 341 self-contained, technically checked examples |
| `label_and_generation_only_examples.jsonl` | 245 examples whose labels and wording are valid but whose answer data is missing, each with a `missing_answer_source` block |
| `splits/{train,validation,test}_examples.jsonl` | All 586 eligible examples, split by leakage group |
| `splits/{split}_transformation_records.jsonl` | One record per source → target pair (5 per example) |
| `splits/{split}_bloom_classification_records.jsonl` | One labelled question per record (source plus 5 transformations, 6 per example) |
| `splits/split_manifest.json` | Seed, method, per-split counts, level distribution, and the example IDs in each split |
| `exclusions.csv` | The 14 examples left out of every training export, with reasons |
| `eligibility_reclassifications.csv` | 31 examples moved from `usable` to `label_and_generation_only` in this export, plus 3 checked false positives |
| `self_containment_audit.csv` | Verdict for each of the 152 rows marked "Self-containment" in the correction log |
| `leakage_report.csv` | Every related pair found and the group it forced |
| `residual_similarity.json` | Most similar cross-split question pairs remaining after grouping |
| `integrity_report.json` | Counts, integrity checks, leakage summary, master-file hashes |
| `schema.json` | JSON Schema for the three record types |
| `scripts/make_training_ready.py` | Rebuilds every file above from the master (standard library only) |
| `scripts/check_leakage.py` | Independent leakage check over the produced splits |

## Fields

See `schema.json` for the full definitions.

- **Example records:** `id`, `discipline`, `topic`, `source_question`, `source_bloom_level`, and `transformations` (an object keyed by the five other levels). Also `eligibility`, `answer_source_available`, `group_id`, and `split`.
- **Optional example fields:**
  - `source_bloom_level_before_correction` (six examples)
  - `data_tables` (428, 442, 475, 548)
  - `proposed_clo` (131–170)
  - `missing_answer_source` (every `label_and_generation_only` example)
  - `review_notes` (minor audit notes)
- **Transformation records:** `record_id` (e.g. `461-Evaluate`), `source_question`, `source_bloom_level`, `target_bloom_level`, `target_question`, plus the example's identifiers, split and eligibility.
- **Classification records:** `question`, `bloom_level`, `role` (`source` or `transformation`), plus identifiers, split and eligibility.

## How the 600 master examples are used

| Status | Examples | In training exports? |
|---|---:|---|
| `usable` | 341 | Yes |
| `label_and_generation_only` | 245 | Yes, in a separate file and marked in every split record |
| Excluded: redundant duplicate | 8 | No |
| Excluded: pending expert (SME) review | 3 | No |
| Excluded: self-containment row flagged | 3 | No |
| **Total** | **600** | 586 eligible |

**Changes from the master `training_use` flags.** The master marked 375 examples `usable` and 214 `label_and_generation_only`. The export differs in two ways; the master file itself is unchanged.

1. **31 examples reclassified to `label_and_generation_only`:** 082, 083, 084, 089, 090, 091, 092, 096, 098, 099, 105, 107, 123, 127, 132, 134, 142, 144, 150, 159, 160, 161, 165, 167, 179, 187, 229, 234, 278, 301, 335.
   - One or more of their rows still refer to "supplied" material (values, a circuit, a dataset) that the item does not contain.
   - The master's missing-data detector looked only at source questions, and these originals say "its …" or "the specified …" rather than "supplied", so they were missed.
   - Three apparent matches are not missing data and stay `usable`: 043 ("heat supplied"), 428 (refers to its own embedded table) and 478 ("reactive power supplied").
2. **3 examples excluded after the self-containment audit:** 158, 319, 393.

## Exclusions (in all training exports; kept in the master)

| Example | Reason | Related |
|---|---|---|
| 156, 262, 502, 509, 517, 530, 578, 597 | Redundant duplicate: same objective, same data, same level | Kept partners: 027, 187, 421, 472, 458, 417, 512, 457 |
| 429 | Pending SME review | Game tree puts MAX nodes under a MAX root; the root value is 9 as written, 3 if the middle level was meant to be MIN |
| 445 | Pending SME review | FIRST/FOLLOW sets were empty in the source and were filled from the grammar; needs author confirmation |
| 477 | Pending SME review | The heuristic is inadmissible at A and C; the path answer (S–A–C–G, cost 7) is still correct |
| 158 | Self-containment row flagged | Apply row moves from pharmacological *activity* to absorption (pharmacokinetics), a different objective |
| 319 | Self-containment row flagged | Apply row treats the 1906 founding of the Muslim League as a consequence of the 1905 partition, which is contested |
| 393 | Self-containment row flagged | Apply and Create rows expand "public vs private goods" to four categories of goods |

The kept partner of each redundant duplicate stays eligible.

## Self-containment audit (152 rows)

These are the rows where the second correction pass wrote concrete data into a transformation because its generic source question had none. Each row was read against the source question's learning objective, domain, cognitive demand and technical scope.
- **148 preserve the objective.** Four carry non-blocking notes:
  - 305: the USD 2.15/day poverty line is time-sensitive; check the current World Bank figure before publication.
  - 120: glass vials may be sterilised by either method.
  - 170: composting is classed under the "disposal" stage.
  - 339: the Create row narrows the topic to the chloride limit test.
- **4 rows across 3 examples are flagged** (see the exclusions table). They were neither reverted nor accepted. Their examples are held out pending an author decision.

## Integrity checks (all pass; see `integrity_report.json`)

- Every eligible example has one source question with a valid Bloom label and exactly five non-empty transformations covering the other five levels.
- No transformation repeats its source question, and no example repeats a transformation.
- No `usable` example cites "supplied" material it doesn't contain, apart from the three documented false positives.
- Excluded examples appear in no training export.
- Every exported record conforms to `schema.json` (7,618 records checked).
- The master files are byte-identical before and after the build.

## Leakage control and splits

**Grouping.** Examples are joined into one leakage group if any of the following holds:
- They share an identical question text.
- Their source questions have token Jaccard ≥ 0.35.
- Any question in one and any question in the other have Jaccard ≥ 0.40.
- Their subtopic titles match, or one subtopic title is contained in the other.
- They are a documented near-duplicate pair.

This gives 405 groups; the largest has 6 examples, and 130 have more than one.

**Splitting.** Whole groups are assigned in seeded random order (seed `20261125`) to the split furthest below its 80/10/10 target. The split is grouped, not stratified.

| Split | Examples | `usable` | `label_and_generation_only` | Groups | Transformation records | — from `usable` | Classification records |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 469 | 270 | 199 | 323 | 2,345 | 1,350 | 2,814 |
| validation | 59 | 36 | 23 | 40 | 295 | 180 | 354 |
| test | 58 | 35 | 23 | 42 | 290 | 175 | 348 |
| **Total** | **586** | **341** | **245** | **405** | **2,930** | **1,705** | **3,516** |

**Leakage results** (`scripts/check_leakage.py`): no example ID, group or identical question text appears in more than one split. The highest remaining cross-split question similarity is Jaccard 0.364, with 6 pairs at or above 0.35. These are related concepts with different problems, for example:
- material vs labour variance (504/591)
- capital budgeting vs NPV (094, 074 / 461)
- foundation design vs bearing capacity (088 / 543, 595)

If your evaluation needs concept-disjoint splits, group by subject area instead.

**Reproducibility.** Run `python -I training_ready/scripts/make_training_ready.py` from the parent folder. Two consecutive runs produced byte-identical outputs.

## Recommended usage

- **Bloom-level classification:** use `*_bloom_classification_records.jsonl` from both eligibility classes. Labels are valid even where answer data is missing.
- **Level-targeted question rewriting (source → target level):** use `*_transformation_records.jsonl` from both classes. These records ask for rewritten questions, not answers.
- **Anything that needs a correct answer** (answer generation, solution checking, reward models built on answers): use **only** `eligibility == "usable"` records. Do not use `label_and_generation_only` records, because their answers cannot be derived from the item.
- **Grouping:** keep `group_id` intact if you re-split (for example, for k-fold). Splitting a group defeats the leakage control.
- **Small splits:** validation and test have about 58 examples each (about 290 transformation records). Results on them will have wide uncertainty; report confidence intervals. Validation has no Remember source questions, and test has none for Remember and only 4 for Evaluate. As target levels, every level appears in every split.

## Limitations and open issues

- **Not validated for model performance.** Nothing here measures model accuracy.
- **Single reviewer.** The correction passes and this audit were done by one reviewer with script support. Law and pharmacy items in particular would benefit from subject-specialist review. Legal items avoid jurisdiction-specific rules.
- **Missing source data:** 245 eligible examples (and many master rows) refer to data absent from the source PDF. Nothing was invented to fill it.
- **Authored data in generic items:** in 148 audited rows, the second pass wrote small, internally consistent scenarios into transformations whose generic source had no data. The values were checked independently, but they are authored content, not source content (see `self_containment_audit.csv`).
- **Deliberate errors:** many Evaluate rows state a wrong value or claim on purpose for error-detection practice. The correct answer is in `../Correction_Log.csv`. A model trained to generate answers must learn to reject these claims, not copy them.
- **Label imbalance:** source labels are skewed (Apply 216, Analyze 175, Remember 7 in the master). Transformation targets are balanced by construction.
- **Seven open items:** 3 SME items (429, 445, 477) and 4 flagged audit rows (158, 319, 393) await an author decision. Once resolved, rerun the build script to include them.
