# EduGuard Bloom Transformation Dataset — Training-Ready Export

This directory holds training exports derived from the corrected 600-example master dataset in the parent folder (`../corrected_dataset.json`). The master and its correction history are **not modified**: `integrity_report.json` records the SHA-256 of every master file before and after the build, and they match.

> **Scope of this release.** The checks here establish **data integrity and training eligibility only**: structure, labels, exclusions, leakage, provenance and reproducible splits. They say nothing about how well a model trained on this data will perform. No model has been trained or evaluated on these splits.

## What each example contains

Each example has one source question with its Bloom level, and five rewritten questions on the same content, one for each of the other five levels of the revised Bloom's taxonomy (Remember, Understand, Apply, Analyze, Evaluate, Create). Levels were judged by the cognitive operation required, not by the leading verb.

**The dataset contains questions only, not answers.** Where the Correction_Log documents a correct answer (for example, for a deliberately false Evaluate claim), it appears as a reviewer note, not as a training target.

## Files

| File | Contents |
|---|---|
| `usable_examples.jsonl` | 341 self-contained, technically checked examples |
| `label_and_generation_only_examples.jsonl` | 245 examples whose labels and wording are valid but whose answer data is missing, each with a `missing_answer_source` block |
| `splits/{train,validation,test}_examples.jsonl` | All 586 eligible examples, split by leakage group |
| `splits/{split}_transformation_records.jsonl` | One record per source → target pair (5 per example), both eligibility classes |
| `splits/{split}_bloom_classification_records.jsonl` | One labelled question per record (source plus 5 transformations, 6 per example) |
| `splits/split_manifest.json` | Seed, method, per-split counts, level distribution, and the example IDs in each split |
| `answer_generation/{split}_transformation_records.jsonl` | **Usable-only** transformation records for any task that needs a correct answer (1,350 / 180 / 175) |
| `answer_generation/answer_generation_manifest.json` | Inclusion rule, allow-list (341 usable IDs), deny-list (245 label-only IDs), per-split counts and file SHA-256s |
| `exclusions.csv` | The 14 examples left out of every training export, with reasons |
| `eligibility_reclassifications.csv` | 31 examples moved from `usable` to `label_and_generation_only` in this export, plus 3 checked false positives |
| `self_containment_audit.csv` | Verdict for each of the 152 rows marked "Self-containment" in the correction log |
| `leakage_report.csv` | Every related pair found and the group it forced |
| `residual_similarity.json` | Most similar cross-split question pairs remaining after grouping |
| `integrity_report.json` | Counts, integrity checks, provenance and false-claim counts, leakage summary, master-file hashes |
| `schema.json` | JSON Schema for the three record types |
| `RECONCILIATION_REPORT.md` | Pre-training reconciliation; its defects D1–D5 are fixed in this build |
| `scripts/make_training_ready.py` | Rebuilds every data file above from the master (standard library only) |
| `scripts/eduguard_loader.py` | Safe loaders; **required** for any answer-dependent use (see below) |
| `scripts/verify_training_ready.py` | Independent post-build verification; exits non-zero on any failure |
| `scripts/check_leakage.py` | Independent leakage check over the produced splits |
| `tests/test_training_ready.py` | Automated tests, including deliberate attempts to slip label-only records into answer data |
| `tests/reference_split_ids.json` | Split ID lists frozen from the build before fixes D1–D5 |

A copy of the export as it was before fixes D1–D5 is in `../training_ready_backup_pre_D1-D5_2026-11-25/`.

## Fields

See `schema.json` for the full definitions.

- **Example records:** `id`, `discipline`, `topic`, `source_question`, `source_bloom_level`, and `transformations` (an object keyed by the five other levels). Also `eligibility`, `answer_source_available`, `group_id`, `split`, and:
  - `transformation_provenance`: provenance of each transformation (see *Content provenance*)
  - `authored_scenario_levels`: levels whose row is an authored scenario
  - `deliberately_false_claim_levels`: `["Evaluate"]` when the Evaluate row states a deliberately false claim, otherwise `[]`
- **Optional example fields:**
  - `source_bloom_level_before_correction` (six examples)
  - `data_tables` (428, 442, 475, 548)
  - `proposed_clo` (131–170)
  - `missing_answer_source` (every `label_and_generation_only` example)
  - `review_notes` (minor audit notes: 120, 170, 305, 339)
- **Transformation records:**
  - `record_id` (e.g. `461-Evaluate`), `source_question`, `source_bloom_level`, `target_bloom_level`, `target_question`, plus the example's identifiers, split and eligibility
  - `content_provenance`, `provenance_evidence`, `deliberately_false_claim`
  - `claim_under_evaluation`, present only when `deliberately_false_claim` is true
- **Classification records:** `question`, `bloom_level`, `role` (`source` or `transformation`), `content_provenance` (`source_question` for source rows) and `deliberately_false_claim`, plus identifiers, split and eligibility.

## How the 600 master examples are used

| Status | Examples | In training exports? |
|---|---:|---|
| `usable` | 341 | Yes, including `answer_generation/` |
| `label_and_generation_only` | 245 | Yes in `splits/` (marked in every record); **never** in `answer_generation/` |
| Excluded: redundant duplicate | 8 | No |
| Excluded: pending expert (SME) review | 3 | No |
| Excluded: self-containment row flagged | 3 | No |
| **Total** | **600** | 586 eligible |

**Changes from the master `training_use` flags.** The master marked 375 examples `usable`, 214 `label_and_generation_only`, 8 `exclude_redundant_duplicate` and 3 `usable_pending_sme_check`. The export differs in two ways; the master file itself is unchanged.

1. **31 examples reclassified to `label_and_generation_only`:** 082, 083, 084, 089, 090, 091, 092, 096, 098, 099, 105, 107, 123, 127, 132, 134, 142, 144, 150, 159, 160, 161, 165, 167, 179, 187, 229, 234, 278, 301, 335.
   - One or more of their rows still refer to "supplied" material (values, a circuit, a dataset) that the item does not contain.
   - The master's missing-data detector looked only at source questions, and these originals say "its …" or "the specified …" rather than "supplied", so they were missed.
   - Three apparent matches are not missing data and stay `usable`: 043 ("heat supplied"), 428 (refers to its own embedded table) and 478 ("reactive power supplied").
2. **3 examples excluded after the self-containment audit:** 158, 319, 393.

So the master's 375 `usable` examples become 341 usable, 31 label-only and 3 held out.

## Exclusions (in all training exports; kept in the master)

**Six examples are held out pending an author decision: 429, 445, 477, 158, 319 and 393.** Three await SME review. The other three contain the four flagged audit rows (393 has two). The eight redundant duplicates are excluded permanently.

| Example | Reason | Related |
|---|---|---|
| 156, 262, 502, 509, 517, 530, 578, 597 | Redundant duplicate: same objective, same data, same level | Kept partners: 027, 187, 421, 472, 458, 417, 512, 457 |
| 429 | Pending SME review | Game tree puts MAX nodes under a MAX root; the root value is 9 as written, 3 if the middle level was meant to be MIN |
| 445 | Pending SME review | FIRST/FOLLOW sets were empty in the source and were filled from the grammar; needs author confirmation |
| 477 | Pending SME review | The heuristic is inadmissible at A and C; the path answer (S–A–C–G, cost 7) is still correct |
| 158 | Self-containment row flagged | Apply row moves from pharmacological *activity* to absorption (pharmacokinetics), a different objective |
| 319 | Self-containment row flagged | Apply row treats the 1906 founding of the Muslim League as a consequence of the 1905 partition, which is contested |
| 393 | Self-containment row flagged (2 rows) | Apply and Create rows expand "public vs private goods" to four categories of goods |

Every kept partner of a redundant duplicate stays eligible. Partner 187 is `label_and_generation_only`; the others are `usable`.

## Self-containment audit and authored scenarios

The self-containment rows are where a correction pass wrote concrete scenario data (values, cases, items) into a transformation because its generic source question had none. All 152 such rows in the Correction_Log were read against the source question's learning objective, domain, cognitive demand and technical scope.

**By correction pass:**

| Pass | All 152 audited rows | 148 preserving the objective | 144 exported |
|---|---:|---:|---:|
| Pass 2 only | 119 | 115 | 112 |
| Passes 1 and 2 | 31 | 31 | 30 |
| Pass 1 only | 2 | 2 | 2 |

**By verdict:**

- **148 rows preserve the objective.** These are the 148 authored scenarios.
  - **144 are in the current exports:** 143 in `usable` examples and 1 in a `label_and_generation_only` example.
  - **4 are excluded with their examples:** 156-Apply and 156-Analyze (156 is a redundant duplicate), and 158-Analyze and 158-Create (158 is held out).
  - Four of the 148 carry non-blocking notes, stored in `review_notes`:
    - 305: the USD 2.15/day poverty line is time-sensitive; check the current World Bank figure before publication.
    - 120: glass vials may be sterilised by either method.
    - 170: composting is classed under the "disposal" stage.
    - 339: the Create row narrows the topic to the chloride limit test.
- **4 rows across 3 examples are flagged:** 158-Apply, 319-Apply, 393-Apply and 393-Create. They were neither reverted nor accepted, and their examples are held out.

## Content provenance

Every transformation carries one `content_provenance` value, taken from the master's Correction_Log. No text was changed.

| Value | Meaning | Transformation records exported |
|---|---|---:|
| `authored_scenario` | Scenario data written into the row because its generic source had none (a row of `self_containment_audit.csv`). `provenance_evidence` gives the correction pass and audit verdict. | 144 |
| `authored_evaluation_object` | An Evaluate or Analyze row where a correction pass wrote in a specific object to judge (a claim, design, proof or value): log category "Evaluate – object/criteria specified" or "Analyze – object specified", with the reason saying the object "is now stated/given/described". | 118 |
| `source_derived` | No authored scenario or object is recorded. The row rewrites the source question's own content, though its wording may have been edited for level or notation (see `../Correction_Log.csv`). | 2,668 |

Source questions are `source_question` in the classification records. The authored values were checked independently, but they are authored content, not content from the source PDF.

## Deliberately false Evaluate claims

Some Evaluate rows state a wrong value, claim, proof or design on purpose, so the learner must detect the error. Each is phrased as something to judge ("Assess whether…", "Verify whether…"), never as a fact.

- **62 such rows are documented in the Correction_Log:**
  - **17** say "error-detection" explicitly: 027, 043, 064, 118, 415, 436, 442, 447, 478, 483, 499, 514, 537, 548, 562, 593, 599.
  - **45** say the stated object is "(incorrect)", "faulty", "flawed", "swapped", "infeasible" or contains "a sign error": 076, 100, 103, 116, 124, 128, 132, 147, 156, 172, 178, 192, 211, 221, 226, 231, 234, 235, 241, 243, 244, 250, 264, 268, 271, 272, 279, 284, 285, 295, 297, 305, 309, 310, 327, 339, 347, 361, 374, 393, 401, 405, 429, 496, 506.
- **59 are exported.** The other three (156, 393, 429) belong to excluded examples.
  - By split: train 48, validation 4, test 7.
  - In `answer_generation/`: 46 / 3 / 6.
- **Checked but not counted:**
  - 573: its reported value was corrected so that the claim is right.
  - 240: "faults" there means page faults.
- **How they are marked:** each record has `deliberately_false_claim: true` and a `claim_under_evaluation` block. The block has `role: "object_of_evaluation"`, `is_ground_truth: false`, the evidence type and the Correction_Log note with the correct reasoning. The question text is unchanged.
- **Where the false claims come from:** 45 of the 59 exported false claims are in `authored_evaluation_object` rows. The other 14 are `source_derived`: the wrong value was already in the source or pass-1 wording and was documented as an answer-key note.
- **Limitation:** 68 other exported Evaluate rows are `authored_evaluation_object` rows whose log entry doesn't say whether the stated object is right or wrong. They and all other rows are marked `deliberately_false_claim: false`, which means "not documented as false", not "verified true".

## Answer-generation safeguard

`label_and_generation_only` examples cannot have correct answers derived from the item, so they must never be used for any answer-dependent task (training, validation or testing).

- **Use `answer_generation/` through `scripts/eduguard_loader.py`:**

  ```python
  import sys; sys.path.insert(0, 'training_ready/scripts')
  from eduguard_loader import load_answer_generation_records, select_answer_eligible
  train = load_answer_generation_records('train')   # 1,350 records; raises LabelOnlyRecordError if anything is wrong
  ```

- **Every record is checked explicitly**, never by looking for a missing answer field:
  - `eligibility == "usable"`
  - `answer_source_available is true`
  - the example ID is in the allow-list of usable IDs
  - the example ID is not in the deny-list of label-only IDs

  A label-only record relabelled as `usable` still fails the allow-list and deny-list checks. The loader also checks each file's SHA-256 against the manifest.
- **Any conversion script** that starts from the general `splits/` files for an answer task must call `select_answer_eligible(records)` (an explicit filter) or `assert_answer_eligible(records)` (rejection).
- **Existing trainers do not filter.** The trainers in `../../experiments/` (for example `multitask_bloom_rewrite/scripts/dataset_io.py`) don't read these files and don't filter on eligibility. Convert through the loader rather than pointing them at `splits/` directly.
- `tests/test_training_ready.py` deliberately tries five ways to get label-only data in and checks that each is rejected:
  - appending a label-only record
  - relabelling one as `usable`
  - passing a general split file
  - tampering with the answer file
  - tampering with both the answer file and its manifest hash

## Integrity checks (all pass; see `integrity_report.json`)

- Every eligible example has one source question with a valid Bloom label and exactly five non-empty transformations covering the other five levels: 341 × 5 = 1,705 usable and 245 × 5 = 1,225 label-only transformation records.
- No transformation repeats its source question, and no example repeats a transformation.
- No `usable` example cites "supplied" material it doesn't contain, apart from the three documented false positives.
- Excluded examples appear in no training export.
- Answer-generation exports contain only `usable` examples.
- Every exported record conforms to `schema.json`: 9,323 records, checked with `jsonschema` 4.26 and with the validator in the test suite.
- The master files are byte-identical before and after the build.

## Leakage control and splits

**Grouping.** Examples are joined into one leakage group if any of the following holds:
- They share an identical question text.
- Their source questions have token Jaccard ≥ 0.35.
- Any question in one and any question in the other have Jaccard ≥ 0.40.
- Their subtopic titles match, or one subtopic title is contained in the other.
- They are a documented near-duplicate pair.

This gives 405 groups; the largest has 6 examples, and 130 have more than one.

**Splitting.** Whole groups are assigned in seeded random order (seed `20261125`) to the split furthest below its 80/10/10 target. The split is grouped, not stratified. The D1–D5 fixes added fields only; split assignment is unchanged (checked against `tests/reference_split_ids.json`).

| Split | Examples | `usable` | `label_and_generation_only` | Groups | Transformation records | Answer-generation records (usable only) | Classification records |
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

**Level coverage.**

| Split | Source levels R / U / Ap / An / E / C (all) | Usable only |
|---|---|---|
| train | 7 / 54 / 168 / 140 / 60 / 40 | 7 / 48 / 85 / 82 / 25 / 23 |
| validation | 0 / 10 / 17 / 16 / 10 / 6 | 0 / 9 / 10 / 11 / 4 / 2 |
| test | 0 / 11 / 21 / 18 / 4 / 4 | 0 / 9 / 9 / 14 / 1 / 2 |

- All 7 Remember source questions are in train, so rewriting *from* Remember cannot be evaluated. Evaluate and Create as source levels are very sparse in validation and test.
- As target levels, every level appears in every split, with at least 21 usable records per level.
- Classification labels are balanced by construction: each example contributes one question per level.

## Reproducing and verifying

From the folder that holds `corrected_dataset.json`:

```
python -I training_ready/scripts/make_training_ready.py      # rebuild all data files
python -I training_ready/scripts/check_leakage.py            # independent leakage check
python -I training_ready/scripts/verify_training_ready.py    # independent verification (exit 1 on failure)
python -I -m unittest discover -s training_ready/tests -v    # 23 automated tests
```

Two consecutive builds produce byte-identical outputs.

## Recommended usage

- **Bloom-level classification:** use `splits/*_bloom_classification_records.jsonl` from both eligibility classes. Labels are valid even where answer data is missing.
- **Level-targeted question rewriting (source → target level):** use `splits/*_transformation_records.jsonl` from both classes. These records ask for rewritten questions, not answers. A rewriting model trained on Evaluate targets will learn to embed claims to judge; 59 of those claims are deliberately false.
- **Anything that needs a correct answer** (answer generation, solution checking, answer-based reward models): use **only** `answer_generation/` through `scripts/eduguard_loader.py`. Treat any record with `deliberately_false_claim: true` as asking the model to reject the stated claim, using `claim_under_evaluation.correction_log_note`. No answers are included, so answer targets would have to be authored and checked separately.
- **Grouping:** keep `group_id` intact if you re-split (for example, for k-fold). Splitting a group defeats the leakage control.
- **Small splits:** validation and test have about 58 examples each (about 290 transformation records, about 178 answer-eligible). Results on them will have wide uncertainty; report confidence intervals and report by target level.

## Limitations and open issues

- **Not validated for model performance.** Nothing here measures model accuracy.
- **Single reviewer.** The correction passes and this audit were done by one reviewer with script support. Law and pharmacy items in particular would benefit from subject-specialist review. Legal items avoid jurisdiction-specific rules.
- **Missing source data:** 245 eligible examples (and many master rows) refer to data absent from the source PDF. Nothing was invented to fill it.
- **Authored content:** 144 exported rows are authored scenarios and 118 are authored evaluation objects. They are marked, checked content, but not content from the source.
- **False claims are only as complete as the log.** Rows introduced without a documented right/wrong status are not marked as false (see *Deliberately false Evaluate claims*). False or flawed objects in non-Evaluate rows (for example, debugging tasks) are not marked.
- **Label imbalance:** source labels are skewed (Apply 216, Analyze 175, Remember 7 in the master). Transformation targets are balanced by construction.
- **Six held-out examples** (429, 445, 477, 158, 319, 393) await an author decision. Once resolved, rerun the build script to include them. Doing so would change the splits, so the reference ID lists in `tests/` would need a deliberate update.
