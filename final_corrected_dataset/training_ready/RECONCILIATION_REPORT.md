# Reconciliation report — training-ready export

> **Update (2026-11-25, after fixes D1–D5): all five defects below are fixed. See README.md.**
>
> **Correction to this report:** section 6 said 17 deliberately false Evaluate claims. That counted only the rows whose log entry says "error-detection". The Correction_Log documents **62**: those 17, plus 45 whose stated object is logged as "(incorrect)", "faulty", "flawed", "swapped", "infeasible" or containing "a sign error". **59** are exported; 156, 393 and 429 are excluded. All 62 are marked.
>
> **Authored rows in excluded examples:** section 2 said 4 authored rows belong to excluded examples. That counted only the PRESERVES rows (156 ×2, 158 ×2). Counting all 152 audited rows, 8 are in excluded examples, because the 4 flagged rows (158, 319, 393 ×2) are also excluded.
>
> The original text follows unchanged.

Date: 2026-11-25. Every figure below was recomputed from the files on disk by a separate script that reads the master and the exports and doesn't call the build script. Nothing in the master, the exports or the splits was changed. **No model was trained, and nothing here measures model performance.**

## Result

The export is structurally consistent with the master: every count, ID and text matches. I found **5 defects**. Four are documentation or metadata defects and one is a missing guard for answer-generation pipelines (D1–D5 below). None of them requires changing the existing splits. There are also **two evaluation blockers** that come from the data itself: rare source levels (B1) and seven open author decisions (B2).

## 1. Held-out examples and master counts

**There are six held-out examples, not seven.** The "seven" in the previous report and in README ("Seven open items") came from adding 3 SME *examples* to 4 flagged audit *rows*. Those 4 rows belong to 3 examples (393 has two flagged rows).

| Master `training_use` | Count | Export outcome |
|---|---:|---|
| `usable` | 375 | 341 usable · 31 reclassified to label-only · 3 held out (158, 319, 393) |
| `label_and_generation_only` | 214 | 214 label-only |
| `exclude_redundant_duplicate` | 8 | excluded (156, 262, 502, 509, 517, 530, 578, 597) |
| `usable_pending_sme_check` | 3 | held out (429, 445, 477) |
| **Total** | **600** | 586 eligible (341 + 245) + 14 excluded = 600 |

- The eligible and excluded sets don't overlap, and together they cover all 600 IDs.
- The 586 eligible IDs are exactly the IDs in the three split files.
- The SHA-256 of each of the 5 master files matches the hash recorded in `integrity_report.json`.

## 2. Documentation vs. exported files

| Item | Claimed | Actual | Status |
|---|---|---|---|
| `eligibility_reclassifications.csv` | 31 moved, 3 false positives | 34 rows: 31 `usable → label_and_generation_only` and 3 `usable (kept)` (043, 428, 478). The moved set equals the master-usable/export-label set exactly | OK |
| `self_containment_audit.csv` | 152 rows | 152 rows across 79 examples; the keys match the Correction_Log self-containment rows exactly | OK |
| Audit verdicts | 148 preserve, 4 flagged | 148 PRESERVES. 4 FLAG: 158-Apply, 319-Apply, 393-Apply, 393-Create | OK |
| README "Seven open items" | 7 | 6 examples (3 SME + 3 audit-flagged) | **D1** |
| README "148 authored scenarios" in the data | 148 | 144 of the 148 PRESERVES rows are in the exports (143 usable, 1 label-only). The other 4 belong to excluded examples (156-Apply, 156-Analyze, 158-Analyze, 158-Create) | **D2** |
| README "the second pass wrote" these rows | pass 2 | Pass 2: 119 rows; passes 1+2: 31; pass 1 only: 2 | **D2** |
| README "many Evaluate rows state a wrong value on purpose" | "many" | 17 rows (section 6) | **D4** |
| Schema conformance | 7,618 records | 586 + 2,930 + 3,516 + 341 + 245 = 7,618 | OK |
| Groups, splits, leakage | 405 groups, largest 6, 130 multi-example; 323/40/42 groups; max Jaccard 0.364; 6 pairs ≥ 0.35 | Same | OK |

## 3. Usable examples: independent count

- **341 usable examples × 5 = 1,705 usable transformation records.** Every usable example has exactly 5 transformations.
- Each set of 5 covers exactly the five non-source levels, and none is empty.
- Every source level equals the master's `original_bloom_level`.
- Every source question and transformation is byte-identical to the master.
- All 2,930 transformation records have unique `record_id`s, match their parent example's text and eligibility, and none has a target level equal to its source level.
- Label-only: 245 × 5 = 1,225 records. The total is 2,930.

## 4. Can label-only examples enter answer-generation training?

**Not from the current files**, for two reasons:
- No exported record has an answer field. Transformation records contain only `source_question` and `target_question`, so there is no answer target to learn.
- No script in the repo reads `final_corrected_dataset/` or `training_ready/`. A search of `experiments/` found none.

**Missing guard (D3).** The split files mix both eligibility classes, and the existing loaders don't check eligibility. Those are `experiments/multitask_bloom_rewrite/scripts/dataset_io.py` and `train_*`, which expect `data_dir/<task>/<split>.jsonl`. If answers are added later, or the files are converted for those loaders, the label-only records would enter answer training unless the converter filters on `eligibility == "usable"`. The label-only records are correctly marked: all 245 have `answer_source_available: false` and a `missing_answer_source` block (214 `master_flag`, 31 `export_reclassification`). But nothing enforces the filter.

## 5. Level distributions

| Split | Source levels R/U/Ap/An/E/C (all) | Same, usable only | Target levels R/U/Ap/An/E/C (all) | Same, usable only |
|---|---|---|---|---|
| train (469) | 7 / 54 / 168 / 140 / 60 / 40 | 7 / 48 / 85 / 82 / 25 / 23 | 462 / 415 / 301 / 329 / 409 / 429 | 263 / 222 / 185 / 188 / 245 / 247 |
| validation (59) | **0** / 10 / 17 / 16 / 10 / 6 | **0** / 9 / 10 / 11 / **4** / **2** | 59 / 49 / 42 / 43 / 49 / 53 | 36 / 27 / 26 / 25 / 32 / 34 |
| test (58) | **0** / 11 / 21 / 18 / **4** / **4** | **0** / 9 / 9 / 14 / **1** / **2** | 58 / 47 / 37 / 40 / 54 / 54 | 35 / 26 / 26 / 21 / 34 / 33 |

- **Classification records:** balanced by construction. Every example contributes one question per level, so each split has n per level (469, 59 and 58).
- **Target levels:** every level appears in every split. The smallest usable count is 21 (Analyze in test).
- **B1 (blocker for evaluating by source level):**
  - Only 7 Remember source questions exist, all in train. Rewriting *from* Remember can't be evaluated at all.
  - Evaluate as a source has 1 usable test example and 4 usable validation examples. Create as a source has 2 in each.
  - Per-direction metrics for these sources are not meaningful.

## 6. Deliberately incorrect Evaluate claims

- **Framing is correct.** All 512 exported Evaluate rows ask the learner to judge something ("Assess whether…", "Verify whether…", "Evaluate the claim…"). None states a claim as fact.
- The 17 rows the Correction_Log documents as error-detection items (deliberately false claims) are all exported:
  - Usable, train: 027, 043, 064, 415, 436, 478, 483, 499, 514, 537, 548, 562, 593, 599.
  - Usable, test: 118, 442.
  - Label-only, train: 447.
  - Validation: none.
- Another 88 exported Evaluate rows have "answer-key note" log entries. The note for 415 overlaps with the list above.
- Because no answer field exists, a false claim can't be learned as a target answer from these files.
- **D4:** the export doesn't mark which Evaluate rows have a false premise. That information is only in `../Correction_Log.csv`. An answer-generation or reward pipeline would need it, and the README overstates the count ("many").

## 7. Are authored scenarios distinguishable?

**Only at the file level, not in the records (D5).**
- `self_containment_audit.csv` lists all 152 rows by `(example_id, target_level)`, but neither example records nor transformation records say which transformations contain authored data.
- `review_notes` appears only on 120, 170, 305 and 339.
- A user who reads only the JSONL files can't separate authored from source-derived content.

## 8. Redundant duplicates

All 8 are excluded and appear in no export. Every kept partner is eligible:

| Excluded | Kept partner | Partner eligibility | Partner split |
|---|---|---|---|
| 156 | 027 | usable | train |
| 262 | 187 | label_and_generation_only (reclassified) | train |
| 502 | 421 | usable | train |
| 509 | 472 | usable | test |
| 517 | 458 | usable | validation |
| 530 | 417 | usable | train |
| 578 | 512 | usable | validation |
| 597 | 457 | usable | train |

## Defects and the smallest reproducible fixes (proposed, not applied)

All five fixes are append-only changes to `scripts/make_training_ready.py`, `schema.json` and `README.md`. Split assignment uses only text and grouping, not these fields. After rebuilding, confirm that every split's ID list in `splits/split_manifest.json` is unchanged, and that every exported text is byte-identical apart from the new fields.

| ID | Defect | Proposed fix |
|---|---|---|
| D1 | "Seven" held-out items | README: "6 held-out examples (429, 445, 477, 158, 319, 393); 4 flagged rows". |
| D2 | Authored-row count and pass wording | README: "144 exported rows (143 usable, 1 label-only); 4 more are in excluded examples; 119 written in pass 2, 31 in passes 1+2, 2 in pass 1". |
| D3 | No eligibility guard for answer training | Also write `splits/answer_eligible/{split}_*.jsonl` containing only `usable` records (1,350 / 180 / 175 transformation records), and document that any answer pipeline must use them. |
| D4 | False-premise Evaluate rows not marked | Add `evaluate_premise: "false_claim"` to the 17 rows' transformation records and examples, read from Correction_Log. Fix the README count. |
| D5 | Authored content not marked per row | Add `authored_scenario_levels: [...]` to examples and `target_contains_authored_scenario: bool` to transformation records, both read from `self_containment_audit.csv`. Add the fields to `schema.json`. |

## Remaining blockers before training

- **B1:** evaluation by source level is impossible for Remember and unreliable for Evaluate and Create as sources (section 5). Report metrics by target level, or pool the rare source levels. Don't re-split silently.
- **B2:** 6 held-out examples await an author decision: 429, 445 and 477 (SME review); 158, 319 and 393 (flagged rows).
- **D3:** should be fixed before any answer-generation work. D1, D2, D4 and D5 are documentation or metadata fixes and don't block rewrite or classification training.
- Validation and test are small (59 and 58 examples), so any metric on them will have wide confidence intervals.
