# EduGuard 600-Example Dataset — Final Correction Summary

Source: the uploaded PDF, *EduGuard: Federated Sub-Billion Bloom Classification and Multitask Academic Generation for Privacy-Aware University Assistance* (449 pages). The source file was read but not modified. The companion DOCX with the same title was used only to recover table structure and equation markup; every question was checked against the PDF text.

## Counts

| Measure | Count |
|---|---:|
| Examples reviewed | 600 |
| Examples retained | 600 (none excluded) |
| Transformations reviewed | 3000 |
| Original Bloom labels corrected | 4 |
| Transformations changed (any reason) | 826 |
| — Create rows strengthened into genuine construction tasks | 340 |
| — Evaluate rows given an explicit justification requirement | 461 |
| — Rows rewritten or added for level coverage after a relabel | 4 added, 4 removed |
| — Notation-only changes (LaTeX or flattened equations to plain notation) | 28 fields |
| Transformations unchanged | 2174 |
| Exact or functional duplicate pairs (documented, both kept) | 8 |
| Near-duplicate pairs noted (distinct values or levels, kept) | 11 |
| Unresolved items (rows in Unresolved_Questions.csv) | 320 across 291 examples |
| Structural fixes | 4 |
| Technical fixes (errors, ambiguity, missing content, self-containment) | 6 |
| Technical answer-key notes (value checked, wording unchanged) | 20 |

## Bloom label corrections

| Example | Before | After | Reason |
|---|---|---|---|
| 090 | Remember | Understand | Classifying materials by their physical and mechanical properties and by their suitability for applications requires grouping instances under categories using understood properties (Understand: classifying), not retrieval of facts. |
| 152 | Remember | Understand | Identifying the addressing mode of each instruction in an unseen code sequence requires recognising each operand's form and assigning it to a mode category (Understand: classifying), not recall of a memorised fact. |
| 182 | Create | Apply | Deriving the output waveform of a standard full-wave rectifier for a given input executes a known procedure (Apply); it does not produce a new design or product. |
| 226 | Create | Apply | Proving a given identity by mathematical induction executes the standard induction procedure (Apply); the parallel proof in Example 044 is labelled Apply, so the label is corrected for consistency. |

For each relabelled example, the transformation at the new source level was removed and one at the former source level was added, so each example still covers the five other levels. In 152 the Apply and Create rows also repeated the original task and were replaced; in 044 the Create row duplicated Example 226's original question and was replaced.

Distribution of source labels, before → after: Remember 9 → 7, Understand 74 → 76, Apply 212 → 214, Analyze 175 → 175, Evaluate 75 → 75, Create 55 → 53.

## What changed and why

- **Create rows (340).** Many Create rows asked for a "complete calculation", "derivation", "table", "worksheet", "trace", "statement", or a standard "representation"/"model" of the same problem. That is Apply or Understand work. Each was rewritten as real construction: a design, plan, modified scenario, generalisation, new test input, or proposal. The concept, scenario, numbers, units, and constraints are kept. Where a design needs a target value the source does not give, the learner is asked to set and justify it ("a chosen …") rather than the editor making one up.
- **Evaluate rows (461).** Rows that asked for a judgement or verification without asking for justification now end with "Justify your judgement with explicit reference to the criteria used." The task itself is unchanged.
- **Technical.** Example 461 treated an NPV of ≈Tk. 824 as given, but the correct value is ≈Tk. 225.39; corrected. In Example 445 the proposed FIRST/FOLLOW sets were empty in the source and have been filled from the grammar (flagged for confirmation). Examples 417 and 530 use the same transformer data but report 5% and 5.26%; the regulation convention for each is now stated. Examples 446 and 462 referred to "proposed" values that were never given and are now self-contained.
- **Answer-key notes, wording unchanged.** Eleven "verify whether the reported value is correct" items state a wrong value (415, 436, 447, 478, 483, 499, 514, 537, 548, 562, 599). They work as error-detection Evaluate tasks, so the wording is kept and the correct value is recorded in the log. Nine items state a correct value (407, 411, 440, 457, 525, 533, 551, 555, 560), also confirmed in the log.
- **Structural.** Example 105's Create row, which sat outside its table, was restored as the fifth row and then strengthened. The "A*" titles of 366 and 477 were repaired. Private-use glyph markers were removed from 465's SQL code block, which is set in monospace. Embedded data tables in 428, 442, 475, 477 and 548 are kept, as are the multi-line data in 405, 429, 445, 465, 502 and 569 and the discipline section headings between Examples 080 and 093.
- **Notation.** Equation markup that had flattened (for example `n=1∞n3n`, `01xexdx`) is rendered in consistent linear notation (`∑_(n=1)^∞ n/(3^n)`, `∫_0^1 x e^x dx`). LaTeX fragments in 072, 419, 421, 445, 502, 525, 569, 571 and 591 are converted to the same notation, with matrices written as `[[2, 1], [5, 3]]`.

## Duplicates

No examples were removed. The following pairs ask the same task at the same level with the same data. Both are kept, and their rewritten rows were differentiated where possible. Each pair is listed in Unresolved_Questions.csv for a merge-or-replace decision.

| Pair | Note |
|---|---|
| 027 / 156 | Same DFA task (binary strings ending in 01), same level (Create). |
| 187 / 262 | Same straight-line depreciation task with unspecified figures, same level (Apply). |
| 417 / 530 | Same transformer data (240 V / 228 V), same level (Apply); reported regulation conventions differed and are now stated. |
| 421 / 502 | Identical matrix inverse task, A=[[2, 1], [5, 3]], same level (Apply). |
| 457 / 597 | Same D-value task (10^6 to 10^3 in 6 minutes), same level (Apply). |
| 458 / 517 | Same Newton's law of cooling task (90°C, 30°C, k=0.1 min⁻¹, 10 min), same level (Apply). |
| 472 / 509 | Same IRR task (Tk. 10,000 outlay, Tk. 12,100 after two years), same level (Apply). |
| 512 / 578 | Same 4-bit Johnson counter from 0000, same level (Apply). |

Near-duplicates, which differ in values or level and are kept as distinct items:

- 437 / 521: Fluid balance, 8-hour shift, different output values (same net +200 mL).
- 437 / 582: Fluid balance, different period and values.
- 408 / 481: Price elasticity, different prices and method (midpoint vs percentage change).
- 497 / 570: Same SCAN disk configuration at different levels (Apply vs Analyze); 570 adds a request at cylinder 199.
- 549 / 599: Consonant classification with different sound sets.
- 127 / 246: Generic break-even analysis with unspecified figures (see also 406, which has figures).
- 543 / 595: Bearing-capacity conceptual items with overlapping content.
- 547 / 592: Primary vs secondary succession items with overlapping content.
- 546 / 600: Cache write-policy items with overlapping content.
- 513 / 587: Chloride limit test items with overlapping content.
- 545 / 596: Working-memory items with overlapping content.

## Unresolved items (need human input)

| Issue | Rows |
|---|---:|
| External material referenced but not included | 237 |
| Transformation evaluates an unstated artefact | 73 |
| Exact/functional duplicate retained | 8 |
| Technical data concern (SME review) | 1 |
| Restored content requires confirmation | 1 |

The largest group is source questions that refer to material not in the dataset ("the supplied matrix", "the following resource allocation state", "the specified circuit"). That material cannot be recovered from the source, and inventing it is outside the brief, so those items keep their references and are listed for an author to supply the data. A smaller group is transformation rows that evaluate an artefact the item never shows ("a proposed testing strategy", "a supplied C program"). Example 477's heuristic is not admissible (h(A)=6 against a true remaining cost of 5, h(C)=4 against 3); the data is unchanged and flagged for review.

## Method notes

- Labels were judged by the cognitive operation each question requires, not by its leading verb.
- Dataset convention kept: many Apply rows ask the learner to *classify, match, or identify* supplied new cases using a known scheme. That is treated as carrying out a procedure on unfamiliar instances (Apply), with the Understand row asking for explanation. The convention is used consistently, so these rows were left as they are. A reviewer who wants strict revised-taxonomy usage, where *classifying* belongs to Understand, should review Apply rows beginning "Classify", "Match", or "Identify" (about 160 rows).
- Every change is in Correction_Log.csv with its original wording and reason. Wording in EduGuard_600_Corrected.pdf and corrected_dataset.json is generated from the same records.

## Final validation checklist

- [x] 600 examples retained (none excluded)
- [x] Every example has exactly 5 transformations
- [x] The 5 target levels are distinct and exclude the source level
- [x] No empty transformation or question
- [x] corrected_dataset.json parses and has 600 records
- [x] Document and JSON wording identical (every JSON question and transformation found verbatim in the PDF text layer)
- [x] Source PDF unchanged (SHA-256 abca692a079b4378…, 5,068,376 bytes)
- [x] No residual LaTeX commands
