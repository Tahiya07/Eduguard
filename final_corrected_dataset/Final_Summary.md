# EduGuard 600-Example Dataset — Final Correction Summary

Source: the uploaded PDF, *EduGuard: Federated Sub-Billion Bloom Classification and Multitask Academic Generation for Privacy-Aware University Assistance* (449 pages). The source file was read, not modified. Two correction passes were made: a first pass (structure, labels, Create strengthening, technical fixes) and a second semantic quality-control pass that re-read all 600 examples and 3,000 transformations. The first-pass deliverables are preserved in `backup_pass1_2026-11-25/`.

## What the two passes did

| Measure | Count |
|---|---:|
| Examples checked (pass 2, read in 10 batches of 60) | 600 |
| Transformations checked | 3000 |
| Original Bloom labels corrected (both passes) | 6 |
| Transformation fields that differ from the source | 713 |
| — changed in pass 1 | 373 |
| — changed in pass 2 | 373 |
| — pass-1 Create rewrites revised again in pass 2 | 32 |
| Transformations identical to the source | 2287 |
| Numeric items recalculated, wording unchanged (answer-key notes in the log) | 93 |
| Rows in Correction_Log.csv (one per changed field, source → final, plus check notes) | 831 |
| Rows in Unresolved_Questions.csv | 250 |

Pass-2 changes by type (transformation fields):

| Type | Fields |
|---|---:|
| Row cited 'supplied' material that the generic source never had; now self-contained | 150 |
| Evaluate row now states the object being judged and the criterion | 127 |
| Apply row required only recognition; now requires applying a rule or method | 56 |
| Create row restated a concept or calculation; now a construction task | 20 |
| Row repeated the original question (or another row); replaced | 9 |
| Analyze row now states what is examined | 5 |
| Row removed/added after a relabel | 4 |
| Technical error corrected | 2 |

## Bloom label corrections

| Example | Before | After | Reason |
|---|---|---|---|
| 090 | Remember | Understand | Classifying materials by their physical and mechanical properties and by their suitability for applications requires grouping instances under categories using understood properties (Understand: classifying), not retrieval of facts. |
| 152 | Remember | Understand | Identifying the addressing mode of each instruction in an unseen code sequence requires recognising each operand's form and assigning it to a mode category (Understand: classifying), not recall of a memorised fact. |
| 182 | Create | Apply | Deriving the output waveform of a standard full-wave rectifier for a given input executes a known procedure (Apply); it does not produce a new design or product. |
| 226 | Create | Apply | Proving a given identity by mathematical induction executes the standard induction procedure (Apply); the parallel proof in Example 044 is labelled Apply, so the label is corrected for consistency. |
| 068 | Create | Apply | Proving the product rule from the limit definition reproduces a standard textbook derivation (Apply), consistent with the relabelling of the induction proof in Example 226 and the waveform derivation in Example 182; it does not produce a new result. |
| 241 | Create | Apply | Deriving e = −N dΦ/dt for an N-turn coil applies Faraday's law to N identical turns in series, a standard derivation (Apply), consistent with Examples 068, 182, and 226; no new product is created. |

For each relabel, the row at the new source level was removed and a row at the former level was added, so every example still covers the five other levels. Distribution of source labels, before → after: Remember 9 → 7, Understand 74 → 76, Apply 212 → 216, Analyze 175 → 175, Evaluate 75 → 75, Create 55 → 51.

## Pass 2: what was found and fixed

- **Evaluate rows that judged an object never shown.** Many rows asked learners to assess "a proposed design/proof/DFA/allocation" that does not appear anywhere. Where the source question is generic, the object and the criterion are now written into the row (for example a specific faulty DFA, a stated wrong efficiency, a described library layout). Several are deliberately flawed so that the row works as an error-detection task; the correct answer is recorded in the log reason. The pass-1 generic suffix "Justify your judgement with explicit reference to the criteria used" was withdrawn: it did not make an under-specified item valid, and the rows now name their criteria directly.
- **Transformations that cited data the source never had.** Many Remember/Understand/Analyze originals are generic (no data), yet their Apply/Analyze/Evaluate/Create rows referred to "the supplied signal/matrix/agreement". These rows now carry their own small, internally consistent data (every value was recalculated independently). This is question authoring for a generic item, not reconstruction of missing source data.
- **Apply rows that only required recognition.** Rows whose answer was printed in the question ("classify X as Y and Z, respectively") or that only labelled named items were rewritten to require applying a rule, method, or formula (for example the Spearman–Brown formula, the slow-start rule, Stokes' law, proportionate stratified sampling). Classification rows that apply stated criteria to new data (for example measurements against limits) were kept as Apply.
- **Rows that repeated the original question.** Several Apply/Evaluate/Create rows were the original task restated (for example 405, 413, 414, 428, 454, 456, 466, 535, 552); each now asks something different at its level.
- **Create rows re-checked.** All Create rows were re-read. Pass-1 rewrites that leaned on non-existent "supplied" data were made self-contained, and remaining "develop a framework distinguishing …" rows that only restated a comparison were replaced with design or planning tasks. Create rows that ask the learner to set a target ("a chosen …") keep the original objective; they do not add constraints the source contradicts.
- **Technical corrections in pass 2.** 573: the reported discharge was ≈0.038 m³/s, but recalculation gives 0.0389 m³/s; corrected to 0.039. 477: the pass-1 note wrongly said both paths cost 7; S–B–D–G costs 9, so S–A–C–G (cost 7) is the unique optimum, and the heuristic remains inadmissible at A and C. 442: by total sales the top region is East (Tk. 275,000), not North (Tk. 270,000), and the Evaluate row is now documented as an error-detection item. 429: MAX nodes sit under a MAX root as written (root value 9), flagged for the author. 453: the row cited unspecified "plasticity criteria" and now names Burmister's ranges.
- **Labels.** 068 (prove the product rule) and 241 (derive e = −N dΦ/dt) were relabelled Create → Apply, consistent with the pass-1 relabels of 182 and 226. All other source labels were re-checked against the cognitive operation required and kept.

## Technical checks

Every item with numbers in the source (Examples 405–600 and the numeric items before them) was recalculated, as was every value introduced in pass 2 (50 values, all confirmed by an independent script). Results are in Correction_Log.csv under "Technical check – no change". Items whose Evaluate row deliberately states a wrong value or claim for error detection (the correct answer is in the log): 027, 043, 064, 076, 100, 103, 116, 118, 124, 128, 132, 147, 156, 159, 165, 169, 172, 178, 192, 211, 221, 226, 231, 234, 235, 243, 244, 250, 264, 268, 271, 272, 279, 282, 284, 285, 295, 297, 305, 309, 310, 327, 339, 347, 361, 374, 390, 393, 401, 405, 415, 429, 436, 442, 447, 478, 483, 496, 499, 506, 514, 537, 548, 562, 593, 599 (66 items). Pass-1 corrections to 445, 461, 417 and 530 were re-verified (FIRST/FOLLOW sets correct for the grammar, NPV ≈ Tk. 225.39, regulation 5.00% vs 5.26% by convention).

## Duplicate pairs

All eight suspected pairs test the same learning objective with the same data at the same Bloom level; none is merely a shared topic. Both members stay in the 600-example file, but the second member of each pair is flagged `exclude_redundant_duplicate` in `corrected_dataset.json` so it can be left out of training and evaluation splits.

| Keep | Redundant | Note |
|---|---|---|
| 027 | 156 | Same DFA task (binary strings ending in 01), same level (Create). |
| 187 | 262 | Same straight-line depreciation task with unspecified figures, same level (Apply). |
| 417 | 530 | Same transformer data (240 V / 228 V), same level (Apply); reported regulation conventions differed and are now stated. |
| 421 | 502 | Identical matrix inverse task, A=[[2, 1], [5, 3]], same level (Apply). |
| 457 | 597 | Same D-value task (10^6 to 10^3 in 6 minutes), same level (Apply). |
| 458 | 517 | Same Newton's law of cooling task (90°C, 30°C, k=0.1 min⁻¹, 10 min), same level (Apply). |
| 472 | 509 | Same IRR task (Tk. 10,000 outlay, Tk. 12,100 after two years), same level (Apply). |
| 512 | 578 | Same 4-bit Johnson counter from 0000, same level (Apply). |

Near-duplicates (different values, level, or scope) are kept as distinct items: 437/521; 437/582; 408/481; 497/570; 549/599; 127/246; 543/595; 547/592; 546/600; 513/587; 545/596; 319/441 (Partition of Bengal at Evaluate and Analyze).

## What remains unresolved

| Type | Rows |
|---|---:|
| Genuine missing source material | 215 |
| Generic placeholder (no data missing) | 24 |
| Redundant duplicate | 8 |
| Needs SME confirmation | 3 |

- **Genuine missing source material (215 examples; 214 carry the `label_and_generation_only` flag and the remaining one, 262, is flagged as a redundant duplicate).** The source question itself refers to data that is not in the PDF ("the supplied matrix", "the following resource allocation state", "a given block diagram"). Nothing in the item allows it to be reconstructed, so it was not invented. These items keep valid Bloom labels and question wording but cannot be answered as written.
- **Generic placeholders (24).** Questions that say "a specified load" or "a specified point" are generic and answerable in general or symbolic terms; nothing is missing. They are listed only for transparency.
- **Needs SME confirmation (3).** 429 (tree structure), 445 (filled-in sets), 477 (inadmissible heuristic).
- **Redundant duplicates (8).** See above.

## Training-use flags (in corrected_dataset.json)

| training_use | Examples | Meaning |
|---|---:|---|
| `usable` | 375 | Self-contained and technically checked |
| `label_and_generation_only` | 214 | Bloom label and question text are valid; the source data needed to answer it is missing, so do not use it for answer generation or answer checking |
| `usable_pending_sme_check` | 3 | Usable; one author question is open |
| `exclude_redundant_duplicate` | 8 | Same objective, data, and level as another example; exclude from training/evaluation splits |

**Examples that should not be used for training as complete question–answer items:** the 8 redundant duplicates and the source-dependent items flagged `label_and_generation_only` (listed in Unresolved_Questions.csv). No example contains a known uncorrected calculation error; three items (429, 445, 477) await author confirmation.

## Method and limits

- Labels were judged by the cognitive operation required, not by the leading verb. Derivations and proofs of standard results are treated as Apply; classifying new cases by stated criteria is Apply; recognising labels already given is not.
- Pass 2 was a manual reading of every example by one reviewer, supported by scripts for detection, recalculation, and consistency. It is not a substitute for subject-specialist review in law (jurisdiction-specific rules were avoided) and pharmacy.
- The checklist below covers structure and consistency only. It does not show semantic validity; that rests on the manual review described above and the reasons recorded per change in Correction_Log.csv.

## Structural checklist

- [x] 600 examples retained in the file (none deleted)
- [x] Every example has exactly 5 transformations
- [x] The 5 target levels are distinct and exclude the (corrected) source level
- [x] No empty transformation or question; no transformation identical to its original question
- [x] corrected_dataset.json parses and has 600 records
- [x] PDF and JSON wording identical (every JSON question and transformation found verbatim in the PDF text layer)
- [x] Source PDF unchanged (SHA-256 abca692a079b4378…, 5,068,376 bytes)
- [x] No residual LaTeX commands
