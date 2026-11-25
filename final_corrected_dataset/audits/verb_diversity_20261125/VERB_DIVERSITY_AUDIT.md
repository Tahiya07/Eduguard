# Action-Verb Diversity and Cognitive-Alignment Audit — Corrected Bloom Dataset

Date: 2026-11-25. This audit is read-only: the master dataset, frozen splits and exports are unchanged, no GPU was used and no training was run.

- **Scope:** the frozen training-ready split files, which hold the 586 eligible examples and their 2,930 transformation rows (train 2,345 / validation 295 / test 290; 1,705 usable, 1,225 label-only). The 586 source questions are examined separately, because they are also Bloom-classification records.
- **Method:** `scripts/audit_verbs.py` (standard library only) extracts the instruction verb and writes every table cited here.
  - It finds the instruction sentence rather than taking the first word, so in "A student reports … Verify whether …" the verb is *verify*. It skips leading clauses such as "Given …, calculate …".
  - Verbs are grouped into 32 cognitive-operation families (30 occur in the data). Synonyms that ask for the same operation share a family (state/recall/name; compare/contrast), and distinct operations stay apart (define ≠ list; examine ≠ compare; verify ≠ judge).
  - "Determine whether" is separated from "determine" (compute).
- **Manual review:**
  - a seeded random sample of 72 rows (12 per level)
  - every heuristic flag outside Apply
  - samples from each pattern list below
  - all verb-swap candidates with similarity ≥ 0.9
  - the 8 most similar Understand/Analyze pairs

  This is **not** a full manual reading of all 2,930 rows.

## Summary

| Finding | Severity |
|---|---|
| The primary instruction verb alone predicts the target Bloom level with **98.3%** held-out accuracy (baseline 20%). The verb removes **97%** of label uncertainty. | **Critical** |
| Understand uses *Explain* in **93%** of rows (1.4 effective verbs); Analyze uses *Examine* in **89%** (1.75 effective verbs). | **Critical** |
| 9 of the 10 most frequent verbs occur at **exactly one** level. No high-volume verb is shared across levels. | **High** |
| Understand vs Analyze is often carried by the verb alone: "Explain how A, B determine X" vs "Examine how changes in A, B affect X" in the same example. | **High** |
| No question is interrogative: 2,918 of 2,930 are imperative, none ends in "?", and the other 12 have no detected instruction verb. | Medium |
| Whole-row templates dominate several levels. | Medium |
| 20 Remember rows state the answer inside the question. Smaller misalignment groups are listed in section 9. | Medium (specific defects) |
| Content preservation is generally good. Create and Apply rows are the most varied and cognitively genuine. | Positive |

The dataset's cognitive content is largely sound; its **wording is highly stereotyped**. A model trained on it can reach low loss by learning a fixed verb and template for each level instead of the cognitive operation. With the verb masked, a bag-of-words classifier still reaches **75%** (95% unmasked). So real cognitive cues exist beyond the verb, but so do template cues.

## 1. Verb frequency by target level (all 2,930 eligible rows)

Full tables: `verb_frequency_by_level.csv` and `family_frequency_by_level.csv`.

**Diversity metrics.** "Effective" counts are 2^entropy, the number of equally used verbs that would give the same diversity.

| Target level | Rows | Unique verbs | Unique families | Top-1 verb share | Top-3 verb share | Effective verbs | Effective families |
|---|---:|---:|---:|---:|---:|---:|---:|
| Remember | 579 | 10 | 5 | 36% | 94.5% | 3.67 | 3.44 |
| Understand | 511 | 7 | 5 | **93.0%** | 99.0% | **1.40** | **1.37** |
| Apply | 380 | 27 | 15 | 25% | 54.5% | 11.44 | 6.28 |
| Analyze | 412 | 12 | 8 | **88.6%** | 95.6% | **1.75** | **1.63** |
| Evaluate | 512 | 12 | 6 | 47% | 95.7% | 2.93 | 2.71 |
| Create | 536 | 14 | 6 | 49% | 92.7% | 3.51 | 3.22 |

**Most frequent verbs per level:**

| Target level | Most frequent verbs (count) |
|---|---|
| Remember | define 211, state 169, identify 167, list 24 |
| Understand | explain 475, interpret 18, describe 13 |
| Apply | calculate 96, apply 66, determine 45, classify 42, identify 29, assign 13, select 12, trace 11 |
| Analyze | examine 365, differentiate 19, distinguish 10, trace 6 |
| Evaluate | verify 242, assess 225, justify 23 |
| Create | design 265, develop 183, construct 49 |

The train split alone shows the same skew: top-1 share is 92.3% for Understand and 88.8% for Analyze. The usable-only subset is the same (93.5% and 85.9%).

**Rare verbs (used once):**
- Remember: recall, enumerate, write, distinguish
- Understand: clarify, summarize, predict
- Analyze: investigate, decompose, contrast, diagnose, interpret, determine
- Evaluate: appraise, challenge, judge, debug, determine whether
- Create: compose, devise, express, redesign, revise, write
- Apply: twelve verbs used once each

**Source questions** (the classification view) follow the same pattern:
- Analyze sources start with *analyze* 146/174 times.
- Evaluate sources start with *evaluate* 55/74 times.
- Apply sources start with *calculate* 100/206 times.

The verb alone predicts the source level for 94% of held-out source questions.

## 2. Cross-level verb distribution

Full matrices: `verb_level_matrix.csv` and `family_level_matrix.csv`.

**Verbs confined to a single level** (count):

| Level | Verbs |
|---|---|
| Understand | explain 475 |
| Analyze | examine 365 |
| Create | design 265, develop 183 |
| Evaluate | verify 242, assess 225, justify 23 |
| Remember | define 211, state 169, list 24 |
| Apply | calculate 96, apply 66, classify 42 |

**The only verbs used at more than one level**, all low-volume:

| Verb | Levels (count) |
|---|---|
| identify | Remember 167, Apply 29 |
| determine | Apply 45, Evaluate 3, Analyze 1 |
| construct | Apply 2, Create 49 |
| interpret | Understand 18, Apply 5, Analyze 1 |
| trace | Apply 11, Analyze 6 |
| formulate | Apply 1, Create 17 |
| distinguish | Remember 1, Analyze 10 |
| determine whether | Apply 7, Evaluate 1 |
| predict | Understand 1, Apply 4 |
| write / rewrite | Remember, Apply, Create (1–3 each) |

**Can a classifier predict the level from verbs alone? Yes, almost perfectly.** Held-out accuracy (trained on train, tested on validation + test, 585 rows):

| Predictor | Accuracy |
|---|---:|
| Majority class | 20.0% |
| First word only | 95.4% |
| Primary instruction verb only | **98.3%** |
| Verb family only | 98.8% |
| Bag-of-words (Naive Bayes), full text | 94.7% |
| Bag-of-words with the instruction verb masked | 75.2% |
| Classification view (sources + transformations), verb only | 97.9% |

H(level) = 2.57 bits, but H(level | verb) = 0.078 bits.

**Natural vs. artefactual association.** Some association is natural: *define* is retrieval, and *design* is construction. The problem is that the association is near-deterministic even where the cognitive operation doesn't require it:
- *Explain* can be Understand, Analyze ("explain why the anomaly occurs, given …") or Evaluate ("explain which option is preferable").
- *Identify* can be Remember or Analyze ("identify the flawed assumption").
- *Compare* can be Understand, Analyze or Evaluate.

The dataset never makes these distinctions with a shared verb, so the model never has to.

## 3. Unique verb and family counts

See the table in section 1. Across all levels there are 68 distinct primary verbs in 30 families; another 12 rows have no detected instruction verb. Only Apply has broad diversity (27 verbs, 15 families).

## 4. Rare and missing action families

These are operations natural to a level that are absent or nearly absent as the primary task. This is a gap list, not a quota.

| Level | Absent (0) or rare (≤ 2) |
|---|---|
| Remember | recognize / select from options (0), match term–definition (0), label (0), recall (1), name (2) |
| Understand | summarize (1), classify by concept (0), exemplify / give an example (0), compare at comprehension level (0), infer / qualitative prediction (1), paraphrase (0), interpret data / graph (most of the 18 interpret rows are physical-significance questions) |
| Apply | well covered; computation is 38% of rows |
| Analyze | compare / contrast (5), organize / structure (0), attribute / identify assumptions or viewpoints (0), diagnose / locate an error (1, plus some inside Examine rows), infer from data (0), decompose (3) |
| Evaluate | **recommend / select / rank / prioritize as the primary task (0)**, critique (9), defend / refute (2), weigh trade-offs (inside "assess" rows only). Verification makes up 47% of rows. |
| Create | plan (0), hypothesize (2 "conjecture"), compose / write an argument or brief (11), generate alternatives (0), synthesize / integrate (0), devise an experiment (rare) |
| All levels | interrogative questions ("Why …?", "What would happen if …?", "Which … and why?"): 0 |

## 5. Cognitive alignment findings

Most sampled rows ask for the cognitive work their label states. Create rows are a strength: they ask for genuine new artefacts with constraints (295, 110, 183, 313). The weaker groups are below.

| Pattern | Count | Assessment |
|---|---:|---|
| Remember rows that state the answer ("Identify A, B and C as the three …") | 20 (16 usable) | **Defect.** There is nothing to retrieve, e.g. 237 and 388 (more under misaligned examples below). |
| Remember rows that read back what the item specifies ("Identify the … specified for the supplied …") | 36 (8 usable) | Weak. This is reading, not recall (318, 247, 354). Mostly label-only rows. |
| Analyze rows that only identify a calculation's inputs ("Examine the supplied … to identify their contributions to the … calculation") | 7 (0 usable) | Weak. This is identifying parameters, closer to Remember/Understand (191, 207, 222, 287, 299, 367, 190). |
| Analyze rows built as "Examine … to identify …" | 109 (23 usable) | Mixed. Many are genuine (identify relationships or the source of unfairness); some are parameter identification. Needs a row-level review. |
| Analyze row that is procedure execution (trace stack push/pop and identify the contents) | 014 | Misaligned. This is Apply. |
| Understand rows with numeric data from an Apply source ("Explain how the runoff coefficient of 0.6 … contribute to the estimated peak runoff") | 75 (70 usable) | Acceptable. Explaining a relationship is Understand, but the numbers invite calculation. Review for drift into Apply. |
| Evaluate rows of the form "Verify whether a reported value is consistent …" | 183 (67 usable) | Legitimate checking (revised Bloom: Evaluate–checking), but many reduce to "recompute and compare" (539, 151, 219). Too dominant as Evaluate's operation. |
| Create rows that implement a standard program or algorithm | 15 matched; about 4 doubtful | 276 (read, sum, write a file) and 007 (iterative binary search) reproduce known solutions, closer to Apply. 021 (hybrid sort) and 126 (class hierarchy) are genuine design. |
| Source labels whose verb is "Analyze" but whose demand is explanation | 9 Analyze→Understand verb-swap pairs | 252 and 157: the source "Analyze how X does Y" and its Understand row "Explain how X does Y" ask the same thing. Either the source label or the transformation must change. **Changing a source label needs an author decision.** |
| Heuristic flags (Understand demanding higher work; Remember asking for explanation; Evaluate without a judgement cue) | 7 / 3 / 2 | Reviewed. Mostly benign compound tasks (e.g. 066 "Distinguish the definitions of correlation and causation" is borderline Understand). |
| Apply rows with no explicit concrete-case cue | 109 | Heuristic too broad. Sampled rows are genuine generic applications ("Given a …"). Not a defect class. |

## 6. Transformation and template diversity

Full candidate lists: `near_identical_transformation_pairs.csv`, `verb_swap_of_source_candidates.csv` and `low_source_overlap_candidates.csv`.

**Dominant templates:**

| Template | Rows | Share of level |
|---|---:|---:|
| "Explain how/why …" (Understand) | 432 / 511 | 85% |
| "Explain how ⟨factors⟩ determine / influence / affect ⟨outcome⟩" (Understand) | 202 / 511 | 40% |
| "Examine the supplied …" (Analyze, mostly label-only) | 118 / 412 | 29% |
| "Examine how / the relationship …" (Analyze) | 107 / 412 | 26% |
| "Verify whether …" / "Assess whether …" (Evaluate) | 240 / 194 of 512 | 85% combined |
| "Design a …" / "Develop a …" (Create) | 433 / 536 | 81% |
| Define / State / Identify openings (Remember) | 547 / 579 | 94% |

Across levels, the top five opening trigrams cover 59% of Evaluate rows, 51% of Analyze and 38% of Understand, but only 7–12% of Remember, Apply and Create.

**Verb-only distinctions:**
- Understand and Analyze rows in the same example share most of their content in 44 examples (Jaccard ≥ 0.5) and 18 examples (≥ 0.6). The worst cases differ only in "Explain how A determine X" vs "Examine how changes in A affect X": 127, 142, 149 and 144. 71 examples pair the "Explain how" and "Examine how/relationship" templates.
- 67 transformations share at least 80% of their source's content after the first word. Most are genuine level changes in a repeated frame: "Calculate X using A, B" becomes "Explain how A, B determine X" (142, 127, 149, 303, 332). True verb-only changes are 252 and 157 (Analyze→Understand).

**Distinctness within examples:** 65 transformation pairs inside the same example have body Jaccard ≥ 0.7. Read at random, these mostly differ correctly in cognitive demand (state the formula vs explain how the values combine) despite shared content, so they are not defects as a class.

**Topic alteration:** 137 rows keep under 15% of the source's content words. A random 10 were all on-topic; the vocabulary changes legitimately (e.g. Remember defining terms, Apply adding a concrete case). No topic changes were found in the sample.

## 7. Lexical shortcut analysis

**Findings:**
- Verb → level is near-deterministic: 98.3% held-out accuracy and 97% of uncertainty removed.
- Level → verb is narrow: H(verb | level) = 1.63 bits overall, and only 1.4 / 1.75 effective verbs for Understand / Analyze.
- For the rewriting model, the risk runs in the generation direction: given the target level, it learns to emit the stereotyped verb and template ("Explain how …", "Examine how …", "Verify whether …", "Design a …") whatever cognitive change the source needs.
- For a Bloom classifier trained on these records, the risk is direct label leakage through the opening word.

**Limitations of this analysis:**
1. Verb–level association is partly legitimate. High predictability alone does not prove leakage; the near-total absence of shared verbs and the verb-only Understand/Analyze pairs are what show the problem.
2. Extraction is rule-based. 12 rows have no detected verb, and secondary verbs are found heuristically.
3. The held-out set is small (585 rows) and grouped, not stratified by verb.
4. The masked-verb result (75%) shows other surface cues exist; not all of them are shortcuts.
5. No model was trained to test whether it exploits the shortcut. That needs a controlled experiment (see the plan).

## 8. Correctly aligned cross-level verbs (examples to imitate)

| Verb | Lower-level use | Higher-level use |
|---|---|---|
| identify | Remember 001: "Identify the normal forms used in relational database normalization." | Apply 026: "Given a source-code statement, identify its tokens, construct its syntax structure …" |
| determine | Apply 029: "… determine which replicas can continue serving read requests after one replica becomes unavailable." | Evaluate 071: "Determine whether non-preemptive SJF is appropriate for an interactive operating system where burst times are not known …" |
| trace | Apply 027: "trace its state transitions for the input string 1101." | Analyze 006: "Trace an instruction's execution and distinguish the roles of the ALU, registers, and control signals …" |
| interpret | Understand 061: "Interpret the physical significance of resonance …" | Apply 227: "Interpret the inflation and unemployment values in the supplied data to identify the position on the short-run Phillips curve." |
| construct | Apply 405: "Construct the precedence graph for the given schedule …" | Create 022: "Construct a VLSM plan for a university network with different host requirements …" |
| formulate | Apply 080: "formulate measurable indicators for both variables." | Create 049: "Formulate a hypothesis-testing procedure …" |
| predict | Understand 072: "Predict the general direction in which a reversible reaction shifts …" | Apply 047: "Given a population with different inherited traits and survival rates, predict how the frequency … may change." |

## 9. Potentially misaligned transformations (for review; not changed)

| ID / level | Issue |
|---|---|
| 237, 388, 451, 594, 589, 528, 587 (Remember) | Answer stated in the question ("Identify … as …"); 20 rows in total |
| 318, 247, 354 (Remember) | Read-back of item-specified information rather than recall |
| 191, 207, 222, 287, 299, 367, 190 (Analyze) | Identify the inputs of a calculation; not analysis |
| 014 (Analyze) | Trace push/pop operations and report the stack contents: procedure execution (Apply) |
| 252, 157 (source Analyze → Understand) | Same task with only the verb changed; the source label or the transformation needs a decision |
| 127, 142, 149, 144 (Understand vs Analyze) | "Explain how A determine X" vs "Examine how changes in A affect X": near-identical demand |
| 276, 007 (Create) | Reproduce a standard program or algorithm (closer to Apply) |
| 539, 151, 219 (Evaluate) | Recompute-and-compare verification; acceptable but formulaic |
| 066 (Remember) | "Distinguish the definitions of …" is closer to Understand |

## 10. Recommended corrections and expansion targets (proposal — nothing has been changed)

### Principles

- **Keep content fixed.** Keep each example's subject, data and constraints, and change only the wording that sets the cognitive task. Make no edits where a row is already aligned unless it serves a diversity target.
- **Diversify through the operation, not synonym swaps.** Replacing *Explain* with *Elucidate* adds no diversity. Each rewrite should change *what the learner does* where the level allows several operations.
- **No forced balance.** The targets below are ceilings on dominance, not quotas. A row stays as it is if the alternative would be awkward.
- **Preserve the frozen splits.** Rewrites change text, and the current build groups examples by text similarity. Any rebuild must assign splits **by example ID** from `training_ready/tests/reference_split_ids.json` rather than regrouping. The leakage check must then be rerun to confirm that no rewrite created a cross-split near-duplicate.
- **Process.** Work as pass 3:
  - Back up the master.
  - Log every change in `Correction_Log.csv` with a new pass number.
  - Rerun this audit and the existing training-ready tests and verifier.
  - Rewrite usable rows first (they feed answer generation), then label-only rows.

### A. Fix specific defects (about 60–80 rows; highest value per edit)

1. **Remember, answer in question (20 rows):** rewrite as genuine retrieval, e.g. 237 becomes "Name the three recognized sources of law."
2. **Remember read-back (36 rows):** replace with recall of domain knowledge tied to the same content.
3. **Analyze input identification (7) and procedure execution (014):** rewrite to require relationships, causes, structure or diagnosis.
4. **Verb-only Understand/Analyze pairs:** the 18 with Jaccard ≥ 0.6, plus the 9 Analyze→Understand source pairs. Rewrite the Analyze row to require decomposition the Understand row does not, e.g. "determine which factor dominates the break-even point when …". Flag 252 and 157 for an author decision on the source label.
5. **Create rows that reproduce known solutions** (about 4, e.g. 276, 007): add a design requirement that needs a new artefact, or keep them and record them as borderline.

### B. Reduce template dominance (about 600–800 rewrites across levels)

Indicative ceilings (no single verb or template above roughly 40–50% of a level):

| Level | Now | Proposed ceiling | Operations to introduce, where natural to the content |
|---|---|---|---|
| Understand (511) | Explain 93%; "Explain how … determine …" 40% | Explain ≤ 50% (about 255); that template ≤ 15% | interpret data or graphs, summarize, classify by concept, give or identify an example, compare at comprehension level, qualitative prediction, "Why does …?" |
| Analyze (412) | Examine 89% | Examine ≤ 40% | compare/contrast structures, differentiate relevant from irrelevant information, organize or outline structure, attribute assumptions or viewpoints, diagnose or locate an error, infer from data, "Which factor accounts for …?" |
| Evaluate (512) | Verify 47%; Assess 44% | Verify ≤ 30%; Assess ≤ 30% | critique against criteria, recommend or select among alternatives with justification, rank or prioritize, weigh a trade-off, defend or refute a claim, "Which … is more appropriate, and why?" |
| Create (536) | Design + Develop 84% | ≤ 55% combined | plan, formulate a hypothesis, compose an argument or brief, construct a model, proof or counter-example, generate alternative solutions, devise an experiment, redesign or improve an existing artefact |
| Remember (579) | Define / State / Identify 94.5% | ≤ 75% combined | list, name, recall, recognize or select from given options, match terms to definitions, label, "What is …?" |
| Apply (380) | already diverse | keep | reduce the "Classify / Match / Assign the supplied …" opening (39, mostly label-only) only where natural |

### C. Break the verb-to-level mapping with natural shared verbs

Write a deliberate subset of rows that use the same verb at different levels with different demands, as the section 8 examples already do:

| Verb | Levels and intended demand |
|---|---|
| Explain | Understand (meaning) · Analyze ("explain why the observed anomaly arises from …") · Evaluate ("explain which option better meets …") |
| Identify | Remember (a term) · Analyze (the flawed assumption or root cause) · Evaluate (the strongest objection) |
| Compare | Understand · Analyze · Evaluate ("compare … and recommend …") |
| Determine | Apply · Analyze · Evaluate ("determine whether … is justified") |
| Justify | Evaluate · Create ("propose … and justify the design") |
| Classify | Understand (by concept) · Apply (new cases) |

- **Indicative target:** about 15–20% of rows use a verb that has at least 10% presence at another level.
- **Diagnostic:** report the verb-only predictor accuracy after the change, and aim for it to fall well below its current 98%. It is a diagnostic, not an optimisation target: do not degrade natural phrasing to move it.

### D. Add question-form diversity

Convert a modest share, about 10–15%, to natural interrogative or conditional forms ("Why …?", "What would happen to … if …?", "Which … and why?"). Keep the cognitive level, and keep imperative exam phrasing where it reads better.

### E. Training and evaluation safeguards (independent of rewriting)

1. **Shortcut tests in evaluation:**
   - (a) a classifier trained on verb-masked text
   - (b) a test subset of shared-verb rows, where the verb doesn't reveal the level
   - (c) per-level opening-verb diversity of model outputs against the references
2. **Prompt review.** The existing BloomShift prompt lists preferred verbs per level and can inject a "Suggested opening action verb", which itself encourages a verb-to-level mapping. Consider dropping the suggested-verb line from evaluation prompts, or testing with and without it.
3. **Bloom classifier.** Train it with verb-masking or dropout augmentation, and report masked accuracy alongside normal accuracy.

### Scale and review

Parts A–D affect roughly 700–900 of the 2,930 rows. Expected effort: A first (small, objective defects); then B and C for usable rows (about 1,705 rows in scope); then label-only rows. Every rewritten row needs the same semantic check used in pass 2 (level genuinely achieved, content preserved, no invented data beyond the documented authored-scenario policy).

## Files in this audit

All files are in `final_corrected_dataset/audits/verb_diversity_20261125/`.

| File | Contents |
|---|---|
| `scripts/audit_verbs.py` | Reproducible extraction and analysis |
| `row_level_extraction.csv` | Every row: primary verb, family, phrase, secondary verbs, form, provenance |
| `verb_frequency_by_level.csv`, `family_frequency_by_level.csv` | Audit 1 |
| `verb_level_matrix.csv`, `family_level_matrix.csv` | Audit 2 |
| `alignment_candidate_flags.csv` | Heuristic candidates (Audit 3) |
| `near_identical_transformation_pairs.csv`, `verb_swap_of_source_candidates.csv`, `low_source_overlap_candidates.csv` | Audit 4 |
| `audit_results.json` | All metrics, including shortcut predictors and entropies (Audit 6) |
