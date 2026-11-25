# EduGuard Bloom transformation dataset v3.1: repair, revision and challenge-set evaluation

Date: 2026-11-25. Folder: `final_corrected_dataset/repaired_v3_20261125/`. The v3.0 report and manifest are kept in `reports/history/`.

**Status: not yet accepted.** The dataset passes every automated check, and the challenge set has been reviewed item by item. Acceptance still requires three things: a person resolves two contested gold labels, a person spot-checks a sample of the rewrites, and a trained model is scored on the challenge slices (see "Acceptance decision").

## How the work was done

| Kind of work | What it covered | Who/what did it |
|---|---|---|
| Model-assisted rewriting | Pass 3: all 586 eligible examples reviewed and rewritten where needed. Pass 3.1: 214 rows revised under the relaxed form rule and after the challenge-set review. Of the 2,930 transformations, 1,408 now differ from v2. | Claude, writing each row individually in `work/patches/` (series `b`, `s`, then `r`) |
| Model-assisted review | Challenge set: every item of all 58 test examples read, with attention to misleading-verb items and adjacent-level contrasts | Claude (`work/challenge_review.json`) |
| Automated checks | Format, structure, number preservation, preservation of false-claim values, unchanged master and v2 files, split pinning, schema, leakage, challenge disjointness, byte-identical rebuild; 37 unit tests | scripts in `scripts/`, tests in `training_ready/tests/` |
| Human verification | **None** | — |

## Pass 3.1: what changed in the rules (user directive)

- **The per-level verb-bank rule was removed.** Rows no longer have to open with a verb from their level's bank.
- **Any verb may be used at any level when the cognitive demand justifies it.** The level is set by the operation the learner must perform (revised Bloom: recall; interpret, classify, compare, explain; execute; differentiate, organize, attribute; check, critique; generate, plan, produce), not by the verb.
- **Imperative and interrogative forms are both allowed,** including wh-questions and judgement questions.
- **Wording is not changed to move classifier scores.** Predictability is reported only as a diagnostic.

Rules: `work/REPAIR_RULES.md`. The automated format check now rejects only empty, one-word or visibly truncated rows. Bloom alignment is checked by review, not by matching verbs.

### Revisions made under the new rule

- **165 rows where the bank verb had forced stilted wording** were rewritten with the natural verb or as a question. Examples:
  - "Substantiate or refute a reported Reynolds number …" → "Check a reported Reynolds number … and state whether it can be accepted."
  - "Reproduce the Darcy–Weisbach equation …" → "Write down the Darcy–Weisbach equation …"
  - "Architect an encryption scheme …" → "Propose an encryption scheme …"
  - "Run the DFA …" → "Simulate the DFA …"
- **Cross-level verbs are now used where they fit the demand:**
  - "Trace … the DNS resolution process" (Apply)
  - "Compare the selection procedures …" (Understand: comparing meanings)
  - "Modify binary search so that …" and "Find a closed-form formula … and prove it" (Create)
  - "Find the production conflicts … and trace each" (Analyze)
- **Questions are spread across levels, so a question mark does not mark one level.** Revising the Evaluate rows produced 56 judgement questions. To stop "?" becoming an Evaluate marker, 41 rows at other levels where a question is the natural phrasing were also converted, for example "What is the therapeutic index of a drug whose …?" (Apply) and "Why does a 4-bit Johnson counter cycle through 8 states …?" (Analyze). A unit test now checks that every level contains interrogative rows.

  | Level | Interrogative rows |
  |---|---|
  | Remember | 14 |
  | Understand | 10 |
  | Apply | 12 |
  | Analyze | 13 |
  | Evaluate | 56 |
  | Create | 3 |

- **One row (426 Create) was reverted** to its v2 wording, which was already natural.
- **9 rows were fixed after the challenge-set review** (next section).

Content rules are unchanged: no new numbers or facts, every value in a false-claim row kept, and source questions identical to v2.

## Challenge-set evaluation

The challenge set (`challenge_set/`) contains 348 items and 406 same-content adjacent-level pairs from the 58 test examples. It is disjoint from train and validation by ID, group and text.

### 1. Item review (model-assisted)

| Verdict | Items |
|---|---|
| Reviewed, gold label and level contrast correct | 340 |
| Weak distinction, kept and noted (520-Understand, 474-Understand, 472-Understand, 306-source) | 4 |
| **Gold label contested; excluded from scoring until resolved** | 4: 213-source/213-Apply and 570-source/570-Apply |

- **Misleading-verb items.** In these items the leading verb points to a different level than the gold label, for example "Classify … under IaaS/PaaS/SaaS" (Understand) and "Determine the likely connectivity problem … using forward chaining" (Apply). Apart from the 213 and 570 Apply rows, every one carries the correct gold label, so they are genuine tests rather than labelling errors.
- **Contested labels.**
  - 213: the source is labelled Analyze but asks only to determine which overridden method executes.
  - 570: the source is labelled Analyze but asks only for the SCAN service order.

  Both tasks execute a known rule, which is Apply, and each duplicates its example's Apply row. The source labels are frozen and changing them is a subject-matter decision, so they were flagged, not changed. The 4 items and the 10 pairs that involve them carry `score_this_item` / `score_this_pair = false`.
- **Rows fixed because adjacent levels asked for nearly the same thing:**
  - 147-Apply and 147-Analyze: the Apply row repeated the Understand source, and the Analyze row only classified coefficients.
  - 392-Apply: repeated the Evaluate source.
  - 093, 488, 501, 585 and 056 Understand: each restated the Analyze source with a different verb.
  - 342-Understand: too close to the Analyze row.
- **Cross-row answer leakage, noted.** 118-Create states the correct 6 ns access time, which answers 118-Apply and the false claim in 118-Evaluate. This is harmless for label prediction, but avoid scoring answers across rows of one example.

### 2. Shortcut baselines on the challenge slices (automated)

Simple predictors were fitted on the train split and scored on the 344 scorable items (`scripts/evaluate_challenge_set.py`, `audits/challenge_baselines.json`).

| Slice | n | Verb only | First word | NB, verb masked | NB, full text |
|---|---|---|---|---|---|
| All scorable items | 344 | 0.919 | 0.907 | 0.718 | 0.884 |
| No difficulty tag | 166 | 1.000 | 1.000 | 0.928 | 0.976 |
| **`verb_shortcut_misleads`** | 26 | **0.385** | 0.423 | 0.654 | 0.769 |
| **`verb_multi_level_in_train`** | 47 | **0.596** | 0.596 | 0.787 | 0.851 |
| `rare_or_unseen_verb` | 24 | 0.500 | 0.500 | 0.625 | 0.667 |
| `shared_bank_verb` | 45 | 0.667 | 0.667 | 0.889 | 0.978 |
| Interrogative | 11 | 0.636 | 0.727 | 0.636 | 0.727 |
| Scenario first | 10 | 1.000 | 0.700 | 0.600 | 0.900 |

Adjacent-level pairs, counting a pair correct only if both levels are right:

| Predictor | Both correct | Weakest pair |
|---|---|---|
| Verb only | 0.795 | Understand–Analyze, 0.696 |
| NB, verb masked | 0.513 | Understand–Apply, 0.357 |
| NB, full text | 0.747 | Understand–Analyze, 0.643 |

**What the evaluation shows.**

- **The slices separate shortcut use from understanding.** A verb lookup is perfect on untagged items but drops to 0.39 on misleading-verb items and 0.60 on multi-level-verb items.
- **The baselines alone cannot meet the acceptance criterion.** A model that recognises the operation from the whole question should score about as well on those slices as on the untagged items; a model that relies on the verb will show the same gap as the verb baseline. The meaningful comparison is between slices, not the overall score.
- **Understand–Analyze is the hardest contrast for every baseline.** It is the pair to watch in model evaluation.
- **Limitations.**
  - The slices are small (10–47 items), so a ±10-point difference is within noise.
  - Only 1 of 406 pairs keeps the same leading verb, so the pairs mostly test content cues, not contrasts with the verb held constant. Same-verb contrast probes built from test-split content would strengthen this test. They have not been added.

## Dataset diagnostics

| Predictor (train → validation + test) | v2 | v3.0 | v3.1 |
|---|---|---|---|
| Leading verb only | 0.974 | 0.949 | **0.930** |
| First word only | 0.954 | 0.939 | 0.918 |
| NB, verb masked | 0.749 | 0.723 | 0.716 |
| NB, full text | 0.947 | 0.906 | 0.903 |
| H(level \| verb), bits | 0.078 | 0.136 | 0.166 |
| H(verb \| level), bits | 1.63 | 2.95 | 2.97 |
| Held-out rows whose verb occurs at 2+ levels in train | 12.8% | 14.0% | 16.9% |

Verb diversity within each level (extended audit instrument):

| Level | Top-verb share v2 → v3.1 | Effective number of verbs v2 → v3.1 | Top-5 opening trigram share v2 → v3.1 |
|---|---|---|---|
| Remember | 0.36 → 0.28 | 3.7 → 6.7 | 0.08 → 0.08 |
| Understand | 0.93 → 0.49 | 1.4 → 5.1 | 0.38 → 0.25 |
| Apply | 0.25 → 0.24 | 11.4 → 8.4 | 0.12 → 0.09 |
| Analyze | 0.89 → 0.28 | 1.8 → 10.9 | 0.51 → 0.23 |
| Evaluate | 0.47 → 0.25 | 2.9 → 11.9 | 0.59 → 0.32 |
| Create | 0.49 → 0.43 | 3.5 → 7.2 | 0.07 → 0.06 |

The verb still predicts the level about 93% of the time on held-out rows. Most remaining verbs are used at only one level because that is how teachers phrase those tasks: "Design" for Create, "Define" for Remember. The rule now permits cross-level verbs where the demand justifies them, but rows were not reworded to force them. That is why the challenge slices, not the dataset statistics, decide acceptance.

The audit heuristics still flag some rows: 15 Evaluate rows without a judgement cue, 7 Create rows without a new-artefact verb, and 8 Understand rows that may demand higher-level work. These rows were read. The flags come from the heuristics' fixed cue lists not recognising the cross-level verbs and question forms the new rule allows, such as "Prioritize … justifying the order", "Modify binary search so that …" and "Compare the selection procedures …".

## Automated validation (v3.1)

| Check | Result |
|---|---|
| 586 eligible examples; 5 transformations each = all levels except the source; no duplicates | pass |
| Format (non-empty, not truncated; imperative or interrogative) | pass |
| Interrogative rows present at every level | pass |
| No new numbers (6 reviewed constants allowlisted) | pass |
| False-claim values preserved (59 rows) | pass |
| Source questions identical to v2; only label change is 252 | pass |
| Authoritative master and v2 export unchanged (SHA-256) | pass |
| Splits identical to `reference_split_ids.json`; groups, eligibility and markers unchanged | pass |
| Records: transformation 2,345/295/290; answer-generation 1,350/180/175; classification 2,814/354/348 | pass |
| Leakage: ID, group and identical-text overlap | 0 / 0 / 0 |
| Schema, all 9,323 records | pass |
| Challenge set: test only, disjoint, hashes match | pass |
| Rebuild byte-identical | pass |
| Unit tests | 37/37 pass |

## Acceptance decision

The dataset is ready for human spot-checking and for a model evaluation run. It should not be called accepted until:

1. A subject-matter expert resolves the 213 and 570 source labels (Analyze or Apply). If they change, the Apply rows for those examples need new content at the vacated level.
2. A person spot-checks a stratified sample of about 10% of the 1,408 rewritten rows, plus 252, 157 and the rows flagged above.
3. A trained classifier is scored on the challenge set, and its accuracy on `verb_shortcut_misleads`, `verb_multi_level_in_train` and `rare_or_unseen_verb`, and on the Understand–Analyze pairs, is close to its accuracy on untagged items. I did not train any model, as instructed.

Optional: add same-verb contrast probes to the challenge set so that pair evaluation can hold the verb constant.

## Exclusions

There are no new exclusions. The 14 example exclusions are unchanged, plus the one 252 row removed after relabelling (`EXCLUSION_REGISTER_v3.csv`).

## Reproducing

From `repaired_v3_20261125/`:

```
python -I scripts/apply_repairs.py --write
python -I scripts/make_training_ready_v3.py
python -I training_ready/scripts/check_leakage.py
python -I scripts/build_challenge_set.py
python -I scripts/evaluate_challenge_set.py
python -I scripts/split_integrity_v3.py
python -I scripts/audit_verbs_v3.py training_ready audits/verb_diversity_v3/v3_extended_lexicon
python -I -m unittest discover -s training_ready/tests -v
python -I scripts/write_manifest_v3.py
```
