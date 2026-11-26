# EduGuard Bloom rewrite dataset, v3.2

This version corrects v3.1 and extends it so the model learns to rewrite a question **from any Bloom level to any other**, keeping its topic and content. It is attached to training as `data/multitask_bloom_rewrite_separate_eduguard_v3_7`, with config `qwen15b_multitask_eduguard_v3_7_20261126.json`.

- **Built by:** `experiments/multitask_bloom_rewrite/scripts/build_bloom_v3_2.py`.
- **Corrections:** `corrections_v3_2.json`, also in `experiments/multitask_bloom_rewrite/data_corrections/`.
- **Counts and hashes:** `dataset_manifest.json`.

| Split | Rows | Direct rows (v3.1) | Revised direct rows | Pivot rows (new) | Source questions |
|---|---:|---:|---:|---:|---:|
| Train | 2,735 | 2,345 | 258 | 390 | 469 |
| Validation | 341 | 295 | 36 | 46 | 59 |
| Test | 361 (351 used) | 290 | 36 | 71 | 58 |

The 10 rows with contested labels (examples 213 and 570, in test) are kept in the source file but excluded from training and scoring, as in v3.1.

## What changed from v3.1

1. **Rewrites that referred to material the student never gets** (215 rows, all splits). For example, "for a given resource allocation graph", or "using the supplied circuit parameters" when the question supplied none. Each was reviewed and rewritten at the same level and on the same topic, in one of three ways:
   - the rewrite restates the values its own source question gives;
   - it asks the student to construct the instance ("an array of eight integers that you choose");
   - it becomes a generic task that is still answerable.

   This enforces the model's own prompt rule: never imply that data, requirements or evidence were supplied when they were not. Where the source question itself carries the data (a schedule, a query, a table), the rewrite may still say "the given schedule".
2. **Level alignment** (6 rows):
   - Four Remember rewrites asked for interpretation ("identify the meaning of…"); they are now pure retrieval.
   - Two Create rewrites asked for a tangent-plane equation, which is an Apply task; they are now construction tasks.
3. **Duplicates** (2 rows): two pairs of near-identical sources shared one rewrite. Each now has its own rewrite.
4. **Analyze opening verb** (113 rows): "Determine why/how/which/whether…" is now "Analyze / Examine / Investigate …". "Determine" is an Apply verb in the prompt's verb bank, and the prompt asks for a verb that names the operation. The rest of the wording is unchanged.
5. **Pivot pairs** (`pair_type: "pivot"`):
   - **Why:** v3.1 has only 7 Remember-level source questions, so the model almost never sees a Remember question as input.
   - **How:** each source question and its five rewrites are six questions on one topic with known levels, so a rewrite can serve as the source for another member. Pivots bring every source level to at least 300 train rows (held-out sets: about 38), and every one of the 30 level-to-level pairs has at least 60 train rows.
   - **A pivot is used only if its target:**
     - is not contested, not a deliberate false claim and not an authored scenario;
     - adds no number absent from its source;
     - does not refer to supplied or stated material that its source doesn't mention;
     - has bge-m3 similarity to its source of at least 0.626 (the 10th percentile of direct pairs), so the topic is kept.
   - **Leakage:** pivots stay inside their group's split, so no text crosses splits.

Every revised row keeps its v3.1 text in `v3_1_target_rewrite`, with the reason in `v3_2_revision`.

## Evaluation

- **Pivot pairs reported separately:** `metrics.json → bloom.by_pair_type` reports direct and pivot pairs on their own.
- **Classifier limits:** the Bloom classifier used in evaluation agrees with the gold rewrites only 44% of the time on Remember (it calls most of them Understand) and 60% on Analyze, measured over all 2,920 v3.1 gold rewrites. These are limits of the classifier, so report its accuracy on gold rewrites next to the model scores.

## Limitations

All corrections are model-assisted (Claude) and reviewed row by row, but not by a subject-matter expert. A blinded expert check of a sample (e.g. 100 revised rows and 100 pivot rows) is recommended before claiming human-verified quality.
