# Held-out Bloom challenge set (evaluation only)

Built by `scripts/build_challenge_set.py` from the v3 **test** split only: 58 examples, 348 items (58 sources + 290 transformations) and 406 same-content adjacent-level pairs, across 25 disciplines. No example, group or question text appears in train or validation; a unit test enforces this. Do not use it for training, tuning, early stopping or checkpoint selection.

It answers one question: does a model recognise the cognitive operation from the whole question, or only from its leading verb? Report accuracy on each slice next to overall accuracy (tags are in `diagnostic_tags`; counts are in `challenge_manifest.json`):

| Slice | Meaning |
|---|---|
| `verb_shortcut_misleads` | the verb's most common level in train is not the gold level |
| `verb_multi_level_in_train`, `shared_bank_verb` | the verb occurs at more than one level |
| `rare_or_unseen_verb` | the verb occurs fewer than 5 times at the gold level in train |
| `scenario_first` | a context sentence comes before the instruction |
| `masked_verb_nb_wrong` | the question is hard to label once the verb is removed |
| `challenge_pairs.jsonl` | two questions on the same content at adjacent levels |

For rewriting evaluation, `content_anchor_tokens` lists the source content words each item keeps. Exclude `label_and_generation_only` items from answer scoring. Never score against a `false_claim_object` claim: its stated claim is deliberately wrong.

Tags are computed automatically and are heuristic; they are not human-verified.

Review (pass 3.1): every item was read; verdicts are in `review_verdict`. Four items (213 and 570, source and Apply) have contested gold labels and carry `score_this_item = false`. Pairs involving them carry `score_this_pair = false`. Skip both when scoring. Baseline results per slice are in `../audits/challenge_baselines.json`.
