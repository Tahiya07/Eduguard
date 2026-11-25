# EduGuard Bloom rewrite dataset, v3.1 (train / validation / test)

This folder holds the repaired EduGuard Bloom transformation dataset as Bloom-rewrite rows: one row per source question and target Bloom level. It was exported by `final_corrected_dataset/repaired_v3_20261125/scripts/export_to_data_dir.py` from that folder's `training_ready/` export.

| Split | Rows | Source questions | Rows with an answer source (`usable`) |
|---|---|---|---|
| train | 2,345 | 469 | 1,350 |
| validation | 295 | 59 | 180 |
| test | 290 | 58 | 175 |

- **Fields:** `source_question`, `source_bloom_level`, `target_bloom_level` and `target_rewrite` (the gold rewrite), plus `group_id`, `eligibility`, `content_provenance`, `deliberately_false_claim`, `pass3_rewritten` and `gold_label_contested`.
- **Splits:** pinned to `final_corrected_dataset/training_ready/tests/reference_split_ids.json`. No source question or group appears in more than one split. File hashes are in `dataset_manifest.json`.
- **Deliberately false claims:** these Evaluate rows state a wrong claim for the learner to judge. Never treat that claim as a correct answer.
- **Contested labels:** the 10 rows from examples 213 and 570 have `gold_label_contested: true`. Their source level (Analyze vs Apply) awaits expert review. Exclude them from scoring, or from training if you want to be strict.
- **Challenge set:** the held-out challenge set is in `final_corrected_dataset/repaired_v3_20261125/challenge_set/`, built from the test split. It is for evaluation only, so it is not copied here.
- **Review status:** corrections are model-assisted (Claude) and not human-verified; checks are automated. This is a finalized candidate, not an accepted release. See `final_corrected_dataset/repaired_v3_20261125/reports/QUALITY_VALIDATION_REPORT.md`.
- **Format:** this folder is not in the multitask layout (`bloom_rewrite/`, `qa/`, `summarization/` with `sft_text`) that `experiments/multitask_bloom_rewrite/scripts/dataset_io.py` validates. Prompt formatting has to be applied by the training pipeline.
