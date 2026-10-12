# Expanded BloomShift split v2

The canonical `train.json`, `validation.json`, and `test.json` files now contain the existing 888 records plus the 136 candidates consolidated from batches 0.3.1–0.3.5.

- Total records: **1024**
- Train: **768 (75%)**
- Validation: **153 (14.94%)**
- Test: **103 (10.06%)**
- Source groups crossing splits: **0**
- New candidates awaiting human validation: **136**

The assignment is group-disjoint and approximately 75/15/10. Whole source groups were kept intact; therefore row counts cannot always match exact percentages.

**Research caution:** all 136 new candidates are still marked `pending_human_validation` and `training_eligible=false`. Their presence in these split files does not mean they are validated. Confirm that the training/evaluation pipeline excludes these records until reviewed. Since the original records were reassigned across splits, this is a new evaluation split and previous metrics are not directly comparable.

Original split snapshots are saved as `baseline_train_v1.json`, `baseline_validation_v1.json`, and `baseline_test_v1.json`.
