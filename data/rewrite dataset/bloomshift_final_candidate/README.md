# EduGuard BloomShift — Expanded Candidate v3

The canonical split files contain 1,422 records. Every record is retained in the multitask corpus under the dataset-owner batch attestation in `batch_review_attestation.json`.

| Split | Records | Role in modeling |
|---|---:|---|
| Train | 1,075 | All rows are used for gradient updates |
| Validation | 212 | All rows are held out for checkpoint selection and diagnostics |
| Test | 135 | All rows remain held out for final evaluation |
| **Total** | **1,422** | **No BloomShift row is excluded for missing legacy review metadata** |

The train/validation/test source-group boundaries remain fixed and group-disjoint. Validation and test records must never be moved into the training split simply to increase training volume. The full collection is used across the modeling workflow, while only the 1,075 training-split rows update model weights.

## Inclusion and review provenance

The dataset owner reported that all rewrites had been reviewed and instructed that all records must be used. The batch attestation records that instruction explicitly. The multitask builder no longer filters records based on missing or false legacy `review_status` and `training_eligible` fields. In the derived JSONL corpus, every row receives `review_status=owner_batch_attested` and `training_eligible=true`, while its original metadata is preserved as `source_review_status` and `source_training_eligible`.

This inclusion policy does **not** fabricate row-level rubric ratings. The repository's reviewer files are currently byte-identical and complete independent held-out ratings are not demonstrated. The attestation is a dataset-owner batch statement—not evidence of independent dual annotation, adjudication, or inter-rater reliability.

Consequently, all rows may be used for SFT and provisional validation/test diagnostics, but do not describe the held-out set as an independently annotated gold benchmark or report inter-rater agreement until genuine independent ratings and required rubric fields have been recorded.

## Required commands

Run the audit before preparing the corpus:

```powershell
python experiments/multitask_bloom_rewrite/scripts/audit_bloomshift_dataset.py
python experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_final.py --mix-bloom 0.40 --mix-qa 0.30 --mix-sum 0.30 --max-seq-length 8192
python experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_final --max-length 8192 --model-id Qwen/Qwen2.5-1.5B-Instruct
```

The builder raises an error for missing required fields or any unexpected row-count change rather than silently dropping an example. Auxiliary summarization examples that exceed the declared token budget remain subject to the documented sequence-length policy; the all-row guarantee applies to the canonical BloomShift dataset.
