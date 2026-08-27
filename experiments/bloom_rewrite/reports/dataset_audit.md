# Dataset audit (experiment copy)

Full public-dataset table: [`docs/bloom_rewrite_dataset_audit.md`](../../../docs/bloom_rewrite_dataset_audit.md).

## Local Figshare

`data/figshare_bloom_v1.csv` is a **Bloom classification** corpus (2330 rows; columns `question`, `original_label`, `bloom_level`). It does **not** contain `original_question → target_level → target_rewrite` pairs and must not be described as rewrite supervision.

Understand is the majority class (915/2330). 120 questions are shorter than 6 words; 68 look like incomplete templates. Exact string duplicates are 0; 5 normalized duplicates exist. The published classifier splits have small normalized leakage and were not reused.

## Public rewrite-pair search

No accepted public dataset provides genuine source→target Bloom **question transformation** pairs. CogBench variants were inspected and **rejected** (classification, passage-to-question generation, math explanation alignment, or unrelated cognitive-dynamics questionnaires). EduQuest, BloomXplain, and BloomVQA fail the same test.

## Strategy

**B.** Figshare questions are the source pool. Target rewrites are produced by `bloom_target_policy.py` (deterministic task templates). Qwen2.5-0.5B and Qwen2.5-1.5B were not used as teachers.

Built corpus: `bloom_rewrite_synth_v1`, hash `681d88530dfc953eb4d230936fade3b43e63b1d4e9369e1dbed4176d856d9e67`, 10002 usable examples (train 6975 / val 1491 / test 1536). All 30 non-self cells are present. Understand→* cells are larger than Create→* cells because the source pool is unbalanced.
