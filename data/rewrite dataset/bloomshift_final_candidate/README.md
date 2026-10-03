# EduGuard BloomShift — Final Candidate

Recommended candidate for supervisor validation before the next multitask retraining run.

Train: 660 records
Validation: 114 records
Test: 114 records
Total: 888 records

Each split is target-balanced. Same-level identity copies are excluded. Source groups remain disjoint across train, validation, and test.

The dataset was rebuilt from the original EduGuard source questions. Previous generated target rewrites were not reused. Incomplete/self-referential source questions and source/task families that could not support a reasonable content-preserving transformation were excluded.

This is a silver candidate, not a human-certified gold dataset. Use supervisor_annotation.csv to review every validation/test example before freezing the dataset.
