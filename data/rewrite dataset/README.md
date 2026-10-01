# EduGuard Figshare Target-Bloom Rewrite Dataset — v3-final-corrected

Main source: `data/figshare_bloom_v1.csv`.

The dataset is built at the source-question level. Each of the 893 usable source questions contributes six records: one for each Revised Bloom target level: Remember, Understand, Apply, Analyze, Evaluate, and Create. Same-level records preserve the original source question.

Cross-level records use source-anchored cognitive transformations. The transformation changes the student operation while retaining the source topic, technical concepts, named entities, quantities, and stated constraints. Transformation wording is selected from the source task family rather than using the legacy generic wrappers.

The splits are source-group disjoint. Train contains 3,768 records (628 per target level), validation contains 756 records (126 per target level), and test contains 834 records (139 per target level).

The canonical files are the compact `.json` and `.jsonl` datasets. Each JSONL record is identical to its corresponding JSON array record.
