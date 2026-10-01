# EduGuard Figshare Target-Bloom Rewrite Dataset — v5

Main source: `data/figshare_bloom_v1.csv`.

This dataset is built at the **source-question level**. Each usable Figshare question contributes six records, one for each revised Bloom target level: Remember, Understand, Apply, Analyze, Evaluate, and Create. The original Figshare question and original Bloom label are retained.

Cross-level records use source-anchored cognitive transformations. The transformation changes the student operation while retaining the source topic, technical concepts, named entities, quantities, and stated constraints. Same-level records preserve the original question.

The train, validation, and test splits are source-group disjoint and balanced within each split. The training formatter can construct the SFT prompt from the core dataset fields; redundant prompt/text copies are not required in the canonical compact train files.

Incomplete or placeholder source questions are recorded in `rejected_sources.json` and are not repaired by inventing missing content.
