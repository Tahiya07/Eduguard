# EduGuard Figshare Target-Bloom Rewrite Dataset — v3-final

Main source: `data/figshare_bloom_v1.csv`.

The dataset is built at the **source-question level**. Each of the 893 usable source questions contributes six records: one for each revised Bloom target level (Remember, Understand, Apply, Analyze, Evaluate, Create). The original source question and source Bloom label are retained.

Non-identity records use source-anchored cognitive transformations. The rewrite changes the cognitive operation while preserving the source topic, technical concepts, named entities, quantities, and stated constraints. Missing information is not invented.

The splits are source-group disjoint. Train contains 3,768 records (628 per target level), validation contains 756 records (126 per target level), and test contains 834 records (139 per target level).

The canonical files are the compact `.json` and `.jsonl` datasets. Redundant prompt/text/message copies are intentionally omitted; the training formatter can construct the SFT prompt from the core fields.

