# EduGuard Figshare Target-Bloom Rewrite Dataset — Final Corrected

Main source: `data/figshare_bloom_v1.csv`.

This dataset is built at the **source-question level**. Each usable Figshare question contributes six records, one for each revised Bloom target level: Remember, Understand, Apply, Analyze, Evaluate, and Create. The original Figshare question and original Bloom label are retained.

Cross-level records use source-anchored cognitive transformations. The construction is deliberately conservative: topic/technical content is retained while the student operation is changed. Same-level records preserve the original question rather than fabricating a needless paraphrase.

The JSONL files include `prompt_text` and `sft_text`, so they are compatible with the existing assistant-only-loss multitask SFT format.

The included `rejected_sources.json` records source questions that were incomplete or contained unresolved placeholders; these are not silently repaired with invented content.
