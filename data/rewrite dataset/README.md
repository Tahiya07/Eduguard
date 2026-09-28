# EduGuard Figshare Target-Bloom Rewrite Dataset — Final Corrected

Main source: `data/figshare_bloom_v1.csv`.

This dataset is built at the source-question level. Each usable Figshare question contributes six records, one for each revised Bloom target level: Remember, Understand, Apply, Analyze, Evaluate, and Create. The original Figshare question and original Bloom label are retained.

Cross-level records use source-anchored cognitive transformations. Same-level records preserve the original source question.

The JSONL files include `prompt_text` and `sft_text`, compatible with the existing assistant-only-loss multitask SFT format.

The canonical dataset package used to generate these files is the corrected package supplied in this conversation. The GitHub connector used here can write UTF-8 text but cannot stream the multi-megabyte JSONL artifacts from the uploaded ZIP into GitHub; therefore this folder contains the reproducibility metadata, while the canonical JSONL files remain in the supplied package.

See `dataset_manifest.json` for exact counts and SHA-256 hashes.
