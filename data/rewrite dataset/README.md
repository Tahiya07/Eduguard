# EduGuard Figshare Target-Bloom Rewrite Dataset — Final Corrected

Main source: `data/figshare_bloom_v1.csv`.

Each usable Figshare source question contributes six target-level records: Remember, Understand, Apply, Analyze, Evaluate, and Create. Same-level records retain the source question; cross-level records use controlled source-anchored cognitive transformations.

## Dataset

- 893 usable source questions
- 5,358 total rewrite records
- Train: 3,768
- Validation: 756
- Test: 834
- Six target levels are balanced within every split.
- No source group is shared across splits.
- Four incomplete/placeholder FigShare rows are excluded and listed in `rejected_sources.json`.

The JSONL records contain `prompt_text` and `sft_text` for the existing assistant-only-loss SFT pipeline.

## GitHub shard layout

The JSONL files are stored as ordered shards:

```
train/train_0001.jsonl ... train/train_0034.jsonl
validation/validation_0001.jsonl ... validation/validation_0004.jsonl
test/test_0001.jsonl ... test/test_0004.jsonl
```

The shards are contiguous pieces of the canonical files and must be concatenated in lexical order.

After cloning, run:

```bash
python "data/rewrite dataset/join_dataset.py"
```

or in PowerShell:

```powershell
& ".\data\rewrite dataset\JOIN_DATASETS.ps1"
```

This creates `train.jsonl`, `validation.jsonl`, and `test.jsonl` in this folder.

See `dataset_manifest.json` for exact counts and SHA-256 hashes.
