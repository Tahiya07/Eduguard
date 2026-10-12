# EduGuard Final 1.5B Multitask Generator Experiment

This directory contains the final training/evaluation pipeline for the shared Qwen2.5-1.5B-Instruct generator.

Architecture:
- Qwen2.5-0.5B-Instruct — Bloom classification only; trained separately.
- Qwen2.5-1.5B-Instruct — one shared LoRA generator for Bloom transformation, SQuAD 1.1 QA, and arXiv paper/abstract summarization.

The 0.5B classification dataset is not part of the 1.5B generator corpus.

## Final dataset contract

Generator inputs:
- data/rewrite dataset/bloomshift_actionverb_v02/ (v0.2.0 action-verb candidate: 691 train / 138 validation / 138 test; held-out annotation worksheet synchronized)
- rajpurkar/squad
- ccdv/arxiv-summarization

Generated corpus: data/multitask_bloom_rewrite_arxiv/ (kept separate from the previous BillSum corpus).

Training sampling: 40% Bloom / 30% QA / 30% summarization. The raw validation split is retained as `validation_full.jsonl`; the active checkpoint-selection split is deterministically row-balanced across Bloom rewrite, QA, and summarization. The test split remains untouched and task-skewed to preserve the generated held-out test data.

SFT context limit: 8192 tokens. SQuAD oversized contexts are windowed while preserving the answer span. arXiv examples whose complete paper+abstract SFT sequence exceeds the fixed budget are excluded deterministically rather than silently truncating the paper. The arXiv dataset's official train/validation/test assignments are preserved.

## Pending BloomShift v0.3 expansion — not yet a training input

The new candidate source pool is stored at `data/rewrite dataset/bloomshift_expansion_v0.3/` as three CSV parts (700 + 700 + 667 = 2,067 source questions) plus a manifest and protocol. These are source questions only; no new rewrites have been generated or human-validated. The count reflects exact normalized-text comparison against the current v0.2 action-verb train/validation/test files only, not every historical inventory. All sources remain pending review and near-duplicate checking.

To regenerate a pool locally, provide the complete set of known source inventories; do not omit historical/cleaned datasets:

    python experiments/bloomshift_expansion/prepare_expansion_source_pool.py --input "data/figshare_bloom_v1.csv" --existing "data/rewrite dataset/bloomshift_actionverb_v02/train.json" --existing "data/rewrite dataset/bloomshift_actionverb_v02/validation.json" --existing "data/rewrite dataset/bloomshift_actionverb_v02/test.json" --output-dir "data/rewrite dataset/bloomshift_expansion_v0.3"

The generator-queue script only accepts source rows marked `approved` and source-target pairs explicitly marked feasible by a reviewer. It creates prompts, not rewrites. The v0.2 dataset remains the active training input until v0.3 has been reconciled against all inventories, reviewed for provenance and near duplicates, generated, human-checked, split by source group, structurally audited, and added to a new manifest. Do not train on the pending source pool.

## Safe workflow

1. Generate the final corpus.
2. Create the balanced validation split; this backs up the original validation file once and never overwrites that backup.
3. Run strict dataset QC with max length 8192.
4. Run sanity_check_multitask_training.py for Bloom, QA, and summarization; use --backward on a suitable GPU machine.
5. Run the resource gate.
6. Train using configs/qwen15b_multitask_taskweighted.json.

Example commands:

    python experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_final.py --output-dir data/multitask_bloom_rewrite_arxiv --model-id Qwen/Qwen2.5-1.5B-Instruct --max-seq-length 8192
    python experiments/multitask_bloom_rewrite/scripts/balance_multitask_validation.py --data-dir data/multitask_bloom_rewrite_arxiv --seed 20261011
    python experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_arxiv --model-id Qwen/Qwen2.5-1.5B-Instruct --max-length 8192
    python experiments/multitask_bloom_rewrite/scripts/sanity_check_multitask_training.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_taskweighted.json --dataset-dir data/multitask_bloom_rewrite_arxiv
    python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_taskweighted.json

Training uses learning rate 1e-4, LoRA r=32/alpha=64/dropout=0.1, and up to 12 epochs with early stopping; it selects the best checkpoint using task-weighted loss on balanced validation rows. Output is written to a new adapter directory so the previous checkpoint is preserved. Balancing is by example count, not by token count, so it reduces row-count dominance but does not mathematically guarantee equal task-loss weighting. The manifest records the seed, original/active counts, and SHA-256 hashes. The test split must not be used for hyperparameter tuning or checkpoint selection; report metrics separately by task.

Historical v2/v3 comparison scripts and old result directories are retained only as provenance; they are not inputs to the final generator training run.
