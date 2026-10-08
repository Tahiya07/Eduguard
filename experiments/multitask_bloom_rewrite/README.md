# EduGuard Final 1.5B Multitask Generator Experiment

This directory contains the final training/evaluation pipeline for the shared Qwen2.5-1.5B-Instruct generator.

Architecture:
- Qwen2.5-0.5B-Instruct — Bloom classification only; trained separately.
- Qwen2.5-1.5B-Instruct — one shared LoRA generator for Bloom transformation, SQuAD 1.1 QA, and BillSum summarization.

The 0.5B classification dataset is not part of the 1.5B generator corpus.

## Final dataset contract

Generator inputs:
- data/rewrite dataset/bloomshift_final_candidate/
- rajpurkar/squad
- FiscalNote/billsum

Generated corpus: data/multitask_bloom_rewrite_final/

Training sampling: 40% Bloom / 30% QA / 30% summarization. Validation and test retain task-specific held-out splits.

SFT context limit: 8192 tokens. SQuAD oversized contexts are windowed while preserving the answer span. BillSum examples whose complete article+summary SFT sequence exceeds the fixed budget are excluded deterministically rather than silently truncating the article.

## Safe workflow

1. Generate the final corpus.
2. Run strict dataset QC with max length 8192.
3. Run sanity_check_multitask_training.py for Bloom, QA, and summarization; use --backward on a suitable GPU machine.
4. Run the resource gate.
5. Train using configs/qwen15b_multitask_final.json.

Example commands:

    python experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_final.py --model-id Qwen/Qwen2.5-1.5B-Instruct --max-seq-length 8192
    python experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_final --model-id Qwen/Qwen2.5-1.5B-Instruct --max-length 8192
    python experiments/multitask_bloom_rewrite/scripts/sanity_check_multitask_training.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json --dataset-dir data/multitask_bloom_rewrite_final
    python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json

Training selects the best checkpoint using validation loss only. The test split must not be used for hyperparameter tuning or checkpoint selection.

Historical v2/v3 comparison scripts and old result directories are retained only as provenance; they are not inputs to the final generator training run.
