# Corrected Figshare Bloom Target Rewrite Dataset

This directory is the final source-anchored Bloom target-level rewrite dataset for the EduGuard Qwen2.5-1.5B multitask experiment.

## Source of truth

The Bloom source questions come directly from:

- `data/figshare_bloom_v1_train.csv`
- `data/figshare_bloom_v1_val.csv`
- `data/figshare_bloom_v1_test.csv`

The original Figshare labels are mapped to the canonical six levels already used by EduGuard.

## What is generated

Each accepted Figshare source question receives six target records:

`Remember, Understand, Apply, Analyze, Evaluate, Create`.

For the source's original level, the question is retained as the identity target. For the five cross-level targets, the generator uses a deterministic source-anchored template that changes the student's cognitive operation instead of only replacing a verb.

Every Bloom record contains `prompt_text` and `sft_text` in the ChatML format expected by `experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py`.

## Quality controls

The builder rejects incomplete placeholder questions, unextractable source topics, malformed source forms, and any source for which all six target rows cannot be constructed. It also checks:

- six targets per accepted source;
- source-level train/validation/test disjointness;
- duplicate example IDs;
- source-topic token preservation;
- target-level cue presence;
- absence of training-time source Bloom labels in the generator prompt.

The exact rejection counts and split statistics are stored in `stats.json` and `dataset_manifest.json`.

## Multitask corpus

`data/multitask_corrected/` combines the corrected Bloom rewrite data with the locked QA and summarization rows used by the earlier multitask experiment. The Bloom portion is selected at source-group level so all six target levels remain together.

The training mix targets approximately 40% Bloom rewrite, 30% QA, and 30% summarization. Validation and test retain their complete task-specific evaluation rows.

## Training

Use:

`data/multitask_corrected/train.jsonl`

and:

`data/multitask_corrected/validation.jsonl`

with the existing assistant-only-loss multitask trainer. The held-out test split is:

`data/multitask_corrected/test.jsonl`.
