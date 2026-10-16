# EduGuard Qwen2.5-1.5B Multitask Generator Experiment

This directory contains the training/evaluation pipeline for the shared Qwen2.5-1.5B-Instruct generator.

Architecture:
- Qwen2.5-0.5B-Instruct — Bloom classification only; trained separately.
- Qwen2.5-1.5B-Instruct — one shared LoRA generator for Bloom transformation, SQuAD 1.1 question answering, and arXiv summarization.

The 0.5B classification dataset is not part of the 1.5B generator corpus.

## BloomShift inclusion contract

Canonical BloomShift source split sizes are 1,075 train + 212 validation + 135 test = 1,422 records. All 1,422 rows must be retained across the modeling workflow. All 1,075 train rows update model weights; the 212 validation rows remain held out for checkpoint selection and diagnostics; and the 135 test rows remain held out for final evaluation. Never move validation/test examples into train to increase training volume.

The batch attestation file `data/rewrite dataset/bloomshift_final_candidate/batch_review_attestation.json` documents the dataset owner's report that all rows were reviewed and the instruction to include every row. The builder does not filter on legacy row-level `review_status` or `training_eligible` flags. It preserves the prior values in the derived corpus for provenance and assigns `review_status=owner_batch_attested` and `training_eligible=true` to included examples.

The repository's reviewer files are byte-identical, and complete independent item-level rubric ratings are not available for the held-out split. Therefore, the batch attestation must not be reported as independent dual annotation, inter-rater reliability, or proof of gold-label status. Independent evaluation evidence remains a separate publication-readiness requirement.

## Multitask corpus contract

Generator sources:
- `data/rewrite dataset/bloomshift_final_candidate/`
- `rajpurkar/squad`
- `ccdv/arxiv-summarization`

Generated corpus: `data/multitask_bloom_rewrite_final/`

The current baseline sampling mix is 40% Bloom / 30% QA / 30% summarization, with the Bloom training split acting as the anchor. SFT context limit: 8192 tokens. SQuAD contexts are windowed while preserving the answer span. arXiv examples whose complete article+abstract SFT sequence exceeds the fixed token budget are excluded and counted in the manifest; this length policy applies to the auxiliary summarization source, not the canonical BloomShift records.

## Safe workflow

1. Run the BloomShift inclusion/provenance audit.
2. Generate the full multitask corpus with all BloomShift split rows retained.
3. Run strict corpus QC with max length 8192.
4. Run `sanity_check_multitask_training.py` for Bloom, QA, and summarization; use `--backward` on a suitable GPU machine.
5. Run the resource gate.
6. Train from a clean output directory using the intended configuration.

Example commands:

```powershell
python experiments/multitask_bloom_rewrite/scripts/audit_bloomshift_dataset.py
python experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_final.py --model-id Qwen/Qwen2.5-1.5B-Instruct --max-seq-length 8192 --mix-bloom 0.40 --mix-qa 0.30 --mix-sum 0.30
python experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_final --model-id Qwen/Qwen2.5-1.5B-Instruct --max-length 8192
python experiments/multitask_bloom_rewrite/scripts/sanity_check_multitask_training.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json --dataset-dir data/multitask_bloom_rewrite_final
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json
```

The existing config uses validation loss for checkpoint selection; a follow-up training-method change should add task-aware evaluation and loss balancing as a separate experiment. Never use the test split for hyperparameter tuning or checkpoint selection. Use a new output directory for each experimental run so previous checkpoints cannot silently contaminate a fresh run.


## Sequential model comparison (A → B → C → D)

Use `EXPERIMENT_MATRIX_A_TO_D.md` as the experiment contract. Start with the untrained base Qwen2.5-1.5B-Instruct prompt-only baseline (A), then evaluate Bloom-only LoRA (B), the existing 40/30/30 multitask LoRA baseline (C), and only then implement the task-balanced 50/25/25 loss experiment (D).

The evaluation script now supports `--split validation|test`, verifies the selected split against the manifest counts and SHA-256, and defaults to validation so A/B/C/D comparisons do not use the test split for model selection. It selects the same Bloom/QA/summarization validation IDs across conditions using stable ID ordering and task-specific seeds. The evaluator saves each run's selected IDs, predictions, metrics, environment metadata, and human-rating export in that run's output directory.

Experiment A configuration: `configs/experiment_A_base_prompt_only.json`.

Run A after the full BloomShift corpus has been rebuilt and passed QC:

```powershell
python experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py --config experiments/multitask_bloom_rewrite/configs/experiment_A_base_prompt_only.json --condition base --split validation --evaluation-per-task 300
```

Do not implement D's weighted loss until A/B/C provide baselines. D must keep the same 40/30/30 example sample mixture as C and change only the response-normalized task loss weights to 50/25/25.
