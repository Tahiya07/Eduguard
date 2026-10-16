# A/B/C/D Experiment Matrix — Task-Separated Data

## Purpose

Compare the same Qwen2.5-1.5B-Instruct model under controlled conditions while keeping BloomShift, SQuAD, and arXiv stored in separate directories. The loader verifies the manifest and combines selected records in memory only; no merged train/validation/test JSONL is persisted.

## Frozen data contract

Data root: `data/multitask_bloom_rewrite_separate/`

- `bloom_rewrite/{train,validation,test}.jsonl`
- `qa/{train,validation,test}.jsonl`
- `summarization/{train,validation,test}.jsonl`
- `dataset_manifest.json`

The baseline preparation uses seed 42 and a requested training sample mix of 40% BloomShift / 30% SQuAD / 30% arXiv. The builder retains all 1,075 BloomShift train records and samples sufficient QA/arXiv train examples to make the requested mix. With 1,075 Bloom rows and ratios 0.40/0.30/0.30, the existing round-to-nearest rule selects 806 QA and 806 arXiv examples, for 2,687 total train records (the realized percentages can differ slightly due to integer rounding).

The 212 BloomShift validation and 135 test records stay held out. QA and arXiv retain their own validation/test splits. All generated file hashes and selected source split fingerprints are recorded in the manifest. Do not regenerate the prepared data between C and D; the manifest, per-task file hashes, and exact selected IDs must match.

The owner batch attestation allows all BloomShift rows into the workflow but does not establish independent double annotation, row-level rubric completeness, or inter-rater reliability.

## Conditions

| Condition | Training | Status |
|---|---|---|
| A — Prompt-only baseline | Base Qwen2.5-1.5B-Instruct, no adapter or fine-tuning | Configured |
| B — Bloom-only LoRA | Train only on `bloom_rewrite/train.jsonl`; all 1,075 BloomShift training records | Configured |
| C — Multitask baseline | The frozen task-separated sample: 40/30/30 data proportions, ordinary assistant-token causal-LM loss | Configured |
| D — Task-balanced loss | Same data files and same sample IDs as C; response-normalized task losses weighted 50/25/25 (Bloom/QA/arXiv) | Requires a separate custom loss implementation; not enabled by current trainer |

D must not be launched with the current trainer just by adding a `loss_weights` config key; the current Trainer still uses its ordinary unweighted loss. D requires an implemented and unit-tested task-aware loss that computes mean assistant-token loss per example, averages examples within each task, and then combines task losses with the requested weights. It must handle batches without all three task types and log effective per-task losses.

## Shared evaluation protocol

- Use `evaluate_rewrite.py --split validation` for choosing between A/B/C/D.
- Evaluate all BloomShift validation examples; deterministically sample at most 300 QA and 300 arXiv validation examples. Preserve the selected IDs in the results.
- Use the same auxiliary Bloom classifier in every condition. Classifier agreement is a proxy, not human ground truth.
- Use the same prompt, maximum new-token limit, greedy decoding, and generation config.
- Bloom primary analysis should include format validity, transformation/cognitive-operation heuristics, overlap gates, trivial verb changes, per-target-level results, and human adjudication when available. QA guardrails are Exact Match/token F1; arXiv guardrails are ROUGE-1/2/L.
- Once the condition has been selected on validation, run final evaluation on the locked test split. Do not tune on test outcomes.

## Commands

Prepare and validate once:

```powershell
python experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_final.py --model-id Qwen/Qwen2.5-1.5B-Instruct --max-seq-length 8192 --mix-bloom 0.40 --mix-qa 0.30 --mix-sum 0.30
python experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_separate --model-id Qwen/Qwen2.5-1.5B-Instruct --max-length 8192
```

A baseline:

```powershell
python experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py --config experiments/multitask_bloom_rewrite/configs/experiment_A_base_prompt_only.json --condition base --split validation --evaluation-per-task 300
```

B training:

```powershell
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/experiment_B_bloom_only.json
```

C training:

```powershell
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json
```

Before D, do not change the generated task files. Use a new output directory for every condition, and retain the training selection manifest and evaluation subset manifests.
