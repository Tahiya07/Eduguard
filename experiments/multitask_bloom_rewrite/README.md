# EduGuard Qwen2.5-1.5B Multitask Generator

This directory contains the SFT, data preparation, QC, and evaluation tools for the shared Qwen2.5-1.5B-Instruct generator.

## Task architecture

- **Qwen2.5-0.5B-Instruct:** Bloom classification only, trained and evaluated separately.
- **Qwen2.5-1.5B-Instruct:** shared generator for Bloom question transformation, SQuAD question answering, and arXiv summarization.

The 0.5B Bloom-classification dataset is not used as the 1.5B generator's rewrite targets.

## Source datasets and persisted layout

The sources remain independent:

- Bloom transformation: `data/rewrite dataset/bloomshift_final_candidate/`
- QA: `rajpurkar/squad`
- Summarization: `ccdv/arxiv-summarization`

Prepared data is stored in **task-specific directories**, not as one merged JSONL corpus:

```text
data/multitask_bloom_rewrite_capped/
├── dataset_manifest.json
├── bloom_rewrite/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── qa/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
└── summarization/
    ├── train.jsonl
    ├── validation.jsonl
    └── test.jsonl
```

There must be no root-level `train.jsonl`, `validation.jsonl`, or `test.jsonl` in this directory. The builder refuses to reuse an output directory that contains such stale merged files. At training/evaluation time, each task's file is verified against `dataset_manifest.json`; the application may combine the loaded rows **in memory only**, but it never writes a combined dataset file.

## Split and sampling contract

BloomShift uses the canonical split exactly as audited: 1,075 training rows, 212 validation rows, and 135 test rows. The owner batch attestation permits all records to be included despite missing legacy row-level flags. It does **not** establish independent double annotation or inter-rater reliability.

The capped baseline retains all 1,075 BloomShift training rows and uses deterministic, independently sampled subsets of 806 QA and 806 summarization training rows (seed 42). This yields approximately 40% BloomShift, 30% QA, and 30% summarization by example count. The builder fails if either auxiliary task has fewer eligible rows than its requested cap, rather than silently producing a different mixture. This is example-count balancing, not token-count balancing: summarization may contribute more supervised tokens per row. The trainer shuffles the loaded rows together in memory; it does not enforce a fixed task ratio in every batch. Keep the per-task files separate on disk. Freeze the generated files for the experiment and preserve their hashes.

The complete prepared validation and test splits for each task stay in their respective task directory. The evaluation script defaults to validation and deterministically samples up to 300 QA rows and 300 summarization rows while evaluating all BloomShift rows. Keep every test file locked until strategy selection is complete.

SQuAD uses its official validation split as the task test split, with an isolated validation subset held out from its official train split by context group. SQuAD oversized examples use answer-preserving context windows. The arXiv dataset's official split boundaries are retained; complete prompt-plus-target examples exceeding the configured 8192-token budget are excluded and counted in the manifest rather than silently truncating an article while retaining its abstract.

## Recommended workflow

Run all commands from the repository root.

1. Audit BloomShift source inclusion and provenance:

```powershell
python experiments/multitask_bloom_rewrite/scripts/audit_bloomshift_dataset.py
```

2. Prepare the separate task directories. Do not pass a combined corpus path:

```powershell
python experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_final.py --output-dir data/multitask_bloom_rewrite_capped --model-id Qwen/Qwen2.5-1.5B-Instruct --max-seq-length 8192 --seed 42 --max-qa-train 806 --max-sum-train 806
```

3. Run strict integrity, split isolation, token length, and assistant-only mask QC:

```powershell
python experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_capped --model-id Qwen/Qwen2.5-1.5B-Instruct --max-length 8192
python experiments/multitask_bloom_rewrite/scripts/sanity_check_multitask_training.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json --dataset-dir data/multitask_bloom_rewrite_capped
```

4. Run the experiments sequentially, using fresh output directories:

```powershell
# A — base prompt-only validation baseline
python experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py --config experiments/multitask_bloom_rewrite/configs/experiment_A_base_prompt_only.json --condition base --split validation --evaluation-per-task 300

# B — BloomShift-only LoRA training
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/experiment_B_bloom_only.json

# C — 40/30/30 multitask LoRA baseline
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json
```

After B and C finish, evaluate each using the same validation split and generation settings. The evaluator requires a fixed auxiliary Bloom classifier; use `--classifier-dir` to select one explicit checkpoint if auto-detection does not select the same classifier in every condition. Classifier agreement is a diagnostic proxy, not human ground truth for Bloom-level alignment.

A task-balanced-loss experiment may be added as a separate method after the capped example-mixture baseline is established. It must use exactly the same prepared per-task files and validation IDs as C, and change only the loss-weighting method. Do not describe example-count capping as token-balanced training.

## Reproducibility requirements

- Preserve `dataset_manifest.json`, generated file hashes, and `qc_report.json` with every experiment.
- Preserve `training_selection_manifest.json` and evaluation subset ID manifests.
- Use the same data manifest and generated task-file fingerprints for C and D.
- Use validation for selection; evaluate the chosen candidate on the untouched test split only once after model selection.
- Report automated cognitive-operation and lexical-overlap checks as heuristics. Do not present them as semantic-equivalence or pedagogical-validity ground truth.
- Do not describe BloomShift's owner batch attestation as independent gold review or inter-rater reliability.
