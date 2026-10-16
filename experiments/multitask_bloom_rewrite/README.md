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
data/multitask_bloom_rewrite_separate/
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

The data preparation step retains all 1,075 BloomShift training rows and can deterministically cap QA and arXiv training rows independently. For the 40/30/30 baseline, use 806 QA and 806 arXiv training rows. The validation/test caps are independent of training caps. Those three per-task training files are frozen inputs for both experiments C and D. Do not regenerate them between C and D: their manifest hashes and selected IDs must match.

Validation and test files stay in their respective task directories. The builder deterministically caps QA and summarization validation to the BloomShift validation count (212 each), and their test splits to the BloomShift test count (135 each), using fixed task-specific seed offsets. The builder records pre-cap availability and selected counts in the manifest. Keep every test file locked until strategy selection is complete.

SQuAD uses its official validation split as the task test split, with an isolated validation subset held out from its official train split by context group. SQuAD oversized examples use answer-preserving context windows. The arXiv dataset's official split boundaries are retained; complete prompt-plus-target examples exceeding the configured 8192-token budget are excluded and counted in the manifest rather than silently truncating an article while retaining its abstract.

## Recommended workflow

Run all commands from the repository root.

1. Audit BloomShift source inclusion and provenance:

```powershell
python experiments/multitask_bloom_rewrite/scripts/audit_bloomshift_dataset.py
```

2. Prepare the separate task directories. Do not pass a combined corpus path:

```powershell
python experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_final.py --model-id Qwen/Qwen2.5-1.5B-Instruct --max-seq-length 8192 --mix-bloom 0.40 --mix-qa 0.30 --mix-sum 0.30
```

3. Run strict integrity, split isolation, token length, and assistant-only mask QC:

```powershell
python experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_separate --model-id Qwen/Qwen2.5-1.5B-Instruct --max-length 8192
python experiments/multitask_bloom_rewrite/scripts/sanity_check_multitask_training.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json --dataset-dir data/multitask_bloom_rewrite_separate
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

The task-balanced D loss implementation remains a separate method change after A/B/C results are available. D must use the exact same prepared per-task train files and validation row IDs as C, and vary only the response-normalized task-loss weights. Do not label an unweighted run as D.

## Reproducibility requirements

- Preserve `dataset_manifest.json`, generated file hashes, and `qc_report.json` with every experiment.
- Preserve `training_selection_manifest.json` and evaluation subset ID manifests.
- Use the same data manifest and generated task-file fingerprints for C and D.
- Use validation for selection; evaluate the chosen candidate on the untouched test split only once after model selection.
- Report automated cognitive-operation and lexical-overlap checks as heuristics. Do not present them as semantic-equivalence or pedagogical-validity ground truth.
- Do not describe BloomShift's owner batch attestation as independent gold review or inter-rater reliability.
