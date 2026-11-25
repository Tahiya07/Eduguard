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

## Summarization quality filter (2026-11-25)

The arXiv rows come from `ccdv/arxiv-summarization`, whose conversion replaced every LaTeX math span with an `@xmathN` token and every citation with `@xcite`. The expressions were never kept. Abstract placeholders are numbered independently of the article, and rows carry no arXiv ID or title, so **no math is recoverable** from the available data, and none is invented.

`scripts/build_clean_summarization_dataset.py` reads `data/multitask_bloom_rewrite_separate/` without modifying it and writes `data/multitask_bloom_rewrite_separate_sumclean_v1/`:

- `bloom_rewrite/` and `qa/` are byte-identical copies, so their eligibility is unchanged.
- `summarization/{split}.jsonl` keeps only eligible rows, with their original content, order and split.
- `summarization/summarization_eligibility.json` records each row's status, rule reasons, placeholder counts, recovery status, row SHA-256 and provenance.
- `summarization/summarization_exclusions.csv` is a human-readable exclusion list.

The rules are defined in `summarization_quality.py`. The target abstract must be free of placeholders, LaTeX residue, appended metadata and bibliography references, and must end with a complete sentence. The input article is excluded only if it is not a paper body or more than 10% of its tokens are placeholders. Thresholds were set from the train split only.

| Split | Original | Eligible | Excluded | Abstracts with `@xmath` (occurrences) |
|---|---:|---:|---:|---:|
| train | 1,075 | 270 | 805 | 680 (10,375) |
| validation | 212 | 81 | 131 | 105 (397) |
| test | 135 | 39 | 96 | 78 (262) |

**Loader enforcement** (`scripts/dataset_io.py`):
- In a directory that has an eligibility record, every summarization row must be listed as eligible, with a matching row hash and no `@xmath` in its target, or loading raises `SummarizationEligibilityError`.
- In the legacy directory, summarization **train** rows are refused unless `allow_unaudited_summarization=True` is passed. Validation and test rows still load, so earlier evaluations can be reproduced.
- Bloom rewriting and QA loads are unaffected.

To train with summarization, pass `--dataset-dir data/multitask_bloom_rewrite_separate_sumclean_v1`. The task counts differ from the earlier runs, so results are not directly comparable with runs on the legacy directory.

## Recommended workflow

Run all commands from the repository root.

1. Audit BloomShift source inclusion and provenance:

```powershell
python experiments/multitask_bloom_rewrite/scripts/audit_bloomshift_dataset.py
```

2. Validate the curated task-separated files in place. Do not rebuild them before this follow-up run:

```powershell
python experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_separate --model-id Qwen/Qwen2.5-1.5B-Instruct --max-length 8192
python experiments/multitask_bloom_rewrite/scripts/check_resources.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_bloom_priority_20261123.json
python experiments/multitask_bloom_rewrite/scripts/sanity_check_multitask_training.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_bloom_priority_20261123.json --dataset-dir data/multitask_bloom_rewrite_separate --backward
```

The priority config uses 4-bit NF4 QLoRA, SDPA attention, and non-reentrant gradient
checkpointing to reduce peak VRAM without shortening the source context. Before
training, the runner checks currently free GPU memory and runs a backward pass on
long training examples. It stops before training if the configured context does not
fit with the required safety headroom. The resource estimate alone is not a guarantee.

4. Run the experiments sequentially, using fresh output directories:

```powershell
# A — base prompt-only validation baseline
python experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py --config experiments/multitask_bloom_rewrite/configs/experiment_A_base_prompt_only.json --condition base --split validation --evaluation-per-task 300

# B — BloomShift-only LoRA training
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/experiment_B_bloom_only.json

# C — 40/30/30 multitask LoRA baseline
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_final.json

# Follow-up — 50/25/25 task-sampling experiment after the 2026-11-23 result
python experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_bloom_priority_20261123.json
```

After training finishes, evaluate the new adapter on validation using the same split and generation settings. The evaluator requires a fixed auxiliary Bloom classifier; use `--classifier-dir` to select one explicit checkpoint if auto-detection does not select the same classifier in every condition. Classifier agreement is a diagnostic proxy, not human ground truth for Bloom-level alignment.

The follow-up changes sampling probabilities only; it does not change examples, labels, prompts, optimizer settings, or validation/test files. Compare its validation results with the previous run before using the locked test split once for the selected model.

## Reproducibility requirements

- Preserve `dataset_manifest.json`, generated file hashes, and `qc_report.json` with every experiment.
- Preserve `training_selection_manifest.json` and evaluation subset ID manifests.
- Use the same data manifest and generated task-file fingerprints for C and D.
- Use validation for selection; evaluate the chosen candidate on the untouched test split only once after model selection.
- Report automated cognitive-operation and lexical-overlap checks as heuristics. Do not present them as semantic-equivalence or pedagogical-validity ground truth.
- Do not describe BloomShift's owner batch attestation as independent gold review or inter-rater reliability.
