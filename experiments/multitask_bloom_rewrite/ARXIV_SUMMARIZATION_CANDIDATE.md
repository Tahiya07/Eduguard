# arXiv summarization candidate for EduGuard

This is an **isolated candidate experiment**. It does not replace or overwrite the existing BillSum dataset, model checkpoint, or reported results.

## Why this candidate exists

BillSum contains U.S. legislative documents. The arXiv summarization dataset is more aligned with scientific-document summarization: its `document` configuration contains `id`, `article`, and `abstract` fields and provides official train/validation/test splits.

Source: https://huggingface.co/datasets/ccdv/arxiv-summarization  
Original dataset paper: Cohan et al. (2018), *Discourse-Aware Attention Model for Abstractive Summarization of Long Documents*, NAACL-HLT, https://aclanthology.org/N18-2097/

## Important limitation: source-paper rights

The dataset card does **not** establish a single blanket license covering every underlying paper. ArXiv papers can have different licenses. The preparation manifest records this as `REVIEW_REQUIRED`; do not claim the entire corpus is permissively licensed, redistribute its source text, or publish the dataset as a redistributable artifact until rights are reviewed. The candidate is not automatically cleared for publication just because it is available on Hugging Face.

## Preparation

From the repository root in PowerShell, with the project virtual environment activated:

```powershell
python experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_final.py --summary-source arxiv --output-dir data/multitask_bloom_rewrite_arxiv_candidate --evaluation-per-task 114
```

The source is loaded in streaming mode to avoid materializing the full multi-gigabyte corpus in memory. The script deterministically selects 495 training examples and 114 examples for each of validation and test. It keeps the complete article and abstract only if the complete SFT sequence fits within the configured 8192-token budget; it does not truncate an article while retaining an abstract written for the full paper. It records selected document/abstract/SFT-token length summaries, skipped records, source IDs, and split metadata in `dataset_manifest.json`.

The target 495 summary-training examples preserve the configured 40% Bloom / 30% QA / 30% summarization training mix. The held-out test has 114 examples per task; examples are sampled without replacement from the respective source test partitions, and all frozen Bloom test examples are retained.

## Validate before training

```powershell
python experiments/multitask_bloom_rewrite/scripts/validate_multitask_dataset.py --data-dir data/multitask_bloom_rewrite_arxiv_candidate --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_arxiv_candidate.json
```

Review the generated `qc_report.json` and `dataset_manifest.json`. Do not proceed if QC fails, source split IDs overlap, any task count differs from the manifest, or the selected-length audit reveals a source-quality problem. A structural QC pass is not proof of summary factuality or academic usefulness; inspect a sample of article/abstract pairs and evaluate factual consistency separately.

## Candidate training (only after preparation and QC pass)

```powershell
python experiments/multitask_bloom_rewrite/scripts/train_multitask.py --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_arxiv_candidate.json
```

Use the repository's actual trainer entry point if its CLI differs; verify `--help` before starting a long run. This candidate config uses separate output and results directories so it cannot overwrite the BillSum baseline. Do not train from the candidate until the data build and QC are complete.

## Interpretation

This is a scientific-paper summarization benchmark/training source, not a representative sample of all university materials. Results support claims about the tested scientific-paper summarization setting only. Report task-specific metrics and assess factual consistency; do not rely on a single aggregate multitask score.
