# EduGuard summarization v2: audit and cleaning report

Date: 2026-11-26. Input: the owner-edited `data/summarization/{train,validation,test}.jsonl` (commit 66886d1, "sum data"), which is not modified. Output: this folder. Tool: `experiments/multitask_bloom_rewrite/scripts/clean_summarization_dataset.py`. The per-row reasons are in `cleaning_log.json`, and the counts and hashes are in `dataset_manifest.json`.

## Result

| Split | Planned | Clean v2 | Shortfall |
|---|---:|---:|---:|
| Train | 1,172 | 429 | 743 |
| Validation | 146 | 30 | 116 |
| Test | 147 | 31 | 116 |
| **Total** | **1,465** | **490** | **975** |

By length (source words), train has 240 short, 169 medium and 20 long rows. Validation has 29 short and 1 medium; test has 29 short and 2 medium. **No held-out row is longer than about 1,050 words.** All long and very long held-out documents in v1 were arXiv rows, which have been removed.

## What the input contained (1,627 rows)

| Problem | Rows dropped | Why it matters |
|---|---:|---|
| Rendered SFT copies (`task`/`sft_text`/`article` fields) of rows already present | 376 | These are exact duplicates. 24 of them sat in validation or test while the same ID was in train, which is direct train/test leakage. |
| `UAP-ACA-TRAIN-*` (`curriculum_aligned_original`) | 586 | Only 98 distinct sources, each repeated about 6 times. The "summary" is the first half of the source copied word for word (summary/source ratio > 0.40), so it teaches copying, not summarizing. |
| New rows padded with template text | 164 | Sentences repeat across many rows, e.g. "In the context of X, this point should be interpreted alongside the specific concepts…" after every sentence, and "Record the limits of each source." repeated 10+ times. Many summaries are generic filler sentences. Median padding share is 77%. |
| Summary is a verbatim lead of the source (`lecture_slide_text_style`) | 8 | The summary is just the source's opening text. |
| New row near-duplicating a validation or test row (similarity ≥ 0.50) | 3 | This would leak a held-out topic into train. |

Other fixes:
- 2 blank lines are skipped.
- 50 IDs written without zero padding (e.g. `UAP-SUM-687`) are normalized to `UAP-SUM-0687`.
- Two IDs, 0487 and 0488, were each used for two different rows. The padded rows were dropped, and the clean ones keep the IDs.
- Department abbreviations `Civil` and `ME` are expanded.
- Added rows with no metadata are labelled `source_type=original_academic_example` and `review_status=owner_added_unverified`.

Kept from the new additions: 25 rows (UAP-SUM-0445–0468, 0487–0492, minus the near-duplicates). The other 444 + 21 rows are the v1 non-arXiv records, unchanged.

## Checks on v2

- **Integrity:** unique IDs; 0 exact duplicate sources or summaries; maximum summary/source ratio 0.381.
- **No cross-split leakage:** maximum cross-split source similarity is 0.486 (train/validation), 0.476 (train/test) and 0.427 (validation/test), all below the 0.50 threshold.
- **Training-ready:** all sequences fit within 8,192 tokens (longest 2,989), and `validate_multitask_dataset.py` and `validate_training_configs.py` pass with no errors.

## Known limitations

- **Held-out sets are small:** 30 and 31 rows, so summarization metrics have wide confidence intervals. They also contain no long documents.
- **Provenance:** all rows are model-generated or owner-added and unverified. Factual faithfulness is not checked automatically.
- **Extractive summaries:** the 21 `lecture_slide_text_style` summaries consist of source sentences (extractive), unlike the abstractive style of the rest.
