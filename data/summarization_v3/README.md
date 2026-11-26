# EduGuard summarization v3

This folder holds the training-ready summarization data at the planned size: **1,172 train / 146 validation / 147 test (1,465 rows)**. It is attached to training as `data/multitask_bloom_rewrite_separate_eduguard_v3_6`, with config `qwen15b_multitask_eduguard_v3_6_20261126.json`. Counts, hashes, the OpenStax repository commits and all checks are recorded in `dataset_manifest.json`.

## Composition

| Source | Train | Validation | Test | Summary author |
|---|---:|---:|---:|---|
| EduGuard v2 (clean, see `data/summarization_v2/dataset_audit_report.md`) | 429 | 30 | 31 | model-generated or owner-added, unverified |
| OpenStax textbook sections (33 books) | 743 | 116 | 116 | textbook authors (section summary) |
| **Total** | **1,172** | **146** | **147** | |

Rows by length (source words):

| Split | Short (≤600) | Medium (601–1,200) | Long (1,201–2,500) | Very long (2,501–4,000) |
|---|---:|---:|---:|---:|
| Train | 240 | 352 | 412 | 168 |
| Validation | 29 | 44 | 51 | 22 |
| Test | 29 | 44 | 52 | 22 |

Train has 2 more long rows and 2 fewer very-long rows than the target, because the eligible very-long pool ran out. The split totals are unaffected.

## OpenStax rows

- **What a row is:** `source_text` is the body of one textbook section, converted from the official CNXML source (`github.com/openstax/osbooks-*`, commits in the manifest). `summary` is that section's author-written summary ("Key Concepts and Summary", "Section Summary" or "Summary").
- **What was removed from the source:** figures, tables, boxed features, inline citations, exercises, review and AP sections, learning objectives, reading lists, URLs and the summary itself.
- **Selection:**
  - Source length 601–4,000 words.
  - Summary at least 40 words and between 2.5% and 40% of the source, and not a copy of its opening.
  - No Markdown, and no text with 4 or more math expressions.
- **Leakage control:** v1/v2 similarity measure; threshold 0.50, with candidates linked at 0.45 for margin.
  - Candidates resembling a v2 row are excluded: 55 rows.
  - Near-identical sections in different books (e.g. Biology 2e vs. Biology for AP Courses) stay in one split, and only one of them is used.
  - Held-out rows are drawn chapter by chapter, and any chapter that contributes a held-out row is excluded from train.
  - Maximum cross-split similarity: 0.496 (train/validation), 0.490 (train/test) and 0.436 (validation/test).
- **Fields per row:** `source_type=openstax_textbook_section`, `review_status=public_author_written_summary`, and `chapter`, `license` and `source_ref` (repository@commit:module) for traceability.

## Licence and attribution (required)

All 975 OpenStax rows come from books licensed **CC BY-NC-SA 4.0**. Reuse must be **non-commercial**, must **credit OpenStax**, and any redistributed derivative, including this dataset, must keep the **same licence**. Suggested attribution:

> Contains text from OpenStax textbooks (Rice University), https://openstax.org, licensed under CC BY-NC-SA 4.0. Text was converted to plain text and figures, exercises and boxed features were removed. Book, chapter, section and source commit are given per row.

## Rebuild

```
# 1. shallow-clone the repositories in dataset_manifest.json → openstax.repositories (collections/ and modules/ only) into RAW_DIR
python -I experiments/multitask_bloom_rewrite/scripts/clean_summarization_dataset.py          # data/summarization -> data/summarization_v2
python -I experiments/multitask_bloom_rewrite/scripts/import_openstax_summarization.py RAW_DIR candidates.jsonl
python -I experiments/multitask_bloom_rewrite/scripts/build_summarization_v3.py candidates.jsonl
python -I experiments/multitask_bloom_rewrite/scripts/attach_summarization_dataset.py --source data/summarization_v3 \
    --out data/multitask_bloom_rewrite_separate_eduguard_v3_6 --require-review-status "" --near-dup-threshold 0.5 \
    --check-tokens --write-config --config-name qwen15b_multitask_eduguard_v3_6_20261126
```

## Limitations

- **Two styles of reference summary:** OpenStax summaries are author-written but brief and selective, while the EduGuard rows are model-generated. Report summarization metrics per source type; the evaluation already breaks them down by `source_type`.
- **Textbook prose, not lecture slides:** the OpenStax rows are textbook text. Some books favour US-centred examples (e.g. U.S. History, American Government).
- **Uneven discipline coverage:** Biology is thin in OpenStax train rows (6, from Biology 2e; none from Anatomy and Physiology), because the biology textbooks overlap heavily and most of their chapters were quarantined for validation and test. Biology and life-science topics are still in train through Microbiology, Neuroscience and the EduGuard rows. Engineering is covered only through the EduGuard rows and one additive-manufacturing book.
