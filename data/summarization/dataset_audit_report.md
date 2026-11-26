# EduGuard summarization dataset audit report

Audit date: 2026-11-26. Data: `data/summarization/{train,validation,test}.jsonl`. Manifest: `dataset_manifest.json`.

**Decision (2026-11-26): frozen as summarization v1 with 713 records (592 / 64 / 57).** The 1,465-record plan is not met; generation was stopped in favour of a verified, smaller set. It is attached to training as `data/multitask_bloom_rewrite_separate_eduguard_v3_3` with config `qwen15b_multitask_eduguard_v3_3_20261126.json`.

**Status vs original plan: INCOMPLETE.** 713 of 1,465 records exist; 752 are still required. LecSumm supplied 0 eligible records (section 3); 240 quality-filtered arXiv records were merged (section 3a).

## 1. Actual vs target

| Category | Train | Validation | Test | Total |
|---|---|---|---|---|
| short | 235/235 | 29/29 | 29/29 | 293/293 |
| medium | 188/352 | 4/44 | 3/44 | 195/440 |
| long | 46/410 | 9/51 | 8/52 | 63/513 |
| very_long | 123/175 | 22/22 | 17/22 | 162/219 |
| **total** | 592/1172 | 64/146 | 57/147 | 713/1465 |

Remaining per split (target minus existing valid unique records): train 580, validation 82, test 90; total 752.

Word counting: whitespace-delimited tokens of `source_text` (`len(text.split())`). Ranges: short 300–600, medium 601–1,200, long 1,201–2,500, very long 2,501–4,000. The Short block is the pre-existing UAP-SUM-0001..0293 records.

## 2. Integrity checks (automated)

- Records parse as JSON, schema keys exact: {'performed': True, 'records_with_wrong_keys': 0, 'generated_records_with_wrong_metadata': 0, 'external_records_keep_own_metadata': 269}.
- Unique IDs: 713 of 713; duplicate IDs: 0.
- Exact duplicate sources: 0; exact duplicate summaries: 0.
- Max cross-split source similarity (threshold 0.50): {'train/validation': 0.4815, 'train/test': 0.4761, 'validation/test': 0.4136}. No near-duplicate crosses splits.
- Near-duplicate pairs within splits: 111 (all inside train, from the original Short block, kept by owner decision); repeated topic names: 23.
- Source types: {'original_academic_example': 444, 'arxiv_scientific_article': 240, 'lecture_slide_text_style': 29}; review status: {'generated_unverified': 444, 'quality_filtered_unverified': 240, 'curriculum_verification_pending': 29}.
- Known deviation: 118 Short records are under 300 words (kept as authored, all in train).
- Summary-to-source factual consistency: NOT verified. Generated records were written with care but are model-generated (`generated_unverified`); no automated or human factual-consistency check has been run.

## 3. LecSumm (https://github.com/DeakinAINLP/LecSumm, commit df266f6)

- Structure: 10 topic folders, each with ONE lecture note (`Input.txt`) and 200 student summaries of that same note. LecSumm therefore has 10 source documents, not ~2,000.
- Licence: none. No LICENSE file, GitHub licence field null, README states no terms. The paper's CC BY 4.0 (ACL Anthology) covers the article only. Reuse permission is not established.
- Labels: folder names are shifted by one topic relative to content (e.g. folder 'Topic3-Clustering' contains the data-wrangling note; 'Topic10-Neural Networks' contains ensemble learning). Summaries match the content, not the folder name.
- Lengths: 8 of 10 notes are 4,668–11,777 words, above the 4,000-word ceiling; trimming sources is prohibited, so they are ineligible.
- The 2 in-range notes (ensemble learning 2,360 words = long; data wrangling 3,979 words = very long) are scraped course pages with ~10% code tokens, notebook/widget imports, 'Output:' markers and course-platform instructions (Deakin unit codes); not coherent academic passages without editing the source.
- Accepted: 0. Rejected: 10 (details per note in `lecsumm_provenance.csv`). Summaries were not evaluated because no source note was eligible.

## 3a. arXiv (ccdv/arxiv-summarization, via the earlier quality-filtered subset)

- Pool: 1,422 arXiv rows used in earlier EduGuard multitask datasets; 1,032 had already been excluded by the `arxiv_summarization_quality_v1_20261125` filter for defects in the target abstract (863 unrecoverable @xmath math placeholders, 386 incomplete endings, 147 LaTeX residue, others). Only the 390 eligible rows were considered; excluded rows were not reused.
- Mapping: `article` -> `source_text`, author abstract -> `summary`. Original row IDs kept (e.g. `arxiv_train_0000010`). No department/course/topic exists in the source, so none was invented: department `Unspecified (arXiv)`, course `arXiv scientific article`, topic names the row. `source_type` = `arxiv_scientific_article`, `review_status` = `quality_filtered_unverified`.
- Splits: original arXiv split kept; 9 very-long validation papers beyond the 22-slot target were moved to train (log in merge_log_arxiv.json). Papers are distinct documents, so this creates no leakage.
- Merged: 240 (medium 15, long 63, very long 162). Rejected: 150 on length (145 over 4,000 words, 5 short-length while Short is full). No duplicate or near-duplicate rejections.
- Known quality limits: source text is lowercased and tokenized ("word , word ."), and articles still contain unrecoverable @xmath placeholders where equations were (median about 109 per article; abstracts contain none). Text was not edited.
- Licence: arXiv papers carry individual author licences and the derived dataset's terms were not verified here; treat reuse rights as unverified.

## 4. Shortage

752 further records are needed: medium 245 (164/40/41); long 450 (364/42/44); very_long 57 (52/0/5). They require further original academic examples or another licensed source.

## 5. Files

- `data/summarization/train.jsonl`, `validation.jsonl`, `test.jsonl` — current records (unchanged by this audit)
- `data/summarization/dataset_manifest.json` — counts, categories, source types, duplicate checks, SHA-256 hashes
- `data/summarization/lecsumm_provenance.csv` — the 10 LecSumm lecture notes inspected, with decisions
- `data/summarization/{train,validation,test}_arxiv.jsonl` and `merge_log_arxiv.json` — staged arXiv inputs and per-row merge decisions
- `data/summarization/dataset_audit_report.md` — this report
