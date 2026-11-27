# EduGuard human evaluation: rater guidelines

Thank you for rating. You will judge model outputs for three tasks. Plan for about 4–5 hours in total; you can stop and continue at any time.

## What you receive

| File | Rows | What you rate |
|---|---:|---|
| `01_bloom_rewrite_R<n>.csv` | 144 | Exam questions rewritten to a requested Bloom level |
| `02_qa_R<n>.csv` | 124 | Short answers to academic questions (no source text given to the system) |
| `03_summarization_R<n>.csv` | 82 | Summaries of academic texts |
| `05_codebook.csv` | — | Allowed values and definitions for every rating column |

Open your own copy (`R1` or `R2`) in Excel or Google Sheets. Do not open the other rater's file, and do not discuss items with the other rater until you have both finished.

You are not told which system produced a row. Rows are shuffled, and some items appear more than once with different outputs. Rate each row on its own.

## How to fill in a sheet

- Fill only the columns to the right of the text columns (from `rater_id` onward). Keep `rater_id` as given.
- Use only the values listed in `05_codebook.csv`, spelled exactly (for example `yes`, `no`, `partially_correct`, `3`).
- Leave nothing blank except `notes`. If you cannot judge a row, write `skip` in `notes` and leave its ratings empty.
- Save as CSV (UTF-8) with the same file name.

## Task 1: Bloom rewrites (`01_bloom_rewrite`)

1. Read `source_question` and `rewritten_question`.
2. **Before** looking at `requested_bloom_level`, decide which level the rewritten question actually demands and enter it in `perceived_bloom_level`. Use the revised taxonomy (Anderson & Krathwohl, 2001):
   - Remember: recall facts or terms.
   - Understand: explain or summarize in one's own words.
   - Apply: use a method in a given case.
   - Analyze: break down, compare, or relate parts.
   - Evaluate: judge or justify against criteria.
   - Create: design or produce something new.
3. Then fill `matches_requested_level`, plus:
   - `topic_preserved`
   - `self_contained`: `no` if the question refers to material that was never given, for example "the supplied diagram".
   - `factually_sound`
   - `clarity_1to5`
   - `exam_usability_1to5`

## Task 2: Question answering (`02_qa`)

Compare `system_answer` with `reference_answer` and your own subject knowledge. The system answered from memory, without any source text.
- `correctness`: correct, partially correct, or incorrect.
- `contains_false_statement`: `yes` if any sentence is wrong, even when the main point is right.
- `completeness_1to5`
- `on_topic_and_concise`

## Task 3: Summarization (`03_summarization`)

Read `source_text`, then `system_summary`. `reference_summary` is there to help judge coverage and length; it is not the only acceptable summary.
- `faithful`: `yes` only if every claim in the summary is supported by the source.
- `unsupported_claim_count`: how many claims are not supported.
- `coverage_1to5`
- `fluency_1to5`
- `length`

## After rating

Return your CSV files to the study coordinator. The coordinator joins them with `04_annotation_key_DO_NOT_SHARE.csv`, which raters must not open, and runs `experiments/multitask_bloom_rewrite/scripts/analyze_human_annotation.py`.

## For the paper (coordinator)

- **Design:** blinded, within-item comparison of EduGuard v3.9 (LoRA) against the base Qwen2.5-1.5B-Instruct. For Bloom rewrites, the reference rewrite is also rated as a ceiling.
- **Sampling:** 8 validation source questions per requested level; the existing stratified QA (by question type) and summarization (by source type) samples. Seed 42. See `manifest.json` for SHA-256 hashes of all inputs.
- **Agreement:** two independent raters. Report Cohen's κ (categorical) and quadratic-weighted κ (ordinal and 1–5), as computed by the analysis script.
- **Rater description:** report each rater's subject expertise. Raters were not authors of the model outputs and did not see system identities.
