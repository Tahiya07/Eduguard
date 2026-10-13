# BloomShift v0.3 expansion reconciliation and near-duplicate audit

Branch: `fix/arxiv-multitask-training`

## Scope

This audit fetched the three pending pool CSVs, all three `bloomshift_actionverb_v02` splits, and all three `bloomshift_final_candidate` splits from this branch. Source text was normalized with Unicode NFKC, lowercasing, and alphanumeric tokenization. Near-duplicate screening used token-set Jaccard similarity at a threshold of 0.80 with length-ratio prefilters. This is a candidate flagger, not a semantic duplicate adjudicator.

## Results

- Pending source pool rows: 2067; unique normalized questions: 2067.
- Pending rows still marked `pending` / `not_reviewed` / `not_checked`: 2067.
- Exact normalized overlap with current v0.2 and final-candidate source questions: 0.
- Near-duplicate candidate pairs within the expansion pool (Jaccard >= 0.80): 31.
- Near-duplicate candidate matches against the v0.2 and final-candidate source inventories (Jaccard >= 0.80): 10.
- Rows flagged by conservative source-quality heuristics for human review: 120 (ellipsis/placeholders, generic/context-dependent stems, possible truncation, or very short prompts). These flags are triage signals, not automatic rejection; short questions may be valid.
- All 2,067 pool questions occur in the Figshare CSV source text; this confirms their source-file presence, not permission to redistribute derivative content.
- Current v0.2 split counts: 691 / 138 / 138 (967 records). Final candidate counts: 660 / 114 / 114 (888 records). The final candidate is not a separate additional source inventory for this check; it overlaps the current rewrite corpus by design.

## Findings / action required

The exact-match screen is clean against these six rewrite files, but the pool is **not globally reconciled**: historical/cleaned inventories outside those paths were not yet exhaustively enumerated. The 31 internal near-duplicate candidate pairs and 10 cross-inventory near-match candidates are listed in the companion CSVs for human adjudication. Potentially incomplete or underspecified source questions are listed in `source_quality_flags.csv`; the source CSVs carry a matching heuristic status column. Several near matches differ in meaningful entities, quantities, negation, or technical terms, so automated similarity must never delete or merge them without review.

The source CSV is named `data/figshare_bloom_v1.csv`. A likely public source match is Figshare's *Exam Question Datasets* (CC BY 4.0; https://figshare.com/articles/dataset/Exam_Question_Datasets/22597957), which describes Bloom-labelled exam questions assembled from prior publications. However, filename and row examples alone do not conclusively establish that this exact repository file is byte-identical to a particular Figshare version/file. Before redistribution, record the exact Figshare version, file name, DOI, attribution, and any third-party rights/provenance restrictions. Do not infer the licence solely from the filename.


## Reconciliation update — 2026-10-13

This update supersedes the earlier statement that candidate inventories had not been enumerated. The earlier audit did not account for all expansion candidate files now present in this directory.

### Candidate inventory found on the repository branch

| File | Records observed | Unique source IDs observed | Status |
|---|---:|---:|---|
| `candidate_batch_0.3.1.jsonl` | 25 | 25 | All pending human validation; training_eligible=false |
| `candidate_batch_0.3.2.jsonl` | 27 | 27 | All pending human validation; training_eligible=false |
| `candidate_batch_0.3.3.jsonl` | 24 | 24 | All pending human validation; training_eligible=false |
| `candidate_batch_0.3.4.jsonl` | 30 | 30 | All pending human validation; training_eligible=false |
| `candidate_batch_0.3.5.jsonl` | 30 | 30 | All pending human validation; training_eligible=false |
| `candidate_batch_0.3.7_assistant_curated.jsonl` | 30 | 10 | All pending human validation; training_eligible=false; multiple target rewrites per source |
| `user_approved_candidates.jsonl` | 10 | 2 | Marked user-approved candidate, explicitly not human-validated |
| `action_verb_rewrite_batch_001_30_sources.csv` | 60 | 30 | Candidate-only; semantic/Bloom and human review pending |
| `action_verb_rewrite_pilot_001.csv` | 37 parsed data records | 12 | Candidate-only; semantic/Bloom and human review pending |

The first five JSONL batches contain 136 records across 136 distinct source IDs. The curated 0.3.7 batch adds 30 records over 10 sources, and the user-approved-candidate JSONL contains 10 records over two sources. The two action-verb CSVs are additional overlapping drafts, not additional unique source inventory. Across the nine files above, the direct inventory scan found 273 candidate rows/records and 154 unique source IDs; some records repeat the same source across different candidate files or target levels. CSV counts should be confirmed with a standards-compliant parser before being used as paper statistics.

### Published split files actually present

The branch tree for `Tahiya07/BloomShift` at `fix/action-verb-rewrites-expand-splits` contains `dataset/v0.2.0/{train,validation,test}.json` with 691 / 138 / 138 records (967 total), and `dataset/v0.1.0/{train,validation,test}.json` with 660 / 114 / 114 records (888 total). A repository-tree search on that branch did **not** find a `bloomshift_final_candidate` directory or an expanded 1,024-record split. Therefore the previously mentioned 1,024 total (768 / 153 / 103) is **not verified in the current GitHub branch** and must not be treated as the current canonical dataset until the exact files/commit are located.

The 273 candidate records checked here had no exact source-ID or normalized source-question match against the six published v0.1.0/v0.2.0 split files. This is a narrow exact-match result, not a full semantic duplicate audit: near-duplicate screening and historical inventories still require review. The source-question inventory has 2,067 rows and remains pending/not reviewed; source presence does not imply source eligibility.

### Decision and next actions

- Keep v0.2.0 (691 / 138 / 138) as the current published candidate split; do not silently add unreviewed drafts to train, validation, or test.
- Do not claim the 1,024-record split exists until its exact tracked paths and commit are identified.
- Do not run training on any of the new candidate files: the rewrite candidates are not human-validated, and the CSVs explicitly mark their semantic/Bloom statuses as pending.
- Before making a new split, adjudicate source eligibility and candidate feasibility, remove/resolve duplicates, validate semantic preservation and target Bloom operations, then split by source group (75/15/10) and record a reproducible split manifest. Hold out human-reviewed evaluation examples to avoid evaluating on synthetic drafts.
- Treat the 273 figure as a raw-file reconciliation count only, not a count of valid examples, independent source questions, or training-eligible rows.

## Release decision

**Not eligible for training/release yet.** Candidate rewrites exist, but none of the expansion records enumerated above is established as human-validated gold data. No independent annotator agreement, adjudication, or final validated 75/15/10 source-group split has been demonstrated. Keep v0.2.0 as the active published candidate input; do not train on the expansion candidates until the documented review gates are met.
