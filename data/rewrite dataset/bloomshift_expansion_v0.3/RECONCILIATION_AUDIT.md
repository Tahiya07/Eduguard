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

## Release decision

**Not eligible for training/release yet.** No new rewrites have been generated; no source has been approved; no source-target feasibility decisions, semantic/Bloom annotations, independent annotator agreement, adjudication, or final 75/15/10 source-group split exist. Keep the v0.2 dataset as the active training input. Human review cannot be replaced by this automated audit.
