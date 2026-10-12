# BloomShift v0.3.0 expansion in EduGuard

## Current status

This branch contains a pending source-question pool, not a completed rewrite dataset. The existing `bloomshift_actionverb_v02` files are preserved and unchanged.

The current pool has 2,067 unique normalized source questions not found by exact normalized-text matching in the three `bloomshift_actionverb_v02` train/validation/test files. This count is specific to those inventories; it must be recomputed against every historical/cleaned source inventory before calling the questions globally unused.

## Files

- `source_pool_pending_review_part_01.csv`: 700 source questions
- `source_pool_pending_review_part_02.csv`: 700 source questions
- `source_pool_pending_review_part_03.csv`: 667 source questions
- `source_pool_manifest.json`: counts and limitations

All rows remain `pending` and `not_reviewed`. A heuristic near-duplicate screen has run: candidate pairs are listed in `near_duplicate_pool_pairs.csv`, cross-inventory near matches in `near_duplicate_existing_matches.csv`, and potential source-quality concerns in `source_quality_flags.csv`. Pool rows are marked as either `flagged_for_human_review` or `screened_no_flag_at_jaccard_0.80`; this is not human adjudication. No new rewrites have been created by adding these source records.

## Draft generation workflow added in v0.3

The candidate list `source_candidates_for_rewrite.csv` contains 2,007 source-question candidates selected by simple quality heuristics. They are not human-approved, and the source pool has not been proven globally unused across every historical inventory.

- `ACTION_VERB_REFERENCE_BANK.md` documents a diverse opening-verb bank cross-checked against university teaching/assessment resources.
- `experiments/bloomshift_expansion/build_candidate_generation_queue.py` creates one draft job per source × each of the five target levels different from its supplied source label. It intentionally permits pending source rows for *draft generation only*; this is not source approval.
- `experiments/bloomshift_expansion/generate_rewrite_candidates.py` runs the queue using a local Hugging Face causal model and saves raw output, parsed JSON, model identifier, timestamp, decoding settings, and unreviewed status. It supports `--resume`.
- `experiments/bloomshift_expansion/audit_raw_generation_candidates.py` checks JSON structure, target labels, duplicate source-target pairs, and opening-verb formatting. Passing this audit is not semantic or pedagogical validation.

Expected queue size is at most 10,035 jobs (2,007 × 5), before missing/duplicate/invalid rows and source-label effects. Some transformations must be marked not feasible. Do not force a rewrite merely to fill a target level, and do not train on raw outputs.

Example from repository root (first test only; use a locally available model path):
```bash
python experiments/bloomshift_expansion/build_candidate_generation_queue.py --source-csv "data/rewrite dataset/bloomshift_expansion_v0.3/source_candidates_for_rewrite.csv" --output "data/rewrite dataset/bloomshift_expansion_v0.3/generation_queue.jsonl" --limit-sources 2
python experiments/bloomshift_expansion/generate_rewrite_candidates.py --queue "data/rewrite dataset/bloomshift_expansion_v0.3/generation_queue.jsonl" --model "models/qwen.gguf" --output "data/rewrite dataset/bloomshift_expansion_v0.3/raw_candidates_smoke.jsonl" --limit 10
python experiments/bloomshift_expansion/audit_raw_generation_candidates.py --input "data/rewrite dataset/bloomshift_expansion_v0.3/raw_candidates_smoke.jsonl" --report "reports/bloomshift_expansion/raw_candidates_smoke_audit.json"
```

The generator supports `.gguf` via `llama-cpp-python` and local Hugging Face causal-LM checkpoints via `transformers`. Use the actual local checkpoint path; the GGUF file must be a question-generation-capable instruction model. If `models/qwen.gguf` is the deployed runtime model, verify its model card/configuration and run the smoke test before the full queue. For a Hugging Face checkpoint, add `--local-files-only` to prohibit downloads.\n\nAfter the smoke test, build the full queue without `--limit-sources`, generate in resumable batches (use `--resume` on continuation), inspect failures, and review candidates before any dataset split or training.

## Required workflow

1. Enumerate and audit every existing source inventory, including historical rewrite inventories and any separate classifier or multitask dataset sources. Use `experiments/bloomshift_expansion/audit_expansion_source_pool.py` with repeated `--existing` arguments; the current committed audit only covers v0.2 and final-candidate split files. Re-run the source-pool preparation script only if the pool needs to be rebuilt.
2. Review source quality, ambiguity, provenance/license, and exact/near duplicates. Do not assume every source supports all six target levels.
3. Record source approval in `review_status`; separately annotate each source-target pair's feasibility in a CSV with columns `source_id,target_bloom_level,feasible,reviewer_notes`.
4. For a conservative, human-approved queue, run `build_generation_queue.py` only for approved sources and feasible target-level pairs. For broad exploratory candidate generation, use the new candidate queue above; keep its outputs explicitly unvalidated.
5. Generate with a pinned model/checkpoint, prompt version, decoding parameters, timestamp, and raw-output retention. Manually review semantic preservation, target cognitive operation, clarity/answerability, and pedagogical validity.
6. Group exact and near-duplicate source questions before a deterministic 75/15/10 source-group split. Ensure no source group crosses partitions. Keep held-out human evaluation examples out of training and model selection.
7. Publish a new train/validation/test candidate and complete audit manifest only after generation, review, and structural checks. Human validation requires independent annotators, agreement statistics, adjudication, and explicit denominators.

## Important limitations

- An action verb at the start is a formatting constraint, not proof of cognitive validity.
- A source-pool count is not a rewrite-record count or a count of validated examples.
- Existing v0.2.0 files remain untouched until a new candidate passes the release gate.
