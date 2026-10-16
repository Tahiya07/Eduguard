# Qwen2.5-1.5B Experiment Matrix: A → B → C → D

## Experimental question

Determine whether Bloom rewriting is primarily limited by the base model's prompt-following behavior, the rewrite supervision, multitask interference, or task-loss weighting. Change one factor at a time and keep the benchmark fixed.

## Shared benchmark contract

- Base checkpoint: `Qwen/Qwen2.5-1.5B-Instruct`.
- Corpus: the current `data/multitask_bloom_rewrite_final/` after rebuilding and passing QC with the all-row BloomShift builder.
- Split for choosing the training strategy: **validation only**.
- Bloom: evaluate every BloomShift validation example (expected canonical count 212).
- QA: sample up to 300 validation examples from the split.
- Summarization: sample up to 300 validation examples from the split.
- Sampling sorts by stable record ID and uses a task-specific seed derived from 42; the evaluator writes selected IDs and hashes to `evaluation_subset_manifest.json`.
- Generation: greedy decoding; common `configs/generation.json`; maximum 128 new tokens unless changed explicitly before the experiment series.
- Bloom proxy metrics must use the same auxiliary classifier checkpoint for all conditions. Record its resolved path and hash with experiment results.
- Preserve source-group disjointness and exact corpus/manifest fingerprints for every run.
- Keep the test split locked until A/B/C/D comparison and selection are complete.

If a source refresh changes the dataset or selected IDs, rerun every condition on the same new benchmark rather than comparing partial results from different corpora.

## Conditions

| Condition | Model/training | Purpose |
|---|---|---|
| A | Base Qwen2.5-1.5B-Instruct, prompts only | Baseline ability without task fine-tuning |
| B | Bloom-rewrite-only LoRA trained on the 1,075 Bloom train records | Test whether the rewrite supervision alone improves target-level task generation |
| C | Existing three-task LoRA SFT trained with the current 40/30/30 example mixture and unweighted causal-LM loss | Baseline for multitask training/interference |
| D | Three-task LoRA SFT with the same examples and 40/30/30 sample mixture as C, but response-normalized task losses weighted 50/25/25 (Bloom/QA/summarization) | Isolate the effect of task-loss weighting |

### Important design choice for D

The 50/25/25 values are **loss weights**, not sampling proportions. Keep the same 40/30/30 selected training examples and seed as C. Normalize each example's causal-LM loss over unmasked assistant tokens, average example losses within each task, then combine:

```text
total_loss = 0.50 * bloom_loss + 0.25 * qa_loss + 0.25 * summarization_loss
```

This means an example's long output does not automatically receive more weight solely because it contains more response tokens. The exact implementation must document whether task losses are averaged by example and how batches with no examples for one task are handled.

## Metrics to compare

### Bloom rewrite (primary)
- Format-valid rate.
- Cognitive-operation heuristic pass rate, explicitly treated as a heuristic.
- Topic/lexical-overlap gate pass rate, not described as semantic equivalence.
- Trivial verb-swap rate.
- Auxiliary-classifier agreement and confidence, reported as diagnostic proxies only.
- Per-target-level results for Remember, Understand, Apply, Analyze, Evaluate, and Create.
- Human-reviewed target-level correctness, meaning/context preservation, and pedagogical validity when real blinded ratings are available.

### QA (guardrail)
- SQuAD Exact Match and token F1.

### Summarization (guardrail)
- ROUGE-1, ROUGE-2, and ROUGE-L; interpret these as lexical overlap metrics, not complete factuality evaluation.

### Efficiency
- Load time, per-example latency, peak memory, GPU/CPU details.

## Order of operations

1. Rebuild the corpus and confirm Bloom counts of 1075/212/135, zero source-group overlap, and all-row inclusion across the workflow.
2. Run A on validation and save `metrics.json`, `predictions.jsonl`, `evaluation_subset_manifest.json`, and the environment metadata.
3. Run B on the same validation benchmark.
4. Run C using the existing adapter. Record exactly which checkpoint/run produced it; do not call it a current-data baseline unless its training manifest proves that.
5. Implement and train D only after A/B/C results are available. Use a fresh output directory.
6. Compare validation results on identical IDs and choose the approach.
7. Evaluate the chosen candidate on the locked test split once. Do not tune on test results.

## Experiment A command

After the all-row dataset preparation and QC both pass:

```powershell
python experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py --config experiments/multitask_bloom_rewrite/configs/experiment_A_base_prompt_only.json --condition base --split validation --evaluation-per-task 300
```

The run requires the same fixed auxiliary Bloom classifier used by the other conditions. If automatic detection selects the wrong classifier, pass the same verified path explicitly using `--classifier-dir`.

## Interpretation guardrail

No metric is a substitute for human assessment of cognitive demand and preservation. Keep validation metrics as model-selection evidence, and only use the untouched test split for the final held-out evaluation.
