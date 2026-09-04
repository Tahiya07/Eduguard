# UNIVERSITY GPU — RUN THESE COMMANDS

Preparation-only code is already in the repo. Run the following **serially** on the university machine.

Assume:

- `D:\Eduguard`
- venv already exists
- RTX 4090 available
- packages already installed

```powershell
cd D:\Eduguard

# 0) Sanity
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# STEP 1 — Generate bloom_rewrite_synth_v3
.\.venv\Scripts\python.exe experiments/bloom_rewrite/scripts/prepare_bloom_rewrite_synth_v3.py `
  --figshare data/figshare_bloom_v1.csv `
  --output-dir data/bloom_rewrite_versions/bloom_rewrite_synth_v3 `
  --seed 42 `
  --freeze-test-from data/bloom_rewrite/test.jsonl

# STEP 2 — Validate/hash/report v3 + diversity vs v2
.\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/compare_bloom_diversity_v2_v3.py `
  --v2 data/bloom_rewrite/train.jsonl `
  --v3 data/bloom_rewrite_versions/bloom_rewrite_synth_v3/train.jsonl `
  --output experiments/multitask_bloom_rewrite/reports/diversity_v2_vs_v3.json

.\.venv\Scripts\python.exe -m unittest experiments.multitask_bloom_rewrite.tests.test_question_answer_detector -v

# STEP 3 — Build multitask Mix-A v3 (FREEZES exact baseline test.jsonl)
.\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset_v3.py `
  --locked-multitask-dir data/multitask_bloom_rewrite `
  --bloom-v3-dir data/bloom_rewrite_versions/bloom_rewrite_synth_v3 `
  --output-dir data/multitask_bloom_rewrite_v3 `
  --seed 42

# STEP 3b — Error analysis on EXISTING 1.5B baseline predictions (no training)
.\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/analyze_bloom_prediction_errors.py `
  --predictions experiments/multitask_bloom_rewrite/results/qwen15b_lora/predictions.jsonl `
  --output experiments/multitask_bloom_rewrite/results/qwen15b_lora/error_analysis_v3.json

# STEP 3c — Summarization length audit (no inference)
.\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/audit_summarization_lengths.py `
  --dataset data/multitask_bloom_rewrite/train.jsonl `
  --predictions experiments/multitask_bloom_rewrite/results/qwen15b_lora/predictions.jsonl `
  --max-seq-length 512 `
  --max-new-tokens 128 `
  --output experiments/multitask_bloom_rewrite/reports/summarization_length_audit.json

# STEP 4 — Train 1.5B LoRA v3 (same hypers as baseline; new output dir)
.\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py `
  --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_v3.json

# STEP 5 — Evaluate 1.5B LoRA v3 on frozen test
.\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py `
  --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_v3.json `
  --condition lora

# STEP 6 — Compare baseline vs v3
.\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/compare_baseline_vs_v3.py `
  --baseline-metrics experiments/multitask_bloom_rewrite/results/qwen15b_lora/metrics.json `
  --improved-metrics experiments/multitask_bloom_rewrite/results/qwen15b_lora_v3/metrics.json `
  --output experiments/multitask_bloom_rewrite/results/qwen15b_lora_v3/baseline_vs_v3_comparison.json

# STEP 7 — Export blinded human-eval sample
.\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/export_human_eval_baseline_vs_v3.py `
  --baseline-predictions experiments/multitask_bloom_rewrite/results/qwen15b_lora/predictions.jsonl `
  --improved-predictions experiments/multitask_bloom_rewrite/results/qwen15b_lora_v3/predictions.jsonl `
  --sample-size 100 `
  --seed 42 `
  --output experiments/multitask_bloom_rewrite/human_eval/blinded_baseline_vs_v3.jsonl

# OPTIONAL — second-pass rewrite stats (separate from primary metrics)
.\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/second_pass_rewrite.py `
  --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_v3.json `
  --condition lora `
  --output-dir experiments/multitask_bloom_rewrite/results/qwen15b_second_pass_v3

# OPTIONAL later — LoRA r=32 controlled experiment (NOT primary)
# .\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py `
#   --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_v3_r32.json

# OPTIONAL later — summarization length follow-up (only if audit recommends)
# .\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py `
#   --config experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_sumfix.json

# PAPER FIGURES / TABLES — final deployed model only (after choosing final metrics.json)
.\.venv\Scripts\python.exe experiments/multitask_bloom_rewrite/scripts/generate_paper_figures.py `
  --metrics experiments/multitask_bloom_rewrite/results/qwen15b_lora_v3/metrics.json `
  --confusion experiments/multitask_bloom_rewrite/results/qwen15b_lora_v3/confusion_matrix.json `
  --failure-analysis experiments/multitask_bloom_rewrite/results/qwen15b_lora_v3/failure_analysis.json `
  --output-dir experiments/multitask_bloom_rewrite/paper_figures
```

## Do not overwrite

- `experiments/multitask_bloom_rewrite/models/qwen15b_multitask_lora/`
- `experiments/multitask_bloom_rewrite/results/qwen15b_lora/`
- `data/bloom_rewrite/` (v2)
- production `models/qwen.gguf`
