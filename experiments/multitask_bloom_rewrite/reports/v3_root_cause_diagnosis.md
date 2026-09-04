# Root-cause diagnosis — Bloom rewrite answer_output / format / Remember–Understand (STATIC)

**Status:** code/data preparation only. No training or evaluation was executed in this Cursor task.

## Key files inspected

| Area | Path |
|------|------|
| v2 synth policy/templates | `experiments/bloom_rewrite/bloom_target_policy.py` |
| v2 dataset builder | `experiments/bloom_rewrite/scripts/prepare_bloom_rewrite_dataset.py` |
| Active Bloom corpus | `data/bloom_rewrite/` (`bloom_rewrite_synth_v2`) |
| Archived v1 | `data/bloom_rewrite_versions/bloom_rewrite_synth_v1/` |
| Multitask prompts | `experiments/multitask_bloom_rewrite/prompts.py` |
| Format/semantic/cognitive validator | `experiments/multitask_bloom_rewrite/bloom_validation.py` |
| Multitask prepare | `experiments/multitask_bloom_rewrite/scripts/prepare_multitask_dataset.py` |
| Train | `experiments/multitask_bloom_rewrite/scripts/train_multitask_lora.py` |
| Eval | `experiments/multitask_bloom_rewrite/scripts/evaluate_rewrite.py` |
| Loss masking | `experiments/multitask_bloom_rewrite/loss_masking.py` |
| Baseline metrics | `experiments/multitask_bloom_rewrite/results/qwen15b_lora/metrics.json` |

## Diagnosis

### A. Likely cause of `answer_output_rate ≈ 42.9%`

1. **Validator false negatives on valid Apply imperatives** starting with `Given ` / `Using ` (not previously in `_looks_like_question` starters). Model copies v2 Apply templates → eval marks `answer_or_declarative`.
2. Model sometimes emits **declarative explanations** (true failures) — need stronger question-only SFT targets + prompt.
3. Multitask QA supervision teaches answering; Bloom system prompt was only weakly “output only one exam question”.

### B. Format failure (~56.8% format_valid)

Coupled to (A): format_valid requires `_looks_like_question`. Missing `Given`/`Calculate`-style acceptance inflated failures. Also genuine non-question outputs.

### C. Cognitive-validity failure (~57%)

`_cognitive_valid` requires question form first; when format fails, cognitive fails. Soft cue OR ≥4 content tokens is weak for distinguishing levels when templates are formulaic.

### D. Remember weakness (F1≈0.50)

Only ~3 Remember templates in v2; many targets are shallow “Identify/Name/State {garbled topic}”. Topic extraction often leaves clause fragments (“game2 can take to…”, truncated Maslow text), hurting distinctive Remember semantics → classifier confusion with Understand.

### E. Understand weakness (F1≈0.59)

Few Understand templates; heavy overlap with Remember surface forms (“Summarize/Describe {topic}”). Adjacent-level confusion expected.

### F. Template / memorization risk

v2 `REWRITE_TEMPLATES` / `_CLAUSE` banks are tiny (3 per level). High structural signature repetition (see planned diversity report). Model can memorize prefix patterns.

### G. Training / production prompt mismatch

Multitask + bloom_rewrite both use ChatML two-input `question + target_level`. Production GGUF path is separate and **must not be modified**. Prompt strengthening for v3 stays in experiment prompts only.

### H. Class balance

Multitask Mix A is 40/30/30. Within Bloom, non-self source→target cells exist but Remember/Understand **template diversity** is the bottleneck more than raw counts.

### I. Leakage risks

Group-based splits exist. v3 multitask builder **byte-freezes** `data/multitask_bloom_rewrite/test.jsonl` and filters v3 train/val against test keys.

### J. Validator FP/FN risks

- **FN (fixed in code):** rejecting valid `Given…` / `Calculate…` exam imperatives.
- **FP risk:** accepting long imperative that is still answer-like — mitigated by declarative opening detectors (`It is…`, `In conclusion…`).
- Thresholds were **not** weakened to inflate scores.

## Summarization audit (static)

With `max_seq_length=512` and `max_new_tokens=128`, PubMed articles/abstracts are likely truncated (char/token audit script provided). Low ROUGE may partly reflect **length caps**, not only model capacity. Optional `qwen15b_multitask_sumfix.json` isolates that follow-up.

## What v3 changes (code prepared)

- `bloom_target_policy_v3` + `prepare_bloom_rewrite_synth_v3.py`
- Stronger question-only system prompt
- Validator exam-imperative coverage
- Multitask corpus v3 with frozen test
- Train/eval configs under `*_v3` namespaces
- Error analysis / diversity / comparison / second-pass / human export / paper figures scripts
