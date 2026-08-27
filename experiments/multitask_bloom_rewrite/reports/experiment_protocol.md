# Experiment Protocol (FROZEN before final test evaluation)

**Experiment:** `experiments/multitask_bloom_rewrite`  
**Status:** Protocol locked for execution. Test-set outcomes must not rewrite this document.

## 1. Research Questions

Primary: Does multi-task LoRA fine-tuned Qwen2.5-0.5B or Qwen2.5-1.5B provide the better quality–resource trade-off for offline academic Q&A, scientific summarization, and target-level Bloom rewriting?

Secondary: RQ1–RQ6 as specified in the master prompt (Bloom vs base; size effect; QA/summarization retention; resource justification; Q4_K_M delta).

## 2. Datasets

| Source | Task | Notes |
|--------|------|-------|
| `data/figshare_bloom_v1.csv` | Bloom source pool | Classification only |
| `bloom_rewrite_synth_v2` | Bloom rewrite | Synthetic supervision; hash must match expected |
| `rajpurkar/squad` | QA | Official train; official validation bipartitioned into exp val/test. **Loader:** official SQuAD 1.1 JSON (`squad_loader.py`) because HF `datasets` 2.19.1 fails on the current `rajpurkar/squad` card (`TypeError` dataclass). Content remains SQuAD 1.1. |
| `ccdv/pubmed-summarization` | Summarization | Official train/validation/test preserved |

## 3. Model Conditions

A. Qwen2.5-0.5B base  
B. Qwen2.5-0.5B + multi-task LoRA  
C. Qwen2.5-1.5B base  
D. Qwen2.5-1.5B + multi-task LoRA  

## 4. Training Method

- PEFT LoRA, HF Transformers, assistant-only loss (`-100` on prompt tokens)
- Shared hyperparameters except `model_id` / output paths
- Seed 42; Mix A (40/30/30 Bloom/QA/Sum) pre-registered
- Best checkpoint by **validation loss only**

## 5. Prompts

Task instructions are explicit in ChatML. Bloom is **question + target_level → rewrite** only (no source Bloom in prompt).

## 6. Splitting / Leakage

Train ∩ validation ∩ test = ∅ at source-question/group (Bloom) or source_id (QA/Sum).  
SQuAD: no public labeled test → deterministic half split of official validation.

## 7. Evaluation

- QA: EM, token F1 (SQuAD held-out; EduGuard RAG separately if available)
- Summarization: ROUGE-1/2/L (+ BERTScore if reproducible)
- Bloom: classifier target accuracy, macro/weighted F1, fully validated rewrite rate, 6×6 matrix
- Human: blinded ≥36 items; ratings 1–5; agreement when multi-rater
- Statistics: paired tests; Holm/BH when multiple comparisons apply
- GGUF: Q4_K_M retention + deployment benchmark

## 8. Selection Rule

See `configs/decision_rule.json` (frozen). Incomplete evidence → `INCONCLUSIVE`.

## 9. Hardware Gate

If `check_resources.py` fails: **TRAINING NOT STARTED — INSUFFICIENT RESOURCES**.

## 10. Synthetic Data Limitation

Bloom rewrite targets are synthetic. Template memorization must be reported. Final paper must state this limitation.
