# EduGuard Final Validation Report

**Purpose:** Lock provenance for the publication pack.  
**Script:** `scripts/final_paper_experiments.py`  
**Generated:** see `paper/paper_experiment_manifest.json` → `timestamp_utc`  
**Git revision:** `76c8d276ee78efd6096c9387c15b22522b056c65` (workspace at audit time; re-run script to refresh)

---

## A. Final selected models

| Role | Model |
|------|--------|
| Bloom classifier | Qwen2.5-0.5B-Instruct sequence-classification + LoRA |
| Multitask generator | Qwen2.5-1.5B-Instruct causal LM + multitask LoRA **v3** |
| Local generator deploy | Q4_K_M GGUF of 1.5B multitask v3 |
| Embeddings / retrieval | BGE-small + FAISS (implementation; no full RAG benchmark) |

## B. Final selected checkpoints

| Component | Path (as recorded in artifacts) |
|-----------|----------------------------------|
| FL LoRA best | `artifacts/federated/models/qwen_bloom_federated0.5B_fedprox_iid_r20_best` |
| FL merged (eval pack) | `artifacts/federated/global/qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged` |
| Central merged 0.5B | `models/qwen_bloom_merged0.5B` |
| Multitask v3 adapter | `experiments/multitask_bloom_rewrite/models/qwen15b_multitask_lora_v3/best_adapter` |
| GGUF | `models/gguf/qwen15b_multitask_v3_q4_k_m.gguf` |

**Selection rule (FL):** max validation accuracy; ties by Macro-F1 then QWK. Winner: `fedprox_iid_r20`, best round **20**.

## C. Dataset hashes

| Dataset | Hash / identifier |
|---------|-------------------|
| Figshare train CSV | `d5cd8c28…` (from `federated_lora_fedprox_iid_r20.json`) |
| Figshare val CSV | `14cd9304…` |
| Figshare test CSV | `7ee161c5…` |
| Multitask locked test SHA256 | `1eb88cd5900fcb9154cbf70e6a62d49636c9ad882aa5d8e303285524c74cc391` |
| Multitask v3 corpus_hash | `eaaa88491f32090d246673356f75a629611dfcba7846f22d89b9340bb8d10ebd` |
| Eval `dataset_hash` (locked corpus id in metrics) | `79fe6c7b7680839ef5ddc8c5065fbb183d9ef30f9a6c605392f04db2b4c666f4` |

## D. Test counts

| Eval | N |
|------|---|
| Bloom classification test | 350 |
| Multitask frozen test | 8321 (Bloom 1536 / QA 5285 / Sum 1500) |
| PrivacyGuard taxonomy | 104 attacks + 15 benign + 6 teacher |
| Privacy ablation | 1344 attacks / 64 benign / 1416 cases |
| GGUF micro-bench prompts | 50 |

## E. Training configurations (selected)

**Classifier FL (FedProx r20):** AdamW lr 1e-4, batch 2, grad accum 8, local epochs 3, rounds 20, clients 8, μ=0.01, LoRA r=32 α=64 dropout 0.1, label smoothing 0.05, class weights on, seed 42.

**Generator v3:** LoRA r=16 α=32, max_seq_length 512, max_new_tokens 128, epochs 3, lr 2e-4, assistant-only loss, greedy decode at eval, seed 42, Mix-A 40/30/30.

## F. Federated configurations executed

IID FedAvg/FedProx (5r and 20r); FedAvg non-IID α∈{0.1,0.5,1.0} (5r); FedProx non-IID α=0.5 (5r); DP-FedProx utility run (excluded from claims).

## G. Final numerical metrics (locked)

### Bloom classifier (test N=350)

| Model | Acc | Macro-F1 | QWK | Within-1 | Severe | ECE |
|-------|-----|----------|-----|----------|--------|-----|
| TF-IDF+SVM | 0.7514 | 0.7253 | 0.7471 | 0.8829 | 0.0657 | — |
| Central 0.5B | 0.8314 | 0.8252 | 0.8794 | 0.9200 | 0.0371 | 0.0542 |
| FedAvg r20 best | 0.8000 | 0.7920 | 0.8412 | 0.8943 | 0.0486 | 0.0395 |
| **FedProx r20 best** | **0.8057** | **0.7982** | **0.8493** | **0.9086** | **0.0429** | **0.0537** (live) |

Gap central→FedProx Acc: 0.8314→0.8057 = **−2.57 pp**.

### Multitask v3

| Task | Metrics |
|------|---------|
| Bloom (merged clf) | Acc 0.888021, Macro-F1 0.876835, FV 0.707031 |
| Bloom (FedProx clf) | Acc 0.951823, Macro-F1 0.944287, FV 0.76237 |
| QA | EM 0.57843, F1 0.785403 |
| Sum | R1 0.249109, R2 0.057348, RL 0.149551 |

### GGUF

Size 986048032 B; startup 0.687 s; mean latency 0.6865 s; tok/s 28.8944; RSS 1300.01 MB; USS 694.46 MB.

### Privacy

Attack block 1.0 (104/104); benign allow 0.0 (15 prompts).

## H. Existing / reused figures

From `artifacts/evaluation/paper/` (FedProx r20 pack, 2026-09-03):  
`fig_confusion_matrix`, `fig_learning_curves`, `fig_best_vs_final`, `fig_per_class_f1`, `fig_reliability` (png+pdf).

## I. Newly generated figures

`fig01_system_architecture`, `fig04_source_target_accuracy`, `fig05_source_target_fully_validated`, `fig06_multitask_task_performance`, `fig07_gguf_deployment`, `fig08_privacy_ablation`, plus numbered aliases `fig02_*`, `fig03_*`.

## J–K. Tables

Generated under `paper/tables/` and copied to IEEE/Elsevier `tables/`:  
`table1_datasets` … `table7_gguf_deployment`, `table2b_fedavg_vs_fedprox`.

## L–M. Excluded experiments

See `paper/final_experiment_inventory.md` (pre-v3 multitask dirs; sumfix as non-primary; DP-FL utility; Acc 0.840 unified row; human ratings absent).

## N. Missing evidence

- Complete 1.5B classifier metrics JSON  
- FedProx non-IID α=0.1 and α=1.0 result files  
- Human rewrite ratings  
- Complete quantitative EduGuard RAG / OCR benchmarks  
- GGUF conversion meta and binary may be absent on this machine  
- FL adapter weight directories may be gitignored  

## O. Reproducibility status

`python scripts/final_paper_experiments.py --all` regenerates figures/tables from locked JSON without retraining. Metric validation gate: **ok=True** at last run.

## P. Git commit identifiers

Workspace HEAD at audit: `76c8d276ee78efd6096c9387c15b22522b056c65`.  
FL run recorded git: `5914e73d…` (in fedprox_r20 JSON).  
v3 train_meta git: `99b5d29e…`.

## Q. Python / package versions

v3 eval environment (from metrics): torch 2.13.0+cu126, transformers 5.16.1, peft 0.20.0, numpy 2.2.6.  
Paper script host: whatever local Python runs matplotlib/numpy.

## R. Hardware information

v3 eval recorded: NVIDIA GeForce RTX 4090; CUDA available.  
GGUF micro-benchmark: CPU-oriented latency/memory fields in `q4_k_m_benchmark.json` (threads=8).

## Disagreement resolutions

1. **ECE 0.0537 vs 0.0565:** Headline uses live_eval 0.0537; disclose stored 0.0565.  
2. **Central Acc 0.8314 vs unified 0.840:** Use 0.8314 only.  
3. **sumfix vs “rewrite degraded”:** Target Acc rose slightly; fully_validated and cognitive validity fell — paper uses the latter for trade-off.  
4. **DP:** Later validation lock passed; DP-FL Acc collapsed — no formal DP claim in contributions.
