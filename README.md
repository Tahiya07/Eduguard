# EduGuard

Offline academic assistant and the locked paper evidence for that system.

## Attach this with the manuscript

```text
paper/attachment/
```

Rebuild it after any paper-artifact change:

```bash
python scripts/build_paper_attachment.py
```

That folder holds the eight figures, final tables, provenance files, and both LaTeX sources. It does not include model weights.

Editable sources (do not edit the attachment by hand):

| Item | Path |
|------|------|
| IEEE manuscript | `paper/ieee_openjournal/manuscript.tex` |
| Elsevier manuscript | `paper/elsevier/manuscript.tex` |
| Shared figures | `paper/figures/` |
| Shared tables | `paper/tables/` |
| Inventory | `paper/final_experiment_inventory.md` |
| Validation | `paper/final_validation_report.md` |

## Repository map

| Path | What it is |
|------|------------|
| `backend/`, `frontend/` | Running application |
| `privacy/` | PrivacyGuard |
| `retriever.py`, `ingestion.py` | Local retrieval and document intake |
| `training/`, `experiments/federated/` | Bloom classifier, including FedAvg and FedProx |
| `experiments/multitask_bloom_rewrite/` | 1.5B v3 generator. Pre-v3 result folders are historical only |
| `artifacts/` | Stored metrics. Paper numbers are taken from the files named in the validation report |
| `scripts/final_paper_experiments.py` | Regenerates figures and tables from those metrics |
| `scripts/proof3_rag_retrieval_eval.py` | Retrieval proof. Current status is not paper-ready |
| `scripts/proof4_protected_content_eval.py` | Attack block, leakage, and false-block rates |
| `data/` | Figshare splits and multitask JSONL |
| `models/` | Local weights (not part of the paper attachment) |

## Selected configuration

- Bloom classifier: FedProx IID, 20 rounds, best round 20, Qwen2.5-0.5B sequence classification
- Generator: Qwen2.5-1.5B multitask LoRA v3
- Deployment: generator only, Q4_K_M GGUF. The classifier stays a sequence-classification model

Pre-v3 multitask runs, the unified-table accuracy of 0.840, empty human ratings, and the differential-privacy utility collapse are not main-paper evidence. See `paper/final_experiment_inventory.md`.
