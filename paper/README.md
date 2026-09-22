# EduGuard paper pack (Overleaf)

## Reproduce artifacts (no retraining)

```bash
python scripts/final_paper_experiments.py --all
```

Other modes: `--audit`, `--validate`, `--figures`, `--tables`.

## Priority proofs ③ and ④ (kept scripts)

```bash
# ③ RAG — Recall@1/3/5 + MRR (smoke summary is NOT paper-ready)
python scripts/proof3_rag_retrieval_eval.py --summarize-existing
python scripts/proof3_rag_retrieval_eval.py --write-scaffold
python scripts/proof3_rag_retrieval_eval.py --run-benchmark data/rag_benchmark/queries.jsonl --corpus data/rag_benchmark/corpus.jsonl

# ④ Protected content — attack block / leakage / false-block
python scripts/proof4_protected_content_eval.py --from-artifact --include-ablation
python scripts/proof4_protected_content_eval.py --rerun
```

Outputs land under `paper/proofs/proof3_rag/` and `paper/proofs/proof4_privacy/`.

## Manuscripts

| Venue | Path |
|-------|------|
| IEEE Open Journal (LaTeX) | `paper/ieee_openjournal/manuscript.tex` |
| IEEE bibliography | `paper/ieee_openjournal/references.bib` |
| Elsevier (LaTeX) | `paper/elsevier/manuscript.tex` |
| Elsevier bibliography | `paper/elsevier/references.bib` |

Figures: `paper/*/figures/` (also `paper/figures/`).  
Tables: `paper/*/tables/` (also `paper/tables/`).

## Provenance docs

- `paper/final_experiment_inventory.md`
- `paper/final_validation_report.md`
- `paper/paper_experiment_manifest.json`

## Attachable bundle

```bash
python scripts/build_paper_attachment.py
```

Upload `paper/attachment/`. Do not edit files inside it; edit the sources under `ieee_openjournal/`, `elsevier/`, `figures/`, and `tables/`, then rebuild.

1. Create a new project.
2. Upload `manuscript.tex`, `references.bib`, and the `figures/` folder for the chosen venue.
3. For IEEE: compiler `pdfLaTeX`, document class `IEEEtran` (add `IEEEtran.cls` if the template is not built-in).
4. For Elsevier: use the `elsarticle` template from Elsevier / Overleaf gallery, then replace with `manuscript.tex`.
5. Fill author blocks before submission.
6. Re-check every number against `final_validation_report.md` after any edit.

## Integrity rules

Do not reintroduce excluded multitask dirs (`qwen05b_lora`, `qwen15b_lora`, `qwen15b_lora_sumfix`) as primary evidence.  
Do not claim formal DP.  
Do not invent RAG/human-eval numbers.
