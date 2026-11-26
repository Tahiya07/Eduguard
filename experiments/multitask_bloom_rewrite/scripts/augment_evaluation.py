#!/usr/bin/env python
"""Calibrated, post-hoc additions to an evaluate_rewrite.py run (no regeneration).

The original metrics are left untouched. This adds measurements that make them
interpretable, each computed the same way for the model and for the human
reference outputs:

1. Reference ceilings (Bloom). The fixed 0.5B Bloom classifier and the rule
   validators are applied to the gold reference rewrites. A model rate is then
   read against what the human references achieve under the same instrument.
2. Embedding topic preservation (Bloom). Cosine similarity of BAAI/bge-m3 dense
   embeddings (CLS, L2-normalised) between source question and rewrite. The
   threshold is the 5th percentile over TRAIN-split reference rewrites, so it is
   fixed before looking at validation/test outputs. Replaces nothing; the Jaccard
   "semantic_valid" check is kept and reported next to it.
3. Semantic similarity to the reference (QA, summarization): bge-m3 cosine of
   prediction vs reference (BERTScore is not installed offline).
4. QA key-point coverage: share of a row's reference key points whose best
   bge-m3 match among the prediction's sentences reaches a threshold calibrated on
   TRAIN-split reference answers (5th percentile of their own key-point matches).
5. Per-task latency.
6. Blinded human-rating sheets for QA correctness and summary faithfulness
   (stratified, seeded), because ROUGE cannot detect factual errors.

Outputs in the results directory: metrics_extended.json,
evaluation_report_extended.md, human_eval_qa.csv, human_eval_summarization.csv.

Usage:
  python augment_evaluation.py                      # active config, LoRA results
  python augment_evaluation.py --results-dir <dir>  # e.g. the _base results
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for p in (EXPERIMENT_DIR, SCRIPT_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
EMBED_MODEL = "BAAI/bge-m3"
CALIBRATION_PERCENTILE = 5
HUMAN_SAMPLE = {"qa": 60, "summarization": 40}
SEED = 42


# ------------------------------------------------------------------ embedding
class Embedder:
    def __init__(self, model_id: str = EMBED_MODEL, max_length: int = 1024) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.max_length = max_length
        self.cache: dict[str, Any] = {}

    def encode(self, texts: list[str], batch_size: int = 16):
        torch = self.torch
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                cls = self.model(**enc).last_hidden_state[:, 0].float()
            cls = torch.nn.functional.normalize(cls, dim=-1).cpu()
            for t, v in zip(chunk, cls):
                self.cache[t] = v
        return [self.cache[t] for t in texts]

    def cos(self, a: list[str], b: list[str]) -> list[float]:
        ea, eb = self.encode(a), self.encode(b)
        return [float((x * y).sum()) for x, y in zip(ea, eb)]


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text or "") if s.strip()]
    return parts or [text or ""]


def keypoint_scores(emb: Embedder, keypoints: list[str], answer: str) -> list[float]:
    """Best cosine between each key point and any sentence of the answer."""
    sents = sentences(answer)
    ek, es = emb.encode(keypoints), emb.encode(sents)
    return [max(float((k * s).sum()) for s in es) for k in ek]


_SCORER = None


def _scorer():
    global _SCORER
    if _SCORER is None:
        from rouge_score import rouge_scorer
        _SCORER = rouge_scorer.RougeScorer(["rougeL", "rougeLsum"], use_stemmer=True)
    return _SCORER


def rouge_l(pred: str, ref: str) -> float:
    return _scorer().score(ref or "", pred or "")["rougeL"].fmeasure


def strip_markdown(text: str) -> str:
    return re.sub(r"(?m)^\s*(#+|[-*]|\d+\.)\s*", "", text or "").replace("**", "")


def length_controlled(pred: str, ref: str) -> dict[str, float]:
    """Scores after removing Markdown and cutting the output to the reference's word
    count, applied identically to every system, so verbosity cannot change the result."""
    cut = " ".join(strip_markdown(pred).split()[: max(1, len((ref or "").split()))])
    s = _scorer3().score(ref or "", cut)
    return {"rougeL_lc": s["rougeL"].fmeasure, "rouge2_lc": s["rouge2"].fmeasure}


def recall_scores(pred: str, ref: str) -> dict[str, float]:
    s = _scorer3().score(ref or "", pred or "")
    return {"rouge1_recall": s["rouge1"].recall, "rouge2_recall": s["rouge2"].recall}


_SCORER3 = None


def _scorer3():
    global _SCORER3
    if _SCORER3 is None:
        from rouge_score import rouge_scorer
        _SCORER3 = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _SCORER3


def length_block(recs: list[dict[str, Any]], cap: int | None) -> tuple[dict[str, Any], list[dict[str, float]]]:
    rows = [{**length_controlled(r["prediction"], r["reference"]), **recall_scores(r["prediction"], r["reference"])} for r in recs]
    agg = {k: round(st.mean(x[k] for x in rows), 4) for k in rows[0]} if rows else {}
    agg["median_output_words"] = st.median(len(r["prediction"].split()) for r in recs) if recs else None
    agg["median_reference_words"] = st.median(len(r["reference"].split()) for r in recs) if recs else None
    agg["markdown_outputs"] = sum(1 for r in recs if strip_markdown(r["prediction"]) != (r["prediction"] or ""))
    if cap:
        agg["output_token_cap"] = cap
    return agg, rows


def rouge_lsum(pred: str, ref: str) -> float:
    """Summary-level ROUGE-L: sentences must be newline-separated for rougeLsum."""
    split = lambda t: "\n".join(sentences(t))
    return _scorer().score(split(ref), split(pred))["rougeLsum"].fmeasure


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))]


def rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


# ------------------------------------------------------------------- main
def augment(results_dir: Path, config_path: Path, skip_classifier: bool = False) -> dict[str, Any]:
    import dataset_io
    from bloom_validation import validate_bloom_example

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir = REPO_ROOT / cfg["dataset_dir"] if not Path(cfg["dataset_dir"]).is_absolute() else Path(cfg["dataset_dir"])
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    split = metrics.get("dataset", {}).get("split", "validation")
    preds = [json.loads(l) for l in (results_dir / "predictions.jsonl").open(encoding="utf-8") if l.strip()]
    rows = {r["id"]: r for r in dataset_io.load_split_rows(data_dir, split, validate=False)[0]}
    train = dataset_io.load_split_rows(data_dir, "train", validate=False)[0]
    bloom = [r for r in preds if r["task"] == "bloom_rewrite"]
    qa = [r for r in preds if r["task"] == "qa"]
    summ = [r for r in preds if r["task"] == "summarization"]
    out: dict[str, Any] = {"generated_utc": datetime.now(timezone.utc).isoformat(), "split": split,
                           "results_dir": str(results_dir), "embedding_model": EMBED_MODEL,
                           "calibration": f"{CALIBRATION_PERCENTILE}th percentile over train-split references"}
    emb = Embedder()

    # ---- Bloom: reference ceilings and calibrated topic preservation
    if bloom:
        gold_val = [validate_bloom_example(r["source_question"], r["target_bloom_level"], r["reference"]) for r in bloom]
        ceilings = {
            "format_valid": (rate([r["format_valid"] for r in bloom]), rate([g.format_valid for g in gold_val])),
            "semantic_valid_jaccard": (rate([r["semantic_valid"] for r in bloom]), rate([g.semantic_valid for g in gold_val])),
            "cognitive_valid": (rate([r["cognitive_valid"] for r in bloom]), rate([g.cognitive_valid for g in gold_val])),
            "answer_or_declarative_flag": (rate([r.get("answer_output", False) for r in bloom]),
                                           rate(["answer_or_declarative" in (g.rejection_reason or "") for g in gold_val])),
        }
        gold_cls = None
        if not skip_classifier:
            from eval_classifier import load_classifier
            clf, _ = load_classifier(None, repo_root=REPO_ROOT, require_smoke=True)
            gold_cls = [clf(r["reference"]) for r in bloom]
            ceilings["classifier_target_match"] = (rate([bool(r["target_match"]) for r in bloom]),
                                                   rate([g["predicted_level"] == r["target_bloom_level"] for g, r in zip(gold_cls, bloom)]))
        per_level = {}
        for lv in LEVELS:
            idx = [i for i, r in enumerate(bloom) if r["target_bloom_level"] == lv]
            per_level[lv] = {
                "n": len(idx),
                "model_target_accuracy": rate([bool(bloom[i]["target_match"]) for i in idx]),
                "reference_target_accuracy": rate([gold_cls[i]["predicted_level"] == lv for i in idx]) if gold_cls else None,
            }

        train_bloom = [r for r in train if r["task"] == "bloom_rewrite"]
        train_sims = emb.cos([r["source_question"] for r in train_bloom], [r["target_rewrite"] for r in train_bloom])
        topic_thr = percentile(train_sims, CALIBRATION_PERCENTILE)
        model_sims = emb.cos([r["source_question"] for r in bloom], [r["prediction"] for r in bloom])
        gold_sims = emb.cos([r["source_question"] for r in bloom], [r["reference"] for r in bloom])
        topic_model = [s >= topic_thr for s in model_sims]
        topic_gold = [s >= topic_thr for s in gold_sims]
        ceilings["topic_preserved_embedding"] = (rate(topic_model), rate(topic_gold))
        calibrated_full = [
            bool(r["target_match"]) and r["classifier_confidence"] >= 0.5 and t and not r.get("trivial_transform")
            and (r["format_valid"] or r["target_bloom_level"] == "Evaluate" and r.get("answer_output"))
            for r, t in zip(bloom, topic_model)
        ]
        jaccard_fail_embed_pass = sum(1 for r, t in zip(bloom, topic_model) if not r["semantic_valid"] and t)

        # Reference-based scores: agreement with the teacher-written rewrite for
        # the same source and target level. Copying the source cannot game these.
        ref_rl = [rouge_l(r["prediction"], r["reference"]) for r in bloom]
        ref_sim = emb.cos([r["prediction"] for r in bloom], [r["reference"] for r in bloom])
        # Near-copy: rewrite closer (ROUGE-L) to its source than 95% of train
        # reference rewrites are to theirs.
        copy_thr = percentile([rouge_l(r["target_rewrite"], r["source_question"]) for r in train_bloom], 100 - CALIBRATION_PERCENTILE)
        src_rl = [rouge_l(r["prediction"], r["source_question"]) for r in bloom]
        gold_src_rl = [rouge_l(r["reference"], r["source_question"]) for r in bloom]
        near_copy = [x > copy_thr for x in src_rl]
        ceilings["near_copy_of_source"] = (rate(near_copy), rate([x > copy_thr for x in gold_src_rl]))
        per_row_bloom = [
            {"id": r["id"], "calibrated_full": bool(c), "topic_embedding": bool(t), "ref_rougeL": round(a, 4),
             "ref_embedding_sim": round(s, 4), "source_rougeL": round(x, 4), "near_copy": bool(n)}
            for r, c, t, a, s, x, n in zip(bloom, calibrated_full, topic_model, ref_rl, ref_sim, src_rl, near_copy)
        ]
        out["bloom"] = {
            "reference_based": {
                "rougeL_to_reference_mean": round(st.mean(ref_rl), 4),
                "embedding_sim_to_reference_mean": round(st.mean(ref_sim), 4),
                "near_copy_threshold_source_rougeL": round(copy_thr, 4),
            },
            "per_row": per_row_bloom,
            "n": len(bloom),
            "model_vs_reference": {k: {"model": m, "reference": g} for k, (m, g) in ceilings.items()},
            "per_level": per_level,
            "topic_similarity": {
                "threshold": round(topic_thr, 4),
                "train_reference_median": round(st.median(train_sims), 4),
                "model_median": round(st.median(model_sims), 4),
                "reference_median": round(st.median(gold_sims), 4),
                "jaccard_fail_but_embedding_pass": jaccard_fail_embed_pass,
            },
            "fully_validated_original": rate([bool(r["fully_validated"]) for r in bloom]),
            "fully_validated_calibrated": rate(calibrated_full),
            "calibrated_definition": "classifier target match (conf >= 0.5) AND embedding topic preserved AND not trivial AND (format valid OR scenario-first Evaluate item flagged declarative)",
            "declarative_flags_on_evaluate": sum(1 for r in bloom if r.get("answer_output") and r["target_bloom_level"] == "Evaluate"),
        }

    # ---- QA: embedding similarity + key-point coverage
    if qa:
        train_qa = [r for r in train if r["task"] == "qa" and r.get("key_points")]
        calib = [s for r in train_qa[:400] for s in keypoint_scores(emb, list(r["key_points"]), r["reference_answer"])]
        kp_thr = percentile(calib, CALIBRATION_PERCENTILE)
        sims = emb.cos([r["prediction"] for r in qa], [r["reference"] for r in qa])
        coverage, per_row = [], []
        for r, s in zip(qa, sims):
            kps = list(rows.get(r["id"], {}).get("key_points") or [])
            scores = keypoint_scores(emb, kps, r["prediction"]) if kps else []
            cov = sum(x >= kp_thr for x in scores) / len(scores) if scores else None
            if cov is not None:
                coverage.append(cov)
            per_row.append({"id": r["id"], "embedding_sim": round(s, 4), "keypoint_coverage": cov})
        qa_len, qa_len_rows = length_block(qa, None)
        for row, extra in zip(per_row, qa_len_rows):
            row.update({k: round(v, 4) for k, v in extra.items()})
        out["qa"] = {
            "n": len(qa),
            "embedding_similarity_to_reference": {"mean": round(st.mean(sims), 4), "median": round(st.median(sims), 4)},
            "keypoint_coverage": {"mean": round(st.mean(coverage), 4) if coverage else None, "rows_with_key_points": len(coverage),
                                  "threshold": round(kp_thr, 4), "full_coverage_rate": rate([c >= 0.999 for c in coverage]),
                                  "zero_coverage_rate": rate([c == 0 for c in coverage])},
            "length_insensitive": qa_len,
            "per_row": per_row,
        }

    # ---- Summarization: embedding similarity by source type
    if summ:
        sims = emb.cos([r["prediction"] for r in summ], [r["reference"] for r in summ])
        lsum = [rouge_lsum(r["prediction"], r["reference"]) for r in summ]
        by = {}
        for st_name in sorted({r.get("source_type") for r in summ}):
            v = [s for r, s in zip(summ, sims) if r.get("source_type") == st_name]
            by[str(st_name)] = {"n": len(v), "mean": round(st.mean(v), 4)}
        out["summarization"] = {
            "n": len(summ),
            "embedding_similarity_to_reference": {"mean": round(st.mean(sims), 4), "median": round(st.median(sims), 4), "by_source_type": by},
            "rougeLsum": {"mean": round(st.mean(lsum), 4),
                          "note": "summary-level ROUGE-L over newline-separated sentences, as reported for arXiv in the literature"},
            "length_insensitive": length_block(summ, None)[0],
            "per_row": [{"id": r["id"], "embedding_sim": round(s, 4), "rougeLsum": round(x, 4), **{k: round(v, 4) for k, v in e.items()}}
                        for r, s, x, e in zip(summ, sims, lsum, length_block(summ, None)[1])],
            "length_ratio_median": round(st.median(len(r["prediction"].split()) / max(1, len(r["reference"].split())) for r in summ), 3),
            "lowercase_tokenized_style_outputs": sum(1 for r in summ if " , " in r["prediction"] or r["prediction"] == r["prediction"].lower()),
        }

    # ---- Latency per task
    out["latency_by_task"] = {
        t: {"n": len(v), "mean_s": round(st.mean(v), 3), "p50_s": round(st.median(v), 3), "p95_s": round(percentile(v, 95), 3)}
        for t, v in ((t, [r["latency_s"] for r in preds if r["task"] == t]) for t in ("bloom_rewrite", "qa", "summarization")) if v
    }

    # ---- Human-rating sheets (blinded, stratified, seeded)
    rng = random.Random(SEED)
    sheets = {}
    for task, recs, fields in (
        ("qa", qa, ["factually_correct (0/1)", "complete (1-5)", "notes"]),
        ("summarization", summ, ["faithful_no_unsupported_claims (0/1)", "coverage (1-5)", "fluency (1-5)", "notes"]),
    ):
        if not recs:
            continue
        key = "qa_type" if task == "qa" else "source_type"
        strata: dict[str, list] = {}
        for r in recs:
            strata.setdefault(str(r.get(key)), []).append(r)
        k = min(HUMAN_SAMPLE[task], len(recs))
        picked = []
        for name, items in sorted(strata.items()):
            take = max(1, round(k * len(items) / len(recs)))
            picked += rng.sample(items, min(take, len(items)))
        rng.shuffle(picked)
        path = results_dir / f"human_eval_{task}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item", "id", "input", "reference", "system_output", *fields])
            for i, r in enumerate(picked, 1):
                src = r.get("question") if task == "qa" else (r.get("article") or "")[:6000]
                w.writerow([i, r["id"], src, r["reference"], r["prediction"], *[""] * len(fields)])
        sheets[task] = {"path": str(path), "items": len(picked), "stratified_by": key}
    out["human_eval_sheets"] = sheets

    (results_dir / "metrics_extended.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_report(results_dir, metrics, out)
    return out


def length_lines(li: dict[str, Any]) -> list[str]:
    if not li:
        return []
    return [f"- Length check: median output {li['median_output_words']} words vs reference {li['median_reference_words']}; "
            f"{li['markdown_outputs']} outputs use Markdown.",
            f"- Length-controlled (Markdown removed, cut to reference length): ROUGE-L {li['rougeL_lc']}, ROUGE-2 {li['rouge2_lc']}. "
            f"Recall (favours longer outputs): ROUGE-1 {li['rouge1_recall']}, ROUGE-2 {li['rouge2_recall']}."]


def write_report(results_dir: Path, metrics: dict[str, Any], ext: dict[str, Any]) -> None:
    L = [f"# Evaluation report: calibrated additions ({ext['split']})", "",
         f"Generated {ext['generated_utc']}. Original metrics are unchanged in `metrics.json` / `evaluation_report.md`; "
         "everything here is computed identically for the model outputs and for the human reference outputs.", ""]
    b = ext.get("bloom")
    if b:
        L += ["## Bloom rewriting: model vs. reference under the same instruments", "",
              "| Measure | Model | Human reference |", "|---|---:|---:|"]
        names = {"classifier_target_match": "Classifier agrees with requested level", "format_valid": "Format check passed",
                 "semantic_valid_jaccard": "Topic check passed (word overlap, original)",
                 "topic_preserved_embedding": f"Topic preserved (bge-m3, threshold {b['topic_similarity']['threshold']})",
                 "cognitive_valid": "Cognitive check passed", "answer_or_declarative_flag": "Flagged declarative",
                 "near_copy_of_source": f"Near-copy of source (ROUGE-L > {b['reference_based']['near_copy_threshold_source_rougeL']})"}
        for k, v in b["model_vs_reference"].items():
            L.append(f"| {names.get(k, k)} | {v['model']} | {v['reference']} |")
        rb = b["reference_based"]
        L += ["", f"- Agreement with the teacher-written rewrite for the same target level: ROUGE-L {rb['rougeL_to_reference_mean']}, "
              f"bge-m3 similarity {rb['embedding_sim_to_reference_mean']}. These cannot be raised by copying the source; "
              "the word-overlap topic check can, so it is not used to compare systems.",
              f"- Fully validated: original {b['fully_validated_original']} → calibrated {b['fully_validated_calibrated']} "
              f"({b['calibrated_definition']}).",
              f"- {b['topic_similarity']['jaccard_fail_but_embedding_pass']} rewrites fail the word-overlap topic check but pass the embedding check.",
              f"- {b['declarative_flags_on_evaluate']} 'declarative' flags are scenario-first Evaluate items (a stated claim to judge), a valid item style in the dataset.", "",
              "| Target level | N | Model accuracy | Reference accuracy (classifier ceiling) |", "|---|---:|---:|---:|"]
        for lv, v in b["per_level"].items():
            L.append(f"| {lv} | {v['n']} | {v['model_target_accuracy']} | {v['reference_target_accuracy']} |")
        L += ["", "Where the reference accuracy is low (the classifier misreads the human rewrite), a low model accuracy is a measurement limit, not a model failure.", ""]
    q = ext.get("qa")
    if q:
        kc = q["keypoint_coverage"]
        L += ["## Closed-book QA: semantic agreement", "",
              f"- bge-m3 similarity to reference: mean {q['embedding_similarity_to_reference']['mean']}, median {q['embedding_similarity_to_reference']['median']}",
              f"- Key-point coverage (threshold {kc['threshold']}, calibrated on train references): mean {kc['mean']}; "
              f"all key points covered {kc['full_coverage_rate']}; none covered {kc['zero_coverage_rate']} (n={kc['rows_with_key_points']})",
              "- Neither measure detects factual errors stated fluently; see `human_eval_qa.csv`.",
              *length_lines(q.get("length_insensitive", {})), ""]
    s = ext.get("summarization")
    if s:
        L += ["## Summarization: semantic agreement", "",
              f"- bge-m3 similarity to reference: mean {s['embedding_similarity_to_reference']['mean']}, median {s['embedding_similarity_to_reference']['median']}",
              *[f"  - {k}: {v['mean']} (n={v['n']})" for k, v in s["embedding_similarity_to_reference"]["by_source_type"].items()],
              f"- ROUGE-Lsum (summary-level, comparable with arXiv literature): {s['rougeLsum']['mean']}",
              f"- Median length ratio (output/reference words): {s['length_ratio_median']}",
              f"- Outputs in lower-cased tokenized arXiv style: {s['lowercase_tokenized_style_outputs']} of {s['n']} (they mirror the arXiv references).",
              "- Faithfulness is not measured automatically; see `human_eval_summarization.csv`.",
              *length_lines(s.get("length_insensitive", {})), ""]
    L += ["## Latency per task", "", "| Task | N | Mean (s) | P50 (s) | P95 (s) |", "|---|---:|---:|---:|---:|"]
    L += [f"| {t} | {v['n']} | {v['mean_s']} | {v['p50_s']} | {v['p95_s']} |" for t, v in ext["latency_by_task"].items()]
    L += ["", "## Human evaluation sheets", ""]
    L += [f"- `{Path(v['path']).name}`: {v['items']} items, stratified by {v['stratified_by']}, seed {SEED}; model identity not shown."
          for v in ext.get("human_eval_sheets", {}).values()]
    L += ["", "## Still required before claiming results", "",
          "- Run the base model on the same split (`--condition base`) and report paired differences.",
          "- Have at least one subject-matter rater complete the human sheets (two raters for agreement statistics).",
          "- Report the test split once, for the final model only.", ""]
    (results_dir / "evaluation_report_extended.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    from paths import ACTIVE_TRAINING_CONFIG, CONFIG_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG_DIR / ACTIVE_TRAINING_CONFIG))
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--skip-classifier", action="store_true")
    a = ap.parse_args()
    config = Path(a.config) if Path(a.config).is_absolute() else REPO_ROOT / a.config
    cfg = json.loads(config.read_text(encoding="utf-8"))
    rd = Path(a.results_dir or cfg["results_dir"])
    rd = rd if rd.is_absolute() else REPO_ROOT / rd
    augment(rd, config, skip_classifier=a.skip_classifier)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print((rd / "evaluation_report_extended.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
