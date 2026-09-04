"""Metric helpers for multitask evaluation (no fabricated scores)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from bloom_validation import BLOOM_LEVELS
from paths import TOPIC_SIMILARITY_THRESHOLD

# Frozen before test evaluation (operational gate for fully_validated_rewrite_rate).
CLASSIFIER_CONFIDENCE_MIN = 0.5

SQUAD_ARTICLES = re.compile(r"[^\w\s]|_")


def normalize_qa(text: str) -> str:
    return " ".join(SQUAD_ARTICLES.sub(" ", (text or "").lower()).split())


def qa_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_qa(prediction) == normalize_qa(reference))


def qa_token_f1(prediction: str, reference: str) -> float:
    pred_toks = normalize_qa(prediction).split()
    ref_toks = normalize_qa(reference).split()
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def aggregate_qa_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "exact_match": None, "f1": None}
    ems = [qa_exact_match(p, r) for p, r in pairs]
    f1s = [qa_token_f1(p, r) for p, r in pairs]
    return {
        "n": len(pairs),
        "exact_match": round(sum(ems) / len(ems), 6),
        "f1": round(sum(f1s) / len(f1s), 6),
    }


def rouge_scores(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "rouge1": None, "rouge2": None, "rougeL": None}
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        return {"n": len(pairs), "error": f"rouge_score unavailable: {exc}"}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in pairs:
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "n": len(pairs),
        "rouge1": round(sum(r1) / len(r1), 6),
        "rouge2": round(sum(r2) / len(r2), 6),
        "rougeL": round(sum(rl) / len(rl), 6),
    }


def optional_bertscore(pairs: list[tuple[str, str]]) -> dict[str, Any] | None:
    if not pairs:
        return None
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return {"available": False, "reason": "bert_score not installed"}
    preds, refs = zip(*pairs)
    p, r, f1 = bert_score_fn(
        list(preds), list(refs), lang="en", verbose=False, device="cpu"
    )
    return {
        "available": True,
        "precision": round(float(p.mean()), 6),
        "recall": round(float(r.mean()), 6),
        "f1": round(float(f1.mean()), 6),
    }


def canonical_bloom(level: str | None) -> str | None:
    if not level:
        return None
    s = level.strip().title()
    return s if s in BLOOM_LEVELS else None


def bloom_classification_metrics(
    y_true: list[str], y_pred: list[str]
) -> dict[str, Any]:
    labels = list(BLOOM_LEVELS)
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for gold, pred in zip(y_true, y_pred):
        g = canonical_bloom(gold) or gold
        p = canonical_bloom(pred) or pred or "MISSING"
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    per: dict[str, dict[str, float | int]] = {}
    f1s, supports = [], []
    for label in labels:
        prec = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        rec = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        support = sum(1 for item in y_true if canonical_bloom(item) == label)
        per[label] = {
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "support": support,
        }
        f1s.append(f1)
        supports.append(support)
    macro_p = sum(per[l]["precision"] for l in labels) / len(labels)
    macro_r = sum(per[l]["recall"] for l in labels) / len(labels)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    total = sum(supports) or 1
    weighted_f1 = sum(f1s[i] * supports[i] for i in range(len(labels))) / total
    correct = sum(1 for g, p in zip(y_true, y_pred) if canonical_bloom(g) == canonical_bloom(p))
    n = len(y_true) or 1
    confusion = {src: {tgt: 0 for tgt in labels + ["MISSING"]} for src in labels}
    for g, p in zip(y_true, y_pred):
        gs = canonical_bloom(g) or g
        ps = canonical_bloom(p) or p or "MISSING"
        if gs in confusion and ps in confusion[gs]:
            confusion[gs][ps] += 1
    return {
        "n": len(y_true),
        "accuracy": round(correct / n, 6),
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_level": per,
        "confusion_matrix": confusion,
    }


def source_target_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    """6×6 non-self source→target analysis using metadata source_bloom_level."""
    level_codes = {
        "Remember": "C1",
        "Understand": "C2",
        "Apply": "C3",
        "Analyze": "C4",
        "Evaluate": "C5",
        "Create": "C6",
    }
    cells: dict[str, dict[str, Any]] = {}
    for src in BLOOM_LEVELS:
        for tgt in BLOOM_LEVELS:
            if src == tgt:
                continue
            key = f"{level_codes[src]}→{level_codes[tgt]}"
            subset = [
                r
                for r in records
                if canonical_bloom(r.get("source_bloom_level")) == src
                and canonical_bloom(r.get("target_bloom_level")) == tgt
            ]
            n = len(subset)
            acc = (
                sum(1 for r in subset if r.get("target_match")) / n if n else None
            )
            fvr = (
                sum(1 for r in subset if r.get("fully_validated")) / n if n else None
            )
            cells[key] = {
                "source_level": src,
                "target_level": tgt,
                "n": n,
                "target_accuracy": round(acc, 6) if acc is not None else None,
                "fully_validated_rate": round(fvr, 6) if fvr is not None else None,
            }
    return {"cells": cells, "topic_similarity_threshold": TOPIC_SIMILARITY_THRESHOLD}


def bloom_rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records) or 1
    return {
        "format_valid_rate": round(sum(1 for r in records if r.get("format_valid")) / n, 6),
        "semantic_valid_rate": round(sum(1 for r in records if r.get("semantic_valid")) / n, 6),
        "cognitive_valid_rate": round(sum(1 for r in records if r.get("cognitive_valid")) / n, 6),
        "classifier_match_rate": round(sum(1 for r in records if r.get("target_match")) / n, 6),
        "fully_validated_rewrite_rate": round(
            sum(1 for r in records if r.get("fully_validated")) / n, 6
        ),
        "trivial_transform_rate": round(
            sum(1 for r in records if r.get("trivial_transform")) / n, 6
        ),
        "meta_response_rate": round(sum(1 for r in records if r.get("meta_response")) / n, 6),
        "answer_output_rate": round(sum(1 for r in records if r.get("answer_output")) / n, 6),
        "empty_output_rate": round(sum(1 for r in records if r.get("empty_output")) / n, 6),
        "interrogative_valid_rate": round(
            sum(1 for r in records if r.get("interrogative_valid")) / n, 6
        ),
        "imperative_valid_rate": round(
            sum(1 for r in records if r.get("imperative_valid")) / n, 6
        ),
        "question_form_valid_rate": round(
            sum(1 for r in records if r.get("question_form_valid")) / n, 6
        ),
        "mean_output_chars": round(
            sum(len(str(r.get("prediction") or "")) for r in records) / n, 2
        ),
        "mean_semantic_similarity": round(
            sum(r.get("semantic_similarity", 0.0) for r in records) / n, 6
        ),
        "classifier_confidence_min": CLASSIFIER_CONFIDENCE_MIN,
    }


def failure_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for r in records:
        if r.get("task") != "bloom_rewrite":
            continue
        if r.get("fully_validated"):
            continue
        reason = r.get("failure_category") or "OTHER"
        counts[reason] += 1
    return {"failure_counts": dict(counts), "n_failures": sum(counts.values())}


def latency_summary(latencies: list[float]) -> dict[str, float | None]:
    if not latencies:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None}
    s = sorted(latencies)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p / 100.0 * (n - 1)))))
        return s[idx]

    return {
        "mean": round(sum(s) / n, 6),
        "p50": round(pct(50), 6),
        "p95": round(pct(95), 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
    }
