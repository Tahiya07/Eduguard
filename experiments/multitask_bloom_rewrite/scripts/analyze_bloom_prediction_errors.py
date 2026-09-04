#!/usr/bin/env python
"""Error analysis over EXISTING Bloom predictions (no training, no test leakage).

Reads predictions.jsonl from a completed evaluate_rewrite run and writes
a categorized failure report. Does NOT copy test targets into training data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from bloom_validation import (  # noqa: E402
    BLOOM_LEVELS,
    is_valid_imperative_exam,
    is_valid_interrogative,
)
from eval_metrics import canonical_bloom  # noqa: E402
from paths import TASK_BLOOM  # noqa: E402

ADJACENT = {
    ("Remember", "Understand"),
    ("Understand", "Apply"),
    ("Apply", "Analyze"),
    ("Analyze", "Evaluate"),
    ("Evaluate", "Create"),
}


def classify_error(rec: dict) -> str:
    pred = (rec.get("prediction") or "").strip()
    target = canonical_bloom(rec.get("target_bloom_level"))
    clf = canonical_bloom(rec.get("classifier_prediction"))
    if rec.get("empty_output") or not pred:
        return "L_malformed_or_empty"
    if rec.get("meta_response"):
        return "L_meta_response"
    if rec.get("trivial_transform"):
        return "J_trivial_rewrite"
    if not rec.get("semantic_valid") or rec.get("topic_preserved") is False:
        return "I_topic_drift"
    if rec.get("answer_output") or (
        not is_valid_interrogative(pred) and not is_valid_imperative_exam(pred)
    ):
        return "B_answer_instead_of_question"
    if not rec.get("format_valid"):
        return "K_malformed_output"
    if not rec.get("cognitive_valid"):
        return "C_wrong_cognitive_demand"
    if clf and target and clf != target:
        pair = (target, clf) if (target, clf) in ADJACENT else (clf, target)
        if (target, clf) == ("Remember", "Understand") or (clf, target) == ("Remember", "Understand"):
            return "D_Remember_Understand_confusion"
        if (target, clf) == ("Understand", "Apply") or (clf, target) == ("Understand", "Apply"):
            return "E_Understand_Apply_confusion"
        if (target, clf) == ("Apply", "Analyze") or (clf, target) == ("Apply", "Analyze"):
            return "F_Apply_Analyze_confusion"
        if (target, clf) == ("Analyze", "Evaluate") or (clf, target) == ("Analyze", "Evaluate"):
            return "G_Analyze_Evaluate_confusion"
        if (target, clf) == ("Evaluate", "Create") or (clf, target) == ("Evaluate", "Create"):
            return "H_Evaluate_Create_confusion"
        return "A_wrong_Bloom_level"
    if rec.get("fully_validated"):
        return "OK"
    return "A_wrong_Bloom_level"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Bloom prediction failures")
    parser.add_argument(
        "--predictions",
        required=True,
        help="predictions.jsonl from evaluate_rewrite.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: alongside predictions)",
    )
    args = parser.parse_args()
    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = REPO_ROOT / pred_path
    rows = [
        json.loads(l)
        for l in pred_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    bloom = [r for r in rows if r.get("task") == TASK_BLOOM]
    counts = Counter(classify_error(r) for r in bloom)
    by_target = {lvl: Counter() for lvl in BLOOM_LEVELS}
    for r in bloom:
        lvl = canonical_bloom(r.get("target_bloom_level")) or "UNKNOWN"
        if lvl in by_target:
            by_target[lvl][classify_error(r)] += 1
    report = {
        "n_bloom": len(bloom),
        "failure_category_counts": dict(counts),
        "by_target_level": {k: dict(v) for k, v in by_target.items()},
        "augmentation_guidance": {
            "note": (
                "Generate analogous TRAIN-pool examples for high-count categories. "
                "Never copy frozen test targets into training."
            ),
            "priority_categories": [
                c
                for c, _ in counts.most_common()
                if c != "OK"
            ][:8],
        },
        "source_predictions": str(pred_path),
    }
    out = Path(args.output) if args.output else pred_path.parent / "error_analysis_v3.json"
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    lines = [
        "# Bloom prediction error analysis",
        "",
        f"N={report['n_bloom']}",
        "",
        "## Category counts",
        "",
    ]
    for k, v in counts.most_common():
        lines.append(f"- {k}: {v}")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["failure_category_counts"], indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
