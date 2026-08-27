#!/usr/bin/env python
"""Audit the local Figshare Bloom classification corpus.

This dataset is NOT rewrite-pair supervision. Report actual statistics only.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from grouping import normalize_question  # noqa: E402
from paths import FIGSHARE_COMBINED, FIGSHARE_V1, REPORTS_DIR, RESULTS_DIR  # noqa: E402


def audit_csv(path: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(path)
    question_col = "question" if "question" in df.columns else "QUESTION"
    label_col = "bloom_level" if "bloom_level" in df.columns else "BT LEVEL"
    report = {
        "filename": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "exists": True,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "nulls": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "has_rewrite_pairs": False,
        "rewrite_pair_columns": [
            c
            for c in df.columns
            if str(c).lower() in {"target_rewrite", "rewritten_question", "target_question", "rewrite"}
        ],
        "dataset_type": "classification_only",
        "cannot_supervise_target_rewriting_by_itself": True,
    }
    if question_col in df.columns:
        q = df[question_col].astype(str)
        norms = q.map(normalize_question)
        words = q.str.split().str.len()
        incomplete = q.str.contains(r"\.\s*\.\s*\.|_{3,}|\.{3}", case=False, regex=True, na=False)
        report["question_column"] = question_col
        report["exact_duplicate_rows"] = int(q.duplicated().sum())
        report["normalized_duplicate_rows"] = int(norms.duplicated().sum())
        report["unique_questions"] = int(q.nunique())
        report["unique_normalized"] = int(norms.nunique())
        report["word_len_min_median_max"] = [int(words.min()), float(words.median()), int(words.max())]
        report["very_short_lt6_words"] = int((words < 6).sum())
        report["incomplete_like"] = int(incomplete.sum())
    if label_col in df.columns:
        report["label_column"] = label_col
        report["label_counts"] = df[label_col].value_counts(dropna=False).to_dict()
    if "original_label" in df.columns:
        report["original_label_counts"] = df["original_label"].value_counts(dropna=False).to_dict()
    return report


def existing_split_overlap() -> dict:
    import pandas as pd

    splits = {}
    norms = {}
    for name in ("train", "val", "test"):
        path = REPO_ROOT / "data" / f"figshare_bloom_v1_{name}.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        splits[name] = {
            "rows": int(len(df)),
            "labels": df["bloom_level"].value_counts().to_dict() if "bloom_level" in df.columns else {},
        }
        norms[name] = set(df["question"].map(normalize_question))
    overlap = {}
    if {"train", "val", "test"} <= set(norms):
        overlap = {
            "train_val": len(norms["train"] & norms["val"]),
            "train_test": len(norms["train"] & norms["test"]),
            "val_test": len(norms["val"] & norms["test"]),
        }
    return {"classifier_splits": splits, "normalized_overlap": overlap, "do_not_reuse_for_rewrite": True}


def main() -> None:
    payload = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Because no suitable public human-authored Bloom target-level "
            "transformation dataset was identified, a controlled synthetic "
            "transformation dataset was constructed from the Figshare Bloom "
            "classification corpus. The synthetic dataset is used for controlled "
            "model comparison, not as a substitute for human-validated ground truth."
        ),
        "figshare_v1": audit_csv(FIGSHARE_V1) if FIGSHARE_V1.is_file() else {"exists": False, "path": str(FIGSHARE_V1)},
        "figshare_combined": audit_csv(FIGSHARE_COMBINED) if FIGSHARE_COMBINED.is_file() else {"exists": False},
        "existing_classifier_splits": existing_split_overlap(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figshare_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORTS_DIR / "dataset_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    v1 = payload["figshare_v1"]
    print("\nSUMMARY")
    print("file", v1.get("filename"))
    print("rows", v1.get("rows"))
    print("columns", v1.get("columns"))
    print("label_counts", v1.get("label_counts"))
    print("rewrite_pairs", v1.get("has_rewrite_pairs"))
    print("dataset_type", v1.get("dataset_type"))


if __name__ == "__main__":
    main()
