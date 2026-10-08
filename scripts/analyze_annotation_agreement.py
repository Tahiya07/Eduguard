#!/usr/bin/env python3
"""Validate and analyze BloomShift independent human annotations.

The script never fills reviewer labels. It uses only the two submitted
reviewer worksheets and excludes the separate calibration batch by design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


LABEL_FIELDS = [
    "bloom_aligned",
    "reviewer_bloom_level",
    "content_preserved",
    "context_preserved",
    "meaningful_transformation",
    "pedagogically_valid",
    "clear_and_grammatical",
    "final_decision",
]
KEY_FIELDS = [
    "example_id",
    "split",
    "source_question",
    "source_bloom_level",
    "target_bloom_level",
    "target_rewrite",
]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in KEY_FIELDS + LABEL_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer1", default="reviewer_1.csv")
    parser.add_argument("--reviewer2", default="reviewer_2.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    r1 = load(Path(args.reviewer1))
    r2 = load(Path(args.reviewer2))

    if len(r1) != len(r2):
        raise ValueError(f"Reviewer row-count mismatch: {len(r1)} vs {len(r2)}")
    if r1["example_id"].duplicated().any() or r2["example_id"].duplicated().any():
        raise ValueError("Duplicate example_id detected in reviewer worksheet.")
    if list(r1["example_id"]) != list(r2["example_id"]):
        raise ValueError("Reviewer worksheets are not aligned by example_id.")

    empty_counts = {
        "reviewer_1": int(r1[LABEL_FIELDS].eq("").sum().sum()),
        "reviewer_2": int(r2[LABEL_FIELDS].eq("").sum().sum()),
    }
    if any(empty_counts.values()):
        raise SystemExit(
            "Human annotation is incomplete; no agreement statistics were computed. "
            f"Empty label cells: {empty_counts}"
        )

    results = {
        "annotation_rows": len(r1),
        "calibration_excluded": True,
        "method": "Cohen's kappa",
        "fields": {},
    }
    for field in LABEL_FIELDS:
        results["fields"][field] = {
            "kappa": float(
                cohen_kappa_score(r1[field].tolist(), r2[field].tolist())
            ),
            "n": len(r1),
        }

    payload = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
