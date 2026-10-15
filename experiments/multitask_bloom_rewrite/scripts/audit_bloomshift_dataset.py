#!/usr/bin/env python
"""Audit BloomShift review eligibility and split integrity without changing labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import BLOOMSHIFT_DIR  # noqa: E402

SPLITS = ("train", "validation", "test")
LEVELS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array in {path}")
    if any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Expected each row to be an object in {path}")
    return value


def eligible(row: dict[str, Any]) -> bool:
    return row.get("review_status") == "human_validated" and row.get("training_eligible") is True


def audit(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_rows = {split: load_rows(data_dir / f"{split}.json") for split in SPLITS}
    errors: list[str] = []
    summary: dict[str, Any] = {
        "dataset_dir": str(data_dir.resolve()),
        "policy": "Only review_status=human_validated AND training_eligible=true is eligible.",
        "missing_review_status_is_not_validation": True,
        "splits": {},
        "cross_split_source_group_overlap": {},
        "annotation_provenance": {},
        "errors": errors,
    }
    group_sets: dict[str, set[str]] = {}
    review_queue: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}

    for split, rows in split_rows.items():
        target_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        groups: set[str] = set()
        eligible_count = 0
        duplicate_ids: list[str] = []
        missing_required = 0

        for index, row in enumerate(rows):
            status = str(row.get("review_status") or "unmarked")
            status_counts[status] += 1
            target = str(row.get("target_bloom_level") or "MISSING")
            target_counts[target] += 1
            group = str(row.get("group_id") or row.get("source_group_id") or row.get("source_id") or "")
            if group:
                groups.add(group)
            row_id = str(row.get("example_id") or row.get("id") or "")
            if row_id:
                if row_id in seen_ids:
                    duplicate_ids.append(row_id)
                    errors.append(f"duplicate example id {row_id!r} in {split}; first seen in {seen_ids[row_id]}")
                else:
                    seen_ids[row_id] = split
            if not all(str(row.get(k) or "").strip() for k in ("source_question", "target_rewrite")):
                missing_required += 1
                errors.append(f"{split}[{index}] has empty source_question or target_rewrite")
            if eligible(row):
                eligible_count += 1
            else:
                review_queue.append({
                    "split": split,
                    "row_index": index,
                    "example_id": row_id,
                    "source_id": row.get("source_id", ""),
                    "group_id": group,
                    "source_question": row.get("source_question", ""),
                    "source_bloom_level": row.get("source_bloom_level", ""),
                    "target_bloom_level": target,
                    "target_rewrite": row.get("target_rewrite", ""),
                    "review_status": status,
                    "training_eligible": row.get("training_eligible", ""),
                    "reason": "missing human validation or training_eligible is not true",
                })

        group_sets[split] = groups
        summary["splits"][split] = {
            "total": len(rows),
            "eligible": eligible_count,
            "excluded_not_explicitly_validated": len(rows) - eligible_count,
            "review_status_counts": dict(status_counts),
            "target_bloom_counts": dict(target_counts),
            "missing_source_or_rewrite": missing_required,
            "duplicate_ids": duplicate_ids,
            "source_group_count": len(groups),
        }

    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlap = sorted(group_sets[left] & group_sets[right])
            summary["cross_split_source_group_overlap"][f"{left}∩{right}"] = len(overlap)
            if overlap:
                errors.append(f"source group overlap between {left} and {right}: {len(overlap)}")

    # Verify that held-out "human_validated" status has independent, populated evidence.
    reviewer_paths = [data_dir / "reviewer_1.csv", data_dir / "reviewer_2.csv"]
    if not all(path.is_file() for path in reviewer_paths):
        errors.append("missing reviewer_1.csv or reviewer_2.csv; held-out annotation provenance is not auditable")
        summary["annotation_provenance"] = {"available": False}
    else:
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in reviewer_paths]
        reviewer_rows = []
        for path in reviewer_paths:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reviewer_rows.append({
                    str(row.get("example_id") or "").strip(): row
                    for row in csv.DictReader(handle)
                    if str(row.get("example_id") or "").strip()
                })
        rating_fields = (
            "bloom_aligned", "reviewer_bloom_level", "content_preserved",
            "context_preserved", "meaningful_transformation", "pedagogically_valid",
            "clear_and_grammatical", "final_decision",
        )
        heldout_eligible = [
            row for split in ("validation", "test") for row in split_rows[split]
            if eligible(row)
        ]
        incomplete: list[str] = []
        for row in heldout_eligible:
            row_id = str(row.get("example_id") or row.get("id") or "").strip()
            for reviewer_idx, by_id in enumerate(reviewer_rows, start=1):
                rating = by_id.get(row_id)
                if not rating or any(not str(rating.get(field) or "").strip() for field in rating_fields):
                    incomplete.append(f"{row_id or '<missing-id>'}:reviewer_{reviewer_idx}")
        summary["annotation_provenance"] = {
            "reviewer_file_sha256": {
                "reviewer_1.csv": hashes[0],
                "reviewer_2.csv": hashes[1],
            },
            "reviewer_files_byte_identical": hashes[0] == hashes[1],
            "heldout_eligible_rows": len(heldout_eligible),
            "incomplete_reviewer_records": len(incomplete),
            "incomplete_examples": incomplete[:20],
            "required_rating_fields": list(rating_fields),
        }
        if hashes[0] == hashes[1]:
            errors.append("reviewer_1.csv and reviewer_2.csv are byte-identical; independent ratings are not demonstrated")
        if incomplete:
            errors.append(
                f"{len(incomplete)} held-out reviewer records lack complete ratings; "
                "do not treat those rows as human-validated gold labels"
            )

    for split, values in summary["splits"].items():
        if values["total"] == 0:
            errors.append(f"{split} split is empty")
        invalid_levels = sorted(set(values["target_bloom_counts"]) - set(LEVELS) - {"MISSING"})
        if invalid_levels:
            errors.append(f"{split} has unrecognized target Bloom labels: {invalid_levels}")

    summary["overall"] = {
        "total": sum(len(rows) for rows in split_rows.values()),
        "eligible": sum(summary["splits"][s]["eligible"] for s in SPLITS),
        "excluded_not_explicitly_validated": len(review_queue),
        "ready_for_training_and_evaluation": (
            not errors and all(summary["splits"][s]["eligible"] > 0 for s in SPLITS)
        ),
    }
    return summary, review_queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(BLOOMSHIFT_DIR))
    parser.add_argument("--output-dir", default="experiments/multitask_bloom_rewrite/reports/bloomshift_audit")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report, queue = audit(data_dir)
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    queue_path = output_dir / "human_review_queue.csv"
    fields = [
        "split", "row_index", "example_id", "source_id", "group_id",
        "source_question", "source_bloom_level", "target_bloom_level",
        "target_rewrite", "review_status", "training_eligible", "reason",
    ]
    with queue_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(queue)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Human-review queue: {queue_path}")
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
