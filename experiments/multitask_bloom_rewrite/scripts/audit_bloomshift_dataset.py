#!/usr/bin/env python
"""Audit BloomShift inclusion, source metadata, split integrity, and review evidence."""
from __future__ import annotations

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
REQUIRED_ROW_FIELDS = (
    "source_id", "group_id", "source_question",
    "target_bloom_level", "target_rewrite", "example_id",
)
RATING_FIELDS = (
    "bloom_aligned", "reviewer_bloom_level", "content_preserved",
    "context_preserved", "meaningful_transformation", "pedagogically_valid",
    "clear_and_grammatical", "final_decision",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array in {path}")
    if any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Expected each row to be an object in {path}")
    return value


def source_row_explicitly_marked(row: dict[str, Any]) -> bool:
    """Report the legacy row flags without using them to exclude rows."""
    return row.get("review_status") == "human_validated" and row.get("training_eligible") is True


def audit(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_rows = {split: load_rows(data_dir / f"{split}.json") for split in SPLITS}
    errors: list[str] = []
    warnings: list[str] = []
    metadata_gaps: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "dataset_dir": str(data_dir.resolve()),
        "policy": (
            "Every canonical BloomShift row is included across the fixed train/validation/test workflow "
            "under batch_review_attestation.json. Legacy review flags are provenance fields, not row filters."
        ),
        "legacy_review_flags_are_not_exclusion_criteria": True,
        "splits": {},
        "cross_split_source_group_overlap": {},
        "annotation_provenance": {},
        "errors": errors,
        "warnings": warnings,
    }

    # Verify the explicit dataset-owner inclusion attestation and its scope.
    attestation_path = data_dir / "batch_review_attestation.json"
    attestation: dict[str, Any] = {}
    if not attestation_path.is_file():
        errors.append("missing batch_review_attestation.json; all-row inclusion scope is not declared")
    else:
        try:
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            scope = attestation.get("scope", {})
            declared_counts = scope.get("split_counts", {})
            actual_counts = {split: len(split_rows[split]) for split in SPLITS}
            if not scope.get("include_every_canonical_record"):
                errors.append("batch attestation does not authorize inclusion of every canonical record")
            if declared_counts != actual_counts:
                errors.append(
                    f"batch attestation split counts do not match source files: "
                    f"declared={declared_counts}, actual={actual_counts}"
                )
            if int(scope.get("total_rows", -1)) != sum(actual_counts.values()):
                errors.append("batch attestation total_rows does not match the canonical source files")
            if attestation.get("provenance_limits", {}).get("qualifies_as_independently_annotated_gold") is not False:
                warnings.append("batch attestation should explicitly state that it is not independently annotated gold")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"cannot parse batch_review_attestation.json: {exc}")

    all_seen_ids: dict[str, str] = {}
    group_sets: dict[str, set[str]] = {}
    for split, rows in split_rows.items():
        status_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        groups: set[str] = set()
        duplicate_ids: list[str] = []
        missing_required = 0
        explicitly_marked = 0
        for index, row in enumerate(rows):
            source_status = str(row.get("review_status") or "unmarked")
            status_counts[source_status] += 1
            target = str(row.get("target_bloom_level") or "MISSING")
            target_counts[target] += 1
            if source_row_explicitly_marked(row):
                explicitly_marked += 1

            group = str(row.get("group_id") or row.get("source_group_id") or row.get("source_id") or "").strip()
            if group:
                groups.add(group)
            row_id = str(row.get("example_id") or row.get("id") or "").strip()
            if row_id:
                if row_id in all_seen_ids:
                    duplicate_ids.append(row_id)
                    errors.append(
                        f"duplicate example id {row_id!r} in {split}; first seen in {all_seen_ids[row_id]}"
                    )
                else:
                    all_seen_ids[row_id] = split

            missing_fields = [
                key for key in REQUIRED_ROW_FIELDS
                if not str(row.get(key) or "").strip()
            ]
            if missing_fields:
                missing_required += 1
                errors.append(f"{split}[{index}] missing required fields: {missing_fields}")

            # All rows remain in the pipeline. This queue is only a report of
            # legacy metadata gaps; it is deliberately not an exclusion queue.
            if not source_row_explicitly_marked(row):
                metadata_gaps.append({
                    "split": split,
                    "row_index": index,
                    "example_id": row_id,
                    "source_id": row.get("source_id", ""),
                    "group_id": group,
                    "source_question": row.get("source_question", ""),
                    "source_bloom_level": row.get("source_bloom_level", ""),
                    "target_bloom_level": target,
                    "target_rewrite": row.get("target_rewrite", ""),
                    "source_review_status": source_status,
                    "source_training_eligible": row.get("training_eligible", ""),
                    "included_under_batch_attestation": True,
                    "reason": "legacy source flags are incomplete; row remains included under batch attestation",
                })

        group_sets[split] = groups
        summary["splits"][split] = {
            "total": len(rows),
            "included_for_modeling_workflow": len(rows),
            "excluded_due_to_review_metadata": 0,
            "explicitly_marked_in_legacy_source": explicitly_marked,
            "included_via_batch_attestation_or_legacy_gap": len(rows),
            "legacy_source_review_status_counts": dict(status_counts),
            "target_bloom_counts": dict(target_counts),
            "missing_required_fields": missing_required,
            "duplicate_ids": duplicate_ids,
            "source_group_count": len(groups),
        }

    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlap = sorted(group_sets[left] & group_sets[right])
            summary["cross_split_source_group_overlap"][f"{left}∩{right}"] = len(overlap)
            if overlap:
                errors.append(f"source group overlap between {left} and {right}: {len(overlap)}")

    # Independent reviewer evidence is reported separately from all-row inclusion.
    reviewer_paths = [data_dir / "reviewer_1.csv", data_dir / "reviewer_2.csv"]
    hashes: dict[str, str] = {}
    rows_by_reviewer: list[dict[str, dict[str, str]]] = []
    for path in reviewer_paths:
        if not path.is_file():
            rows_by_reviewer.append({})
            continue
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows_by_reviewer.append({
                str(row.get("example_id") or "").strip(): row
                for row in csv.DictReader(handle)
                if str(row.get("example_id") or "").strip()
            })

    independent_files = (
        len(hashes) == 2 and hashes.get("reviewer_1.csv") != hashes.get("reviewer_2.csv")
    )
    heldout_rows = split_rows["validation"] + split_rows["test"]
    incomplete = []
    for row in heldout_rows:
        row_id = str(row.get("example_id") or row.get("id") or "").strip()
        for reviewer_index, reviewer_rows in enumerate(rows_by_reviewer, start=1):
            rating = reviewer_rows.get(row_id)
            if not rating or any(not str(rating.get(field) or "").strip() for field in RATING_FIELDS):
                incomplete.append(f"{row_id or '<missing-id>'}:reviewer_{reviewer_index}")
    independent_verified = independent_files and not incomplete
    if not independent_files:
        warnings.append("Independent reviewer files are missing or byte-identical; inter-rater reliability is not established.")
    if incomplete:
        warnings.append(
            f"{len(incomplete)} held-out reviewer records lack complete rubric ratings; "
            "do not present the held-out targets as an independently rated gold benchmark."
        )
    summary["annotation_provenance"] = {
        "inclusion_basis": "dataset_owner_batch_attestation",
        "owner_reported_all_rows_human_reviewed": bool(
            attestation.get("dataset_owner_report", {}).get("all_records_reported_human_reviewed")
        ),
        "all_source_rows_included": True,
        "reviewer_file_sha256": hashes,
        "reviewer_files_byte_identical": (
            hashes.get("reviewer_1.csv") == hashes.get("reviewer_2.csv")
            if len(hashes) == 2 else None
        ),
        "independent_reviewer_files": independent_files,
        "heldout_rows": len(heldout_rows),
        "incomplete_reviewer_records": len(incomplete),
        "incomplete_examples": incomplete[:20],
        "required_rating_fields": list(RATING_FIELDS),
        "independent_review_verified": independent_verified,
        "inter_rater_reliability_established": False,
    }

    for split, values in summary["splits"].items():
        if values["total"] == 0:
            errors.append(f"{split} split is empty")
        invalid_levels = sorted(set(values["target_bloom_counts"]) - set(LEVELS) - {"MISSING"})
        if invalid_levels:
            errors.append(f"{split} has unrecognized target Bloom labels: {invalid_levels}")

    total_rows = sum(len(rows) for rows in split_rows.values())
    included_rows = sum(summary["splits"][s]["included_for_modeling_workflow"] for s in SPLITS)
    summary["overall"] = {
        "total": total_rows,
        "included_for_modeling_workflow": included_rows,
        "excluded_due_to_review_metadata": 0,
        "legacy_source_rows_with_incomplete_review_metadata": len(metadata_gaps),
        "ready_for_training_and_provisional_evaluation": (
            not errors and included_rows == total_rows and total_rows == int(attestation.get("scope", {}).get("total_rows", -1))
        ),
        "ready_for_independent_gold_evaluation": not errors and independent_verified,
    }
    return summary, metadata_gaps


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

    report, metadata_gaps = audit(data_dir)
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    gaps_path = output_dir / "legacy_source_metadata_gaps.csv"
    fields = [
        "split", "row_index", "example_id", "source_id", "group_id",
        "source_question", "source_bloom_level", "target_bloom_level",
        "target_rewrite", "source_review_status", "source_training_eligible",
        "included_under_batch_attestation", "reason",
    ]
    with gaps_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata_gaps)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Legacy source metadata report (does not exclude rows): {gaps_path}")
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
