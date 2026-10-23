#!/usr/bin/env python
"""Conservatively curate malformed Bloom-rewrite SFT rows without touching QA/sum data.

Default is a dry run. Pass --apply to back up the current dataset, quarantine flagged
Bloom rows, rewrite only the Bloom train/validation/test files, and refresh fingerprints.

This script deliberately does NOT synthesize replacement answers. High-confidence bad
rows are excluded and preserved for manual correction; inventing replacements with
another unreviewed template would recreate the original problem.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate"
TASK_DIR = DATA_DIR / "bloom_rewrite"
BACKUP_DIR = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate_pre_quality_curation_backup"
REPORT_DIR = REPO_ROOT / "experiments" / "multitask_bloom_rewrite" / "reports" / "bloom_quality_curation"
SPLITS = ("train", "validation", "test")
LEVELS = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}

# High-confidence defects: broken source-clause grafts, grammar artefacts from generic
# templates, and Evaluate templates that assert criteria/requirements the source never gave.
BAD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("principle_underlying_graft", re.compile(r"\bthe principle underlying\b", re.I)),
    ("factors_responsible_graft", re.compile(r"\bthe factors responsible for\b", re.I)),
    ("source_instruction_grafted_into_noun_phrase", re.compile(r"\b(?:a family of networks|the story|hazard identification)\s+to demonstrate that\b", re.I)),
    ("components_constraints_graft", re.compile(r"\bwhat components and constraints in\b", re.I)),
    ("broken_explain_why_clause", re.compile(r"\b(?:factors explain why|associated with)\s+[^?.!]{0,180}\b(?:might be required|is important for|has the title that it does)\b", re.I)),
    ("create_model_graft", re.compile(r"\bconstruct a model showing the factors responsible for\b", re.I)),
]

UNSUPPORTED_EVALUATION = re.compile(
    r"\b(?:meets|satisfies)\s+(?:its|the|stated)\s+(?:stated\s+)?requirements\b"
    r"|\bstated requirements and constraints\b"
    r"|\busing relevant criteria and evidence\b"
    r"|\bbased on the given specifications\b",
    re.I,
)
GENERIC_EVALUATION = re.compile(
    r"^(?:what evidence or criteria would you use to assess|determine whether)\b",
    re.I,
)
GENERIC_APPLY = re.compile(
    r"^(?:use|apply) the principle (?:underlying|illustrated by)\b",
    re.I,
)
GENERIC_CREATE = re.compile(
    r"^(?:construct a model showing the factors responsible for|propose a decision framework that)\b",
    re.I,
)
ACTION_PREFIX = re.compile(
    r"^(?:identify|list|state|name|define|recall|recognize|specify|explain|describe|interpret|"
    r"summarize|classify|illustrate|apply|use|calculate|compute|solve|determine|demonstrate|"
    r"implement|test|predict|examine|compare|trace|differentiate|investigate|contrast|analyze|"
    r"assess|evaluate|appraise|critique|justify|validate|rank|recommend|judge|design|develop|"
    r"formulate|construct|propose|devise|compose|plan|generate|create)\b[\s,:-]*",
    re.I,
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row in {path}:{line_no}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def find_rejection_reasons(row: dict[str, Any]) -> list[str]:
    source = str(row.get("source_question") or "").strip()
    rewrite = str(row.get("target_rewrite") or "").strip()
    level = str(row.get("target_bloom_level") or "").strip().title()
    reasons: list[str] = []

    if level not in LEVELS:
        reasons.append("invalid_target_bloom_level")
    if not source or not rewrite:
        reasons.append("missing_source_or_target")
        return reasons

    for name, pattern in BAD_PATTERNS:
        if pattern.search(rewrite):
            reasons.append(name)

    # Do not permit a generic compliance/evidence wrapper when the original question
    # does not actually state criteria, requirements, constraints, specifications, or evidence.
    source_has_criteria = bool(re.search(
        r"\b(criteria|criterion|requirement|constraint|specification|evidence|standard|rubric|"
        r"benchmark|trade[- ]?off|validity|accuracy|effectiveness)\b", source, re.I
    ))
    if level == "Evaluate" and UNSUPPORTED_EVALUATION.search(rewrite) and not source_has_criteria:
        reasons.append("unsupported_generic_evaluation_rubric")
    if level == "Evaluate" and GENERIC_EVALUATION.search(rewrite) and not source_has_criteria:
        reasons.append("stock_evaluate_template_without_source_criteria")
    if level == "Apply" and GENERIC_APPLY.search(rewrite):
        reasons.append("stock_apply_template")
    if level == "Create" and GENERIC_CREATE.search(rewrite):
        reasons.append("stock_create_template")

    # If the target is a different level but merely copies the source after its opening
    # command, it has probably changed the verb rather than the required cognitive work.
    source_level = str(row.get("source_bloom_level") or "").strip().title()
    if source_level and source_level != level:
        src_tail = norm(ACTION_PREFIX.sub("", source))
        tgt_tail = norm(ACTION_PREFIX.sub("", rewrite))
        if len(src_tail.split()) >= 6 and len(tgt_tail.split()) >= 6:
            similarity = SequenceMatcher(None, src_tail, tgt_tail).ratio()
            if similarity >= 0.94:
                reasons.append("near_verbatim_source_with_only_action_prefix_change")

    return list(dict.fromkeys(reasons))


def action_start(text: str) -> str:
    match = re.match(r"^\s*([A-Za-z]+)", text or "")
    return match.group(1).lower() if match else "(no verb)"


def build_report(rows_by_split: dict[str, list[dict[str, Any]]],
                 rejects_by_split: dict[str, list[tuple[dict[str, Any], list[str]]]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "conservative_high_confidence_exclusion_no_synthetic_replacements",
        "splits": {},
        "limitations": [
            "This pass removes high-confidence malformed or stock-template rows; it does not claim to replace human semantic review.",
            "Rejected rows are preserved with reasons for manual correction and possible future inclusion.",
            "The held-out test split is filtered using the same predeclared, text-only rules; it is never moved into training.",
            "Action-verb counts are descriptive diagnostics, not a target to force by synonym substitution."
        ]
    }
    for split, rows in rows_by_split.items():
        rejected = rejects_by_split[split]
        before_by_level = Counter(str(r.get("target_bloom_level", "UNKNOWN")) for r in rows)
        kept = [r for r in rows if not any(r is bad_row for bad_row, _ in rejected)]
        after_by_level = Counter(str(r.get("target_bloom_level", "UNKNOWN")) for r in kept)
        starts = defaultdict(Counter)
        for row in kept:
            starts[str(row.get("target_bloom_level", "UNKNOWN"))][action_start(str(row.get("target_rewrite") or ""))] += 1
        reason_counts = Counter(reason for _, reasons in rejected for reason in reasons)
        report["splits"][split] = {
            "before": len(rows),
            "kept": len(kept),
            "rejected": len(rejected),
            "before_by_target_level": dict(before_by_level),
            "kept_by_target_level": dict(after_by_level),
            "rejection_reasons": dict(reason_counts),
            "action_verb_distribution_after_filtering": {
                level: dict(counter.most_common())
                for level, counter in sorted(starts.items())
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--backup-dir", type=Path, default=BACKUP_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--apply", action="store_true", help="Back up originals and apply the curation.")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    task_dir = data_dir / "bloom_rewrite"
    manifest_path = data_dir / "dataset_manifest.json"
    qc_path = data_dir / "qc_report.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing dataset manifest: {manifest_path}")

    rows_by_split = {
        split: read_jsonl(task_dir / f"{split}.jsonl")
        for split in SPLITS
    }
    rejects_by_split: dict[str, list[tuple[dict[str, Any], list[str]]]] = {}
    kept_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, rows in rows_by_split.items():
        rejects = []
        kept = []
        for row in rows:
            reasons = find_rejection_reasons(row)
            if reasons:
                rejects.append((row, reasons))
            else:
                kept.append(row)
        rejects_by_split[split] = rejects
        kept_by_split[split] = kept

    report = build_report(rows_by_split, rejects_by_split)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "quality_curation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for split, rejected in rejects_by_split.items():
        rejected_path = args.report_dir / f"{split}_rejected.jsonl"
        write_jsonl(rejected_path, [
            {**row, "quality_rejection_reasons": reasons}
            for row, reasons in rejected
        ])

    print(json.dumps({
        "mode": "APPLY" if args.apply else "DRY RUN",
        "data_dir": str(data_dir),
        "report": str(args.report_dir / "quality_curation_report.json"),
        "splits": report["splits"],
    }, ensure_ascii=False, indent=2))

    if not args.apply:
        print("\nDry run only. Inspect the report and rejected JSONL files, then rerun with --apply.")
        return

    if not any(rejects_by_split.values()):
        print("No rows matched the high-confidence rejection rules; dataset left unchanged.")
        return
    if args.backup_dir.exists():
        raise SystemExit(
            f"Backup directory already exists: {args.backup_dir}. "
            "Move it aside before applying again; the script will not overwrite a backup."
        )

    args.backup_dir.mkdir(parents=True)
    for split in SPLITS:
        shutil.copy2(task_dir / f"{split}.jsonl", args.backup_dir / f"{split}.jsonl")
    shutil.copy2(manifest_path, args.backup_dir / "dataset_manifest.json")
    if qc_path.is_file():
        shutil.copy2(qc_path, args.backup_dir / "qc_report.json")

    for split in SPLITS:
        write_jsonl(task_dir / f"{split}.jsonl", kept_by_split[split])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = manifest.setdefault("counts", {})
    generated_hashes = manifest.setdefault("generated_files_sha256", {})
    for split in SPLITS:
        counts.setdefault(split, {})["bloom_rewrite"] = len(kept_by_split[split])
        rel = f"bloom_rewrite/{split}.jsonl"
        generated_hashes[rel] = sha256_file(task_dir / f"{split}.jsonl")
    for split in SPLITS:
        manifest[f"{split}_rows"] = sum(int(v) for v in counts.get(split, {}).values())
    manifest["quality_curation"] = {
        "version": "bloom_rewrite_quality_curation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "conservative_high_confidence_exclusion_no_synthetic_replacements",
        "backup_dir": str(args.backup_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "report_path": str((args.report_dir / "quality_curation_report.json").relative_to(REPO_ROOT)).replace("\\", "/"),
        "counts": {
            split: {
                "before": len(rows_by_split[split]),
                "kept": len(kept_by_split[split]),
                "rejected": len(rejects_by_split[split]),
            }
            for split in SPLITS
        },
        "training_note": "Rejected examples are quarantined, not relabelled. Manually corrected examples must be reviewed before re-adding."
    }
    policy_note = (
        "Bloom rewrite rows are conservatively quality-filtered after split assignment; "
        "the original train/validation/test boundaries and source-group isolation are preserved. "
        "Rejected rows are quarantined and are not used for SFT."
    )
    manifest["training_split_eligibility_policy"] = policy_note
    bloom_source = manifest.setdefault("bloom_source", {})
    bloom_source["training_split_eligibility_policy"] = policy_note
    training_policy = manifest.setdefault("training_data_policy", {})
    retained = training_policy.setdefault("retained_training_rows", {})
    retained["bloom_rewrite"] = len(kept_by_split["train"])
    fractions_total = sum(int(v) for v in retained.values())
    training_policy["task_train_fractions_of_retained_union"] = {
        task: round(int(count) / fractions_total, 6) if fractions_total else 0.0
        for task, count in retained.items()
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Mark the previous broad QC report as historical because its tokenizer percentiles
    # were calculated before filtering. The training loader relies on manifest hashes/counts.
    if qc_path.is_file():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        qc["status"] = "PASS_WITH_BLOOM_QUALITY_CURATION"
        qc["historical_token_length_stats_pre_curation"] = qc.pop("token_length_stats", None)
        qc["historical_tokenizer_report_pre_curation"] = qc.pop("tokenizer", None)
        qc["tokenizer_report_status"] = "historical_pre_curation; rerun tokenizer audit for current percentiles"
        qc["quality_curation"] = report
        qc["integrity"] = qc.get("integrity", {})
        qc["integrity"]["manifest_sha256"] = sha256_file(manifest_path)
        qc["integrity"]["counts"] = {
            split: dict(counts.get(split, {})) for split in SPLITS
        }
        qc["integrity"]["generated_files_sha256"] = dict(generated_hashes)
        qc["counts"] = {split: dict(counts.get(split, {})) for split in SPLITS}
        qc["task_counts"] = {split: dict(counts.get(split, {})) for split in SPLITS}
        qc["rows"] = sum(sum(int(v) for v in counts.get(split, {}).values()) for split in SPLITS)
        qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nApplied. Original split files and manifests are backed up.")
    print("Run the dataset validator before training:")
    print("  python experiments/multitask_bloom_rewrite/scripts/audit_datasets.py")


if __name__ == "__main__":
    main()
