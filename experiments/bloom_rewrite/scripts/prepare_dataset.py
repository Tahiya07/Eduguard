#!/usr/bin/env python
"""Canonical dataset entry point: validate existing data or build without silent overwrite."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bloom_target_policy import POLICY_VERSION, TOPIC_OVERLAP_THRESHOLD  # noqa: E402
from grouping import normalize_question  # noqa: E402
from paths import (  # noqa: E402
    DATASET_VERSION,
    HUMAN_EVAL_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    REWRITE_ARCHIVE_DIR,
    REWRITE_DATA_DIR,
    SEED,
)
from template_memorization import analyze_examples  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def leakage_check(rows: list[dict]) -> dict:
    split_norms: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    split_groups: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
    split_sources: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for row in rows:
        split = row["split"]
        split_norms[split].add(normalize_question(row["source_question"]))
        split_groups[split].add(int(row["group_id"]))
        split_sources[split].add(row["source_id"])
    return {
        "normalized_question_overlap": {
            "train_validation": len(split_norms["train"] & split_norms["validation"]),
            "train_test": len(split_norms["train"] & split_norms["test"]),
            "validation_test": len(split_norms["validation"] & split_norms["test"]),
        },
        "group_overlap": {
            "train_validation": len(split_groups["train"] & split_groups["validation"]),
            "train_test": len(split_groups["train"] & split_groups["test"]),
            "validation_test": len(split_groups["validation"] & split_groups["test"]),
        },
        "source_id_overlap": {
            "train_validation": len(split_sources["train"] & split_sources["validation"]),
            "train_test": len(split_sources["train"] & split_sources["test"]),
            "validation_test": len(split_sources["validation"] & split_sources["test"]),
        },
    }


def write_human_eval(rows: list[dict], seed: int) -> Path:
    import random

    rng = random.Random(seed)
    test = [ex for ex in rows if ex["split"] == "test"]
    wanted = [
        "Remember->Understand",
        "Understand->Apply",
        "Apply->Analyze",
        "Analyze->Evaluate",
        "Evaluate->Create",
        "Remember->Create",
        "Remember->Analyze",
        "Understand->Evaluate",
        "Apply->Create",
        "Create->Remember",
        "Create->Apply",
        "Evaluate->Understand",
    ]
    sampled = []
    for trans in wanted:
        pool = [ex for ex in test if ex["transformation_type"] == trans]
        rng.shuffle(pool)
        for ex in pool[:3]:
            sampled.append(
                {
                    "item_id": f"he_{len(sampled)+1:03d}",
                    "source_question": ex["source_question"],
                    "target_bloom_level": ex["target_bloom_level"],
                    "system_output": "",
                    "hidden_system_id": "",
                    "rater_target_bloom_alignment_1to5": "",
                    "rater_topic_preservation_1to5": "",
                    "rater_cognitive_demand_1to5": "",
                    "rater_question_quality_1to5": "",
                    "rater_grammatical_quality_1to5": "",
                    "rater_non_triviality_1to5": "",
                    "notes": "",
                    "example_id": ex["example_id"],
                    "transformation_type": ex["transformation_type"],
                    "source_bloom_level_withheld": ex["source_bloom_level"],
                }
            )
    HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = HUMAN_EVAL_DIR / "blinded_eval_items.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sampled[0].keys()) if sampled else ["item_id"])
        writer.writeheader()
        writer.writerows(sampled)
    (HUMAN_EVAL_DIR / "README.md").write_text(
        "Human scoring remains pending. Raters see ONLY original question, "
        "target Bloom level, and generated rewrite (1–5 scales). "
        "Do NOT show source Bloom level or model identity.\n",
        encoding="utf-8",
    )
    return path


def validate_existing(data_dir: Path) -> dict:
    required = ["train.jsonl", "validation.jsonl", "test.jsonl", "dataset_statistics.json", "dataset_manifest.json"]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Incomplete dataset at {data_dir}; missing {missing}")
    rows = []
    counts = {}
    for split in ("train", "validation", "test"):
        part = read_jsonl(data_dir / f"{split}.jsonl")
        counts[split] = len(part)
        rows.extend(part)
    leak = leakage_check(rows)
    # Assert production-aligned 2-input prompts (no source Bloom in SFT text).
    leaked = 0
    for row in rows:
        text = row.get("text") or ""
        user = ""
        for msg in row.get("messages") or []:
            if msg.get("role") == "user":
                user = msg.get("content") or ""
        blob = f"{text}\n{user}"
        if "Original Bloom level:" in blob or "Source Bloom level:" in blob:
            leaked += 1
    if leaked:
        raise SystemExit(
            f"VALIDATION FAILED: {leaked} records still contain source Bloom level in generator text."
        )
    stats = json.loads((data_dir / "dataset_statistics.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    template = analyze_examples(rows)
    coverage = stats.get("transformation_coverage") or {}
    missing_cells = [k for k, v in coverage.items() if not v]
    report = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": manifest.get("dataset_version"),
        "dataset_hash": manifest.get("dataset_hash"),
        "source_dataset": "data/figshare_bloom_v1.csv",
        "source_sha256": (manifest.get("source_files") or [{}])[0].get("sha256"),
        "generated_dataset_sha256": manifest.get("dataset_hash"),
        "random_seed": manifest.get("seed", SEED),
        "split_sizes": counts,
        "class_distribution_source": stats.get("source_bloom_distribution"),
        "class_distribution_target": stats.get("target_bloom_distribution"),
        "source_target_matrix": coverage,
        "missing_transformations": missing_cells,
        "filtering_counts": stats.get("filter_stats"),
        "rejection_reasons": (stats.get("synthesis") or {}).get("rejection_categories"),
        "transformation_policy_version": POLICY_VERSION,
        "similarity_threshold": TOPIC_OVERLAP_THRESHOLD,
        "leakage_methodology": "normalize, near-duplicate, question-family grouping, split by group",
        "leakage_check": leak,
        "leakage_pass": all(v == 0 for v in leak["normalized_question_overlap"].values())
        and all(v == 0 for v in leak["group_overlap"].values()),
        "generator_task": "question + target_level → rewrite",
        "source_bloom_in_generator_prompt": False,
        "source_bloom_level_leak_count": leaked,
        "template_memorization": template,
        "synthetic_or_original": "synthetic",
        "human_ground_truth": False,
        "interpretation": (
            "Because no suitable public human-authored Bloom target-level "
            "transformation dataset was identified, a controlled synthetic "
            "transformation dataset was constructed from the Figshare Bloom "
            "classification corpus. The synthetic dataset is used for controlled "
            "model comparison, not as a substitute for human-validated ground truth. "
            "BEFORE: question + source Bloom + target Bloom → rewrite. "
            "AFTER: question + target Bloom → rewrite; source Bloom is metadata only."
        ),
    }
    write_human_eval(rows, int(manifest.get("seed") or SEED))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORTS_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def archive_existing(data_dir: Path) -> Path | None:
    manifest_path = data_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("dataset_version") or DATASET_VERSION
    dest = REWRITE_ARCHIVE_DIR / version
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "dataset_statistics.json", "dataset_manifest.json"):
        src = data_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    print("Archived existing dataset to", dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or validate the Bloom rewrite dataset")
    parser.add_argument("--validate-only", action="store_true", help="Check existing files; never overwrite")
    parser.add_argument("--new-version", action="store_true", help="Archive current files, then rebuild")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild in place (discouraged)")
    args = parser.parse_args()

    existing = (REWRITE_DATA_DIR / "dataset_manifest.json").is_file()
    if args.validate_only or (existing and not args.new_version and not args.overwrite):
        if not existing:
            raise SystemExit("No dataset present. Re-run with --new-version to build.")
        report = validate_existing(REWRITE_DATA_DIR)
        print(json.dumps({k: report[k] for k in (
            "dataset_version", "dataset_hash", "split_sizes", "leakage_check",
            "leakage_pass", "missing_transformations", "human_ground_truth",
        )}, indent=2))
        print("Refusing silent overwrite. Dataset validated in place.")
        if not report["leakage_pass"]:
            raise SystemExit("LEAKAGE CHECK FAILED")
        return

    if existing and args.new_version:
        archive_existing(REWRITE_DATA_DIR)
    elif existing and args.overwrite:
        print("WARNING: --overwrite rebuilds data/bloom_rewrite in place.")

    from prepare_bloom_rewrite_dataset import main as build_main

    old_argv = sys.argv
    sys.argv = [old_argv[0], "--overwrite"]
    try:
        build_main()
    finally:
        sys.argv = old_argv
    report = validate_existing(REWRITE_DATA_DIR)
    print("leakage_pass", report["leakage_pass"])


if __name__ == "__main__":
    main()
