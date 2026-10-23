#!/usr/bin/env python
"""Repair known malformed/template-driven Bloom rewrites in the prepared SFT splits.

Run from the repository root:
  python experiments/multitask_bloom_rewrite/scripts/repair_bloom_rewrite_examples.py --dry-run
  python experiments/multitask_bloom_rewrite/scripts/repair_bloom_rewrite_examples.py --apply

Only target_rewrite/opening_verb/transformation_strategy are updated. Split membership,
IDs, source questions, source groups, and all non-Bloom tasks remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "multitask_bloom_rewrite_separate"
BLOOM_DIR = DATA_DIR / "bloom_rewrite"
REPORT_DIR = ROOT / "experiments" / "multitask_bloom_rewrite" / "reports" / "bloom_rewrite_repairs"
SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line: {path}:{line_no}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object: {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def choose_repair(row: dict[str, Any]) -> tuple[str, str] | None:
    source = str(row.get("source_question") or "").strip()
    target = str(row.get("target_rewrite") or "").strip()
    level = str(row.get("target_bloom_level") or "").strip().title()
    s = source.lower()
    t = target.lower()

    # Source-specific, semantically grounded repairs for the known malformed grafts.
    if "what components and constraints in" in t:
        if "mincuts" in s or "minimum cuts" in s:
            return (
                "Analyze how the structure of a family of networks can produce exponentially many distinct minimum cuts, identifying the role of the network components and their relationships.",
                "Analyze",
            )
        if "poster" in s and ("jeans" in s or "food product" in s):
            if "jeans" in s:
                return (
                    "Analyze the poster's intended audience, message, visual elements, and persuasive techniques, and explain how these elements work together to appeal to senior citizens in Malaysia's multiracial population.",
                    "Analyze",
                )
            return (
                "Analyze the poster's visual hierarchy, product message, and persuasive techniques, and explain how these elements work together to attract attention to the new food product.",
                "Analyze",
            )
        return (
            "Analyze the key components of the task described in the original question and explain how their relationships affect the intended outcome.",
            "Analyze",
        )

    if "principle underlying" in t:
        if "story" in s and "title" in s:
            return (
                "Apply your understanding of how a title reflects a story's central themes to explain the title of a different story.",
                "Apply",
            )
        if "nitrogen" in s and ("tillage" in s):
            return (
                "Apply agronomic principles to a no-tillage field and determine how its nitrogen needs may differ from those of a conventionally tilled field.",
                "Apply",
            )
        if "hazard identification" in s and "pathogen" in s:
            return (
                "Apply a hazard-identification framework to a case involving pathogens, and explain which features make the assessment difficult.",
                "Apply",
            )
        return (
            "Apply the relevant concept from the original question to a new, concrete case and explain how it works in practice.",
            "Apply",
        )

    if "construct a model showing the factors responsible for" in t:
        if "base quasineutral region" in s or "minority carrier diffusion length" in s:
            return (
                "Develop a model explaining why the base quasineutral region should be narrow relative to the minority-carrier diffusion length in a bipolar junction transistor.",
                "Develop",
            )
        if "nitrogen" in s and "tillage" in s:
            return (
                "Develop a causal model showing why nitrogen requirements may differ between no-tillage and conventional tillage.",
                "Develop",
            )
        if "hazard identification" in s and "pathogen" in s:
            return (
                "Develop a causal model explaining why identifying hazards associated with pathogens can be difficult.",
                "Develop",
            )
        return (
            "Develop a causal model that explains the main factors and relationships described in the original question.",
            "Develop",
        )

    if (
        level == "Evaluate"
        and "determine whether" in t
        and "satisfies its requirements" in t
        and "using relevant criteria and evidence" in t
    ):
        if "mincuts" in s or "minimum cuts" in s:
            return (
                "Evaluate whether the proposed family of networks demonstrates exponential growth in the number of distinct minimum cuts as network size increases. Justify your judgment using the construction and its properties.",
                "Evaluate",
            )
        if "poster" in s and "food product" in s:
            return (
                "Assess how effectively the poster is likely to advertise the new food product, considering the clarity of its message, visual appeal, and persuasive design choices.",
                "Assess",
            )
        if "vendor-rating form" in s or "universities" in s:
            return (
                "Evaluate whether the vendor-rating form provides a useful basis for comparing the educational offerings of the universities under consideration. Justify your judgment with explicit comparison criteria.",
                "Evaluate",
            )
        return (
            "Assess the claim made in the original question and justify your judgment with evidence relevant to the topic, stating the criteria used.",
            "Assess",
        )

    if "while preserving the stated requirements" in t and not re.search(
        r"\b(requirements|constraints|specifications|criteria)\b", s
    ):
        if "poster" in s and "jeans" in s:
            return (
                "Design a poster promoting jeans to senior citizens in Malaysia's multiracial population, using accessible language and visuals that appeal to this audience.",
                "Design",
            )
        if "poster" in s and "food product" in s:
            return (
                "Design a persuasive poster for the new food product, combining a clear benefit-focused message with engaging visual elements.",
                "Design",
            )
        if "family of networks" in s and ("mincuts" in s or "minimum cuts" in s):
            return (
                "Design a family of networks in which the number of distinct minimum cuts grows exponentially with network size, and explain the construction.",
                "Design",
            )
        return (
            "Develop a solution for the situation described in the original question and explain why the proposed design fits that situation.",
            "Develop",
        )

    # Correct a common malformed source-fragment graft in other generated rewrites.
    if re.search(r"\b(?:is important for|might be required|is complicated)\b", t) and re.search(
        r"\b(?:factors explain why|the factors responsible for|the principle underlying)\b", t
    ):
        if "nitrogen" in s and "tillage" in s:
            return (
                "Analyze the agronomic factors that may cause nitrogen requirements to differ between no-tillage and conventional tillage.",
                "Analyze",
            )
        if "base quasineutral region" in s:
            return (
                "Explain how the base quasineutral-region width relates to minority-carrier diffusion length and affects bipolar-junction-transistor operation.",
                "Explain",
            )
        if "hazard identification" in s:
            return (
                "Analyze the biological and procedural factors that make hazard identification for pathogens difficult.",
                "Analyze",
            )

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up and rewrite matching rows.")
    parser.add_argument("--dry-run", action="store_true", help="Report repairs without modifying data (default).")
    args = parser.parse_args()

    if not (DATA_DIR / "dataset_manifest.json").is_file():
        raise SystemExit(f"Missing task-separated manifest: {DATA_DIR / 'dataset_manifest.json'}")

    original: dict[str, list[dict[str, Any]]] = {}
    updated: dict[str, list[dict[str, Any]]] = {}
    changes: list[dict[str, Any]] = []
    for split in SPLITS:
        path = BLOOM_DIR / f"{split}.jsonl"
        rows = read_jsonl(path)
        original[split] = rows
        fixed_rows = []
        for row in rows:
            new_row = dict(row)
            repair = choose_repair(row)
            if repair:
                new_text, verb = repair
                changes.append({
                    "split": split,
                    "id": row.get("id"),
                    "target_bloom_level": row.get("target_bloom_level"),
                    "source_question": row.get("source_question"),
                    "old_target_rewrite": row.get("target_rewrite"),
                    "new_target_rewrite": new_text,
                    "repair_action_verb": verb,
                })
                new_row["target_rewrite"] = new_text
                new_row["opening_verb"] = verb
                new_row["transformation_strategy"] = "manually_rule_repaired_source_grounded"
            fixed_rows.append(new_row)
        updated[split] = fixed_rows

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "source-grounded repairs of known malformed/template patterns; no split movement",
        "mode": "apply" if args.apply else "dry-run",
        "counts": {
            split: {
                "rows": len(original[split]),
                "repaired": sum(1 for c in changes if c["split"] == split),
            }
            for split in SPLITS
        },
        "changes": changes,
        "warning": (
            "This pass repairs known high-confidence malformed patterns, not every possible semantic error. "
            "Review the change report and manually review held-out examples before using scores as final evidence."
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "repair_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"mode": report["mode"], "counts": report["counts"], "report": str(REPORT_DIR / "repair_report.json")}, indent=2))

    if not args.apply or not changes:
        print("No dataset files changed. Use --apply after reviewing repair_report.json.")
        return

    backup_dir = DATA_DIR.with_name(DATA_DIR.name + "_pre_manual_repairs_backup")
    if backup_dir.exists():
        raise SystemExit(f"Backup already exists; refusing to overwrite: {backup_dir}")
    backup_dir.mkdir(parents=True)
    for split in SPLITS:
        shutil.copy2(BLOOM_DIR / f"{split}.jsonl", backup_dir / f"{split}.jsonl")
    for split in SPLITS:
        write_jsonl(BLOOM_DIR / f"{split}.jsonl", updated[split])

    manifest_path = DATA_DIR / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in SPLITS:
        rel = f"bloom_rewrite/{split}.jsonl"
        manifest["generated_files_sha256"][rel] = sha256(BLOOM_DIR / f"{split}.jsonl")
        manifest["counts"][split]["bloom_rewrite"] = len(updated[split])
    prior_repairs = manifest.get("quality_repairs", {})
    prior_counts = prior_repairs.get("counts", {}) if isinstance(prior_repairs, dict) else {}
    cumulative_counts = {}
    for split in SPLITS:
        previous = prior_counts.get(split, {}) if isinstance(prior_counts, dict) else {}
        previous_repaired = int(previous.get("repaired", previous.get("repaired_rows", 0)) or 0)
        current_repaired = int(report["counts"][split]["repaired"])
        cumulative_counts[split] = {
            "repaired": previous_repaired + current_repaired,
            "this_run": current_repaired,
            "rows": len(updated[split]),
        }
    manifest["quality_repairs"] = {
        "version": "bloom_rewrite_source_grounded_repairs_v1",
        "report": str((REPORT_DIR / "repair_report.json").relative_to(ROOT)).replace("\\", "/"),
        "backup": str(backup_dir.relative_to(ROOT)).replace("\\", "/"),
        "repaired_rows_this_run": len(changes),
        "counts": cumulative_counts,
        "full_semantic_review": "pending",
        "warning": "Targeted rule repairs are not a substitute for human semantic review of every row.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    qc_path = DATA_DIR / "qc_report.json"
    if qc_path.is_file():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        qc["quality_repairs"] = manifest["quality_repairs"]
        qc["integrity"] = qc.get("integrity", {})
        qc["integrity"]["generated_files_sha256"] = dict(manifest["generated_files_sha256"])
        qc["integrity"]["counts"] = {
            split: dict(manifest["counts"][split]) for split in SPLITS
        }
        qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Applied repairs and backed up all three original Bloom split files.")
    print("Run validate_separate_dataset before training.")


if __name__ == "__main__":
    main()
