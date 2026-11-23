#!/usr/bin/env python
"""Audit canonical BloomShift inputs and the active task-separated corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = ROOT / "experiments" / "multitask_bloom_rewrite"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from dataset_io import SPLITS, TASKS, read_jsonl, validate_separate_dataset  # noqa: E402
from paths import BLOOMSHIFT_DIR, REPORTS_DIR, TASK_SEPARATED_DATA_DIR  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-prepared",
        action="store_true",
        help="Audit canonical BloomShift inputs without loading prepared task files.",
    )
    args = parser.parse_args()

    report = {
        "canonical_bloomshift": {"path": str(BLOOMSHIFT_DIR.relative_to(ROOT)), "splits": {}},
        "prepared_task_separated": {},
        "errors": [],
    }
    for split in SPLITS:
        path = BLOOMSHIFT_DIR / f"{split}.json"
        if not path.is_file():
            report["errors"].append(f"Missing canonical BloomShift source: {path}")
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report["errors"].append(f"Cannot read {path}: {exc}")
            continue
        report["canonical_bloomshift"]["splits"][split] = {
            "rows": len(rows),
            "sha256": sha256(path),
            "source_groups": len({str(row.get("group_id", "")) for row in rows}),
        }

    if not args.skip_prepared:
        try:
            integrity = validate_separate_dataset(TASK_SEPARATED_DATA_DIR, ROOT)
            per_split = {}
            for split in SPLITS:
                per_split[split] = {}
                for task in TASKS:
                    path = TASK_SEPARATED_DATA_DIR / task / f"{split}.jsonl"
                    rows = read_jsonl(path)
                    entry = {
                        "rows": len(rows),
                        "sha256": sha256(path),
                        "missing_sft": sum(not row.get("sft_text") for row in rows),
                        "missing_prompt": sum(not row.get("prompt_text") for row in rows),
                    }
                    if task == "bloom_rewrite":
                        entry["target_level_counts"] = dict(
                            Counter(row.get("target_bloom_level") for row in rows)
                        )
                        rewrites = [
                            str(row.get("target_rewrite", "")).strip().casefold()
                            for row in rows
                        ]
                        entry["exact_duplicate_rewrites"] = len(rewrites) - len(set(rewrites))
                    per_split[split][task] = entry
            report["prepared_task_separated"] = {
                "status": "PASS",
                "manifest_sha256": integrity["manifest_sha256"],
                "counts": integrity["counts"],
                "splits": per_split,
                "heldout_rows_used_for_curation": False,
            }
        except (OSError, ValueError, KeyError) as exc:
            report["errors"].append(f"Prepared task-separated corpus failed validation: {exc}")

    report["status"] = "PASS" if not report["errors"] else "FAIL"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "dataset_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
