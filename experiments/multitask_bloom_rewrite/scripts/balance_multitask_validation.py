#!/usr/bin/env python
"""Create a deterministic task-balanced validation split for multitask SFT.

The complete generated validation split is preserved as validation_full.jsonl.
The active validation.jsonl contains every Bloom rewrite row and an equal
number of QA and summarization rows sampled without replacement.

This is row balancing for checkpoint selection, not a claim that token-level
loss contributions are exactly equal across tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TASKS = ("bloom_rewrite", "qa", "summarization")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}: blank line at {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(rows: list[dict]) -> dict[str, int]:
    observed = Counter(str(row.get("task", "")) for row in rows)
    return {task: observed.get(task, 0) for task in TASKS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/multitask_bloom_rewrite_final")
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    active_path = data_dir / "validation.jsonl"
    backup_path = data_dir / "validation_full.jsonl"
    manifest_path = data_dir / "balanced_validation_manifest.json"

    if not active_path.exists() and not backup_path.exists():
        raise SystemExit(f"Neither {active_path} nor {backup_path} exists.")
    if not backup_path.exists():
        if not active_path.exists():
            raise SystemExit(f"Missing source validation file: {active_path}")
        # Preserve the exact original bytes; never overwrite an existing backup.
        backup_path.write_bytes(active_path.read_bytes())

    source_rows = read_jsonl(backup_path)
    observed_tasks = set(row.get("task") for row in source_rows)
    if observed_tasks != set(TASKS):
        raise SystemExit(
            f"Expected exactly tasks {TASKS}; found {sorted(map(str, observed_tasks))}"
        )
    for row in source_rows:
        if row.get("split") != "validation":
            raise SystemExit(
                f"Row {row.get('id')!r} has split={row.get('split')!r}, expected 'validation'."
            )

    by_task = {
        task: [row for row in source_rows if row.get("task") == task]
        for task in TASKS
    }
    target_count = len(by_task["bloom_rewrite"])
    if target_count == 0:
        raise SystemExit("No Bloom rewrite validation examples found.")
    too_small = {
        task: len(by_task[task])
        for task in ("qa", "summarization")
        if len(by_task[task]) < target_count
    }
    if too_small:
        raise SystemExit(
            "Cannot retain every Bloom row and balance other tasks without replacement; "
            f"insufficient rows: {too_small}"
        )

    rng = random.Random(args.seed)
    selected = list(by_task["bloom_rewrite"])
    for task in ("qa", "summarization"):
        candidates = list(by_task[task])
        selected.extend(rng.sample(candidates, target_count))
    rng.shuffle(selected)

    if len({str(row.get("id", "")) for row in selected}) != len(selected):
        raise SystemExit("Selected validation examples contain duplicate IDs.")

    write_jsonl(active_path, selected)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "retain_all_bloom_rewrite; deterministic_sample_without_replacement_for_qa_and_summarization",
        "seed": args.seed,
        "source_validation_file": backup_path.name,
        "source_rows": len(source_rows),
        "source_task_counts": counts(source_rows),
        "balanced_validation_file": active_path.name,
        "balanced_rows": len(selected),
        "balanced_task_counts": counts(selected),
        "sampling_unit": "examples/rows",
        "bloom_rewrite_rows_all_retained": len(by_task["bloom_rewrite"]),
        "qa_sampled_without_replacement": target_count,
        "summarization_sampled_without_replacement": target_count,
        "source_sha256": sha256(backup_path),
        "balanced_sha256": sha256(active_path),
        "limitations": [
            "Row counts are balanced, but sequence lengths differ by task; standard eval_loss is not guaranteed to weight task losses equally.",
            "The balanced validation split is for checkpoint selection only. Keep test.jsonl untouched and report test metrics separately by task.",
            "This procedure does not validate Bloom rewrite semantic preservation or cognitive-level correctness."
        ]
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
