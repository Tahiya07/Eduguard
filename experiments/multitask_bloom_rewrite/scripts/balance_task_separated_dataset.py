#!/usr/bin/env python
"""Balance existing task-separated QA/summarization JSONL files to BloomShift counts.

This edits the existing task-separated dataset in place, first creating a backup
of every file it will change. It does not download datasets or regenerate prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate"
DEFAULT_BLOOM_DIR = REPO_ROOT / "data" / "rewrite dataset" / "bloomshift_final_candidate"
TASKS = ("bloom_rewrite", "qa", "summarization")
SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bloom-dir", type=Path, default=DEFAULT_BLOOM_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    bloom_dir = args.bloom_dir.resolve()
    manifest_path = data_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing task-separated manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("persisted_merged_jsonl") is True:
        raise SystemExit("Expected task-separated data, not a merged corpus.")

    targets: dict[str, int] = {}
    for split in SPLITS:
        bloom_path = bloom_dir / f"{split}.json"
        if not bloom_path.is_file():
            raise SystemExit(f"Missing canonical BloomShift split: {bloom_path}")
        source = json.loads(bloom_path.read_text(encoding="utf-8"))
        if not isinstance(source, list) or not source:
            raise SystemExit(f"Expected a non-empty JSON array: {bloom_path}")
        targets[split] = len(source)

    # Validate every expected input before changing anything.
    rows_by_task_split: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in TASKS:
        for split in SPLITS:
            path = data_dir / task / f"{split}.jsonl"
            if not path.is_file():
                raise SystemExit(f"Missing task split: {path}")
            rows = read_jsonl(path)
            if not rows:
                raise SystemExit(f"Empty task split: {path}")
            wrong = [r for r in rows if r.get("task") != task or r.get("split") != split]
            if wrong:
                raise SystemExit(f"Task/split tags mismatch in {path}")
            if task != "bloom_rewrite" and len(rows) < targets[split]:
                raise SystemExit(
                    f"{task}/{split} has {len(rows)} rows, but needs {targets[split]}; "
                    "refusing to upsample or duplicate examples."
                )
            if task == "bloom_rewrite" and len(rows) != targets[split]:
                raise SystemExit(
                    f"Bloom rewrite {split} count ({len(rows)}) does not match canonical "
                    f"BloomShift count ({targets[split]})."
                )
            rows_by_task_split[(task, split)] = rows

    balanced: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in TASKS:
        for split in SPLITS:
            rows = rows_by_task_split[(task, split)]
            if task == "bloom_rewrite" or len(rows) == targets[split]:
                selected = list(rows)
            else:
                # Distinct deterministic seeds prevent identical selections across tasks/splits.
                offset = TASKS.index(task) * 1000 + SPLITS.index(split) * 100
                selected = random.Random(args.seed + offset).sample(rows, targets[split])
                selected.sort(key=lambda row: str(row.get("id", "")))
            balanced[(task, split)] = selected

    # Back up the six auxiliary files and manifest before the first write.
    backup_dir = data_dir / f"_pre_balance_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    changed_paths = [
        data_dir / task / f"{split}.jsonl"
        for task in ("qa", "summarization")
        for split in SPLITS
    ] + [manifest_path]
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in changed_paths:
        destination = backup_dir / path.relative_to(data_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    try:
        counts = {
            split: {task: len(balanced[(task, split)]) for task in TASKS}
            for split in SPLITS
        }
        generated_hashes = dict(manifest.get("generated_files_sha256", {}))
        for task in TASKS:
            for split in SPLITS:
                path = data_dir / task / f"{split}.jsonl"
                write_jsonl(path, balanced[(task, split)])
                generated_hashes[path.relative_to(data_dir).as_posix()] = sha256(path)

        manifest["counts"] = counts
        manifest["generated_files_sha256"] = generated_hashes
        manifest["storage_layout"] = "one_directory_per_task"
        manifest["persisted_merged_jsonl"] = False
        manifest["balanced_to_bloomshift_counts"] = {
            "enabled": True,
            "seed": args.seed,
            "reference_split_counts": targets,
            "policy": (
                "QA and summarization rows deterministically downsampled without replacement "
                "to match canonical BloomShift counts for each split. Original auxiliary files "
                "and manifest are backed up in the _pre_balance_backup_* directory."
            ),
            "backup_directory": backup_dir.name,
            "balanced_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        for key, task in (("qa_source", "qa"), ("summarization_source", "summarization")):
            source_info = manifest.get(key)
            if isinstance(source_info, dict):
                source_info["prepared_train_rows"] = counts["train"][task]
                source_info["prepared_validation_rows"] = counts["validation"][task]
                source_info["prepared_test_rows"] = counts["test"][task]

        tokenizer_info = manifest.get("tokenizer")
        if isinstance(tokenizer_info, dict):
            checked = tokenizer_info.get("checked_sequences")
            if isinstance(checked, dict):
                for split in SPLITS:
                    if isinstance(checked.get(split), dict):
                        for task in TASKS:
                            if isinstance(checked[split].get(task), dict):
                                checked[split][task]["rows"] = counts[split][task]

        temp_path = manifest_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp_path.replace(manifest_path)
    except Exception:
        # Restore all files changed by this operation if anything fails.
        for path in changed_paths:
            backup = backup_dir / path.relative_to(data_dir)
            if backup.exists():
                shutil.copy2(backup, path)
        raise

    report = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "counts": counts,
        "backup_directory": str(backup_dir),
        "persisted_merged_jsonl": False,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
