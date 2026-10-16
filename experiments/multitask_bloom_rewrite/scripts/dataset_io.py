"""I/O helpers for task-separated multitask datasets.

Persisted dataset files stay physically separated by task. A single training/evaluation
split may be combined in memory only after each task file and its manifest fingerprint
has been verified.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

TASKS = ("bloom_rewrite", "qa", "summarization")
SPLITS = ("train", "validation", "test")
MANIFEST_VERSION = "multitask_bloom_rewrite_separate_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line in {path} at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def load_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "dataset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset_manifest.json at {path}. Run "
            "prepare_multitask_dataset_final.py first."
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read dataset manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {path}")
    return manifest


def validate_separate_dataset(
    data_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless every task/split file matches the manifest fingerprints."""
    data_dir = data_dir.resolve()
    repo_root = (repo_root or REPO_ROOT).resolve()
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = load_manifest(data_dir)
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported dataset manifest version {manifest.get('version')!r}; "
            f"expected {MANIFEST_VERSION!r}. Rebuild the task-separated datasets."
        )
    if manifest.get("storage_layout") != "one_directory_per_task":
        raise ValueError("Manifest does not declare the task-separated storage layout.")

    declared_counts = manifest.get("counts")
    hashes = manifest.get("generated_files_sha256")
    if not isinstance(declared_counts, dict) or not isinstance(hashes, dict):
        raise ValueError("Manifest must contain per-split counts and generated file hashes.")

    observed_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        if not isinstance(declared_counts.get(split), dict):
            raise ValueError(f"Manifest has no task counts for {split!r}.")
        observed_counts[split] = {}
        for task in TASKS:
            rel = f"{task}/{split}.jsonl"
            path = data_dir / task / f"{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing task-separated dataset file: {path}")
            expected_hash = hashes.get(rel)
            if not expected_hash:
                raise ValueError(f"Manifest has no SHA-256 for {rel}.")
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"SHA-256 mismatch for {rel}; data may be stale or edited. "
                    "Rebuild the task-separated datasets."
                )
            rows = read_jsonl(path)
            actual_count = len(rows)
            try:
                expected_count = int(declared_counts[split][task])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid manifest count for {split}/{task}.") from exc
            if actual_count != expected_count:
                raise ValueError(
                    f"Count mismatch for {split}/{task}: manifest={expected_count}, "
                    f"file={actual_count}."
                )
            seen_ids: set[str] = set()
            for row_number, row in enumerate(rows, 1):
                if row.get("task") != task:
                    raise ValueError(
                        f"{rel}:{row_number} has task={row.get('task')!r}, expected {task!r}."
                    )
                if row.get("split") != split:
                    raise ValueError(
                        f"{rel}:{row_number} has split={row.get('split')!r}, expected {split!r}."
                    )
                rid = str(row.get("id") or "").strip()
                if not rid:
                    raise ValueError(f"{rel}:{row_number} has an empty id.")
                if rid in seen_ids:
                    raise ValueError(f"{rel}:{row_number} duplicates id {rid!r} within {task}.")
                seen_ids.add(rid)
            observed_counts[split][task] = actual_count

    # Verify BloomShift source and its all-row inclusion attestation have not
    # changed since preprocessing. These are source files, not merged corpora.
    bloom_source = manifest.get("bloom_source", {})
    source_rel = bloom_source.get("path")
    source_hashes = bloom_source.get("files_sha256", {})
    if not source_rel or not isinstance(source_hashes, dict):
        raise ValueError("Manifest is missing canonical BloomShift source fingerprints.")
    source_dir = (repo_root / source_rel).resolve()
    required_source_files = (
        "train.json", "validation.json", "test.json", "batch_review_attestation.json"
    )
    for name in required_source_files:
        path = source_dir / name
        expected = source_hashes.get(name)
        if not path.is_file():
            raise FileNotFoundError(f"Canonical BloomShift source file is missing: {path}")
        if not expected or sha256_file(path) != expected:
            raise ValueError(
                f"Canonical BloomShift source fingerprint mismatch for {path}. "
                "Rebuild the task-separated datasets."
            )

    manifest_hash = sha256_file(manifest_path)
    return {
        "layout": "task-separated",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_version": manifest["version"],
        "data_dir": str(data_dir),
        "counts": observed_counts,
        "generated_files_sha256": hashes,
        "bloom_source": str(source_dir),
        "bloom_source_sha256": source_hashes,
        "bloom_annotation_provenance": bloom_source.get("heldout_annotation_provenance"),
    }


def load_task_split(
    data_dir: Path,
    task: str,
    split: str,
    *,
    validate: bool = True,
) -> list[dict[str, Any]]:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}.")
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
    if validate:
        validate_separate_dataset(data_dir)
    return read_jsonl(data_dir / task / f"{split}.jsonl")


def load_split_rows(
    data_dir: Path,
    split: str,
    *,
    tasks: tuple[str, ...] | list[str] | None = None,
    validate: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read task-specific split files and combine rows in memory, never on disk."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
    if validate:
        meta = validate_separate_dataset(data_dir)
    else:
        manifest_path = data_dir / "dataset_manifest.json"
        manifest = load_manifest(data_dir)
        meta = {
            "layout": "task-separated",
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_version": manifest.get("version"),
            "counts": manifest.get("counts", {}),
            "bloom_annotation_provenance": manifest.get("bloom_source", {}).get(
                "heldout_annotation_provenance"
            ),
        }
    selected_tasks = tuple(tasks) if tasks is not None else TASKS
    unknown = set(selected_tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"Unknown task(s): {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for task in selected_tasks:
        rel = f"{task}/{split}.jsonl"
        path = data_dir / task / f"{split}.jsonl"
        task_rows = read_jsonl(path)
        rows.extend(task_rows)
        counts[task] = len(task_rows)
        hashes[rel] = sha256_file(path)
    rows.sort(key=lambda row: (str(row.get("task", "")), str(row.get("id", ""))))
    meta.update(
        {
            "split": split,
            "split_counts_loaded": counts,
            "split_file_sha256": hashes,
            "loaded_tasks": list(selected_tasks),
            "combined_in_memory_only": True,
        }
    )
    return rows, meta


def filter_rows_to_tasks(
    rows: list[dict[str, Any]], tasks: list[str] | tuple[str, ...] | None
) -> list[dict[str, Any]]:
    if tasks is None:
        return list(rows)
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"Unknown task(s): {sorted(unknown)}")
    selected = set(tasks)
    return [row for row in rows if row.get("task") in selected]
