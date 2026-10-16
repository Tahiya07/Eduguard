"""Manifest-verified loading for physically task-separated evaluation datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset_io import TASKS, load_split_rows, validate_separate_dataset
from paths import MULTITASK_FINAL_DATA_DIR


def load_evaluation_split(
    data_dir: Path | None = None,
    *,
    split: str = "validation",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in ("validation", "test"):
        raise ValueError(f"Evaluation split must be validation or test, got {split!r}")
    root = data_dir or MULTITASK_FINAL_DATA_DIR
    root = root.resolve()
    rows, meta = load_split_rows(root, split, validate=True)
    counts_by_task = {
        task: sum(row.get("task") == task for row in rows)
        for task in TASKS
    }
    meta.update({
        "split": split,
        "counts": {
            "row_count": len(rows),
            "test_count": len(rows) if split == "test" else None,
            "validation_count": len(rows) if split == "validation" else None,
            "by_task": counts_by_task,
        },
        "dataset_hash": meta.get("manifest_sha256"),
        "file_sha256": meta.get("split_file_sha256"),
        "test_manifest_path": str(root / "dataset_manifest.json") if split == "test" else None,
        "validation_manifest_path": str(root / "dataset_manifest.json") if split == "validation" else None,
        "manifest_path": str(root / "dataset_manifest.json"),
        "full_source_split_unchanged": True,
        "corpus_physically_task_separated": True,
        "combined_in_memory_only": True,
    })
    return rows, meta


def load_test_split(
    data_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backward-compatible wrapper for callers that explicitly need test data."""
    return load_evaluation_split(data_dir, split="test")
