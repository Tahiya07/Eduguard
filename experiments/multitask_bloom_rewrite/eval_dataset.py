"""Manifest-verified loading for task-separated evaluation datasets.

Each task remains stored in its own directory. This module concatenates only the
selected split in memory for generation; it does not write a merged corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts"
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for search_path in (REPO_ROOT, EXPERIMENT_DIR, SCRIPTS_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from dataset_io import TASKS, load_split_rows, validate_separate_dataset  # noqa: E402
from paths import TASK_SEPARATED_DATA_DIR  # noqa: E402


def load_evaluation_split(
    data_dir: Path | None = None,
    *,
    split: str = "validation",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in ("validation", "test"):
        raise ValueError(f"Evaluation split must be validation or test, got {split!r}")
    root = (data_dir or TASK_SEPARATED_DATA_DIR).resolve()
    rows, meta = load_split_rows(root, split, validate=True)
    counts_by_task = {
        task: sum(row.get("task") == task for row in rows)
        for task in TASKS
    }
    meta.update(
        {
            "split": split,
            "counts": {
                "row_count": len(rows),
                "test_count": len(rows) if split == "test" else None,
                "validation_count": len(rows) if split == "validation" else None,
                "by_task": counts_by_task,
            },
            "dataset_hash": meta.get("manifest_sha256"),
            "file_sha256": meta.get("split_file_sha256"),
            "manifest_path": str(root / "dataset_manifest.json"),
            "test_manifest_path": str(root / "dataset_manifest.json") if split == "test" else None,
            "validation_manifest_path": str(root / "dataset_manifest.json") if split == "validation" else None,
            "full_source_split_unchanged": True,
            "corpus_physically_task_separated": True,
            "combined_in_memory_only": True,
        }
    )
    return rows, meta


def load_test_split(
    data_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compatibility wrapper for code that explicitly requests test data."""
    return load_evaluation_split(data_dir, split="test")
