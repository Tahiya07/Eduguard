"""Manifest-driven split loading and leakage checks for multitask evaluation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from paths import MULTITASK_FINAL_DATA_DIR, TASK_BLOOM, TASK_QA, TASK_SUMMARIZATION, TASKS

# Legacy fallback retained for callers that test assertions without a manifest.
EXPECTED_TEST_TOTAL = 14329
EXPECTED_TEST_BY_TASK = {
    TASK_BLOOM: 135,
    TASK_QA: 10570,
    TASK_SUMMARIZATION: 3624,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def load_manifest(data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or MULTITASK_FINAL_DATA_DIR
    path = data_dir / "dataset_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dataset manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {path}")
    return manifest


def _leakage_key(row: dict[str, Any]) -> tuple[str, str]:
    task = row["task"]
    if task == TASK_BLOOM:
        key = str(row.get("group_id") or row.get("source_id") or row["source_question"]).lower()
    else:
        key = str(row.get("source_id") or row.get("id") or row.get("example_id"))
    return task, key


def collect_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {_leakage_key(r) for r in rows}


def assert_split_matches(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    expected_split: str,
) -> None:
    if expected_split not in {"validation", "test"}:
        raise ValueError(f"Only validation/test splits can be evaluated, got {expected_split!r}")
    if path.name != f"{expected_split}.jsonl":
        raise AssertionError(
            f"Evaluator requested split={expected_split!r}, but file is {path.name!r}."
        )
    wrong_tags = [
        index for index, row in enumerate(rows, 1)
        if row.get("split") not in (None, expected_split)
    ]
    if wrong_tags:
        raise AssertionError(
            f"{len(wrong_tags)} records are not tagged split={expected_split!r} in {path}; "
            f"first row numbers: {wrong_tags[:10]}"
        )


def assert_split_is_test(rows: list[dict[str, Any]], *, path: Path) -> None:
    """Backward-compatible test-only assertion."""
    assert_split_matches(rows, path=path, expected_split="test")


def assert_test_counts(
    rows: list[dict[str, Any]], manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Backward-compatible alias used by legacy tests and callers."""
    return assert_split_counts(rows, manifest=manifest, split="test")


def assert_split_counts(
    rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any] | None,
    split: str,
) -> dict[str, Any]:
    actual_by_task = Counter(str(row.get("task") or "") for row in rows)
    if manifest is not None:
        reported = manifest.get("counts", {}).get(split)
        if not isinstance(reported, dict):
            raise AssertionError(
                f"Manifest has no per-task counts for split={split!r}; regenerate the corpus."
            )
        expected_by_task: dict[str, int] = {}
        for task in TASKS:
            if task not in reported:
                raise AssertionError(
                    f"Manifest counts for {split!r} omit task {task!r}."
                )
            try:
                expected_by_task[task] = int(reported[task])
            except (TypeError, ValueError) as exc:
                raise AssertionError(
                    f"Invalid count for task {task!r} in manifest split={split!r}."
                ) from exc
            if expected_by_task[task] < 0:
                raise AssertionError(f"Negative manifest count for {split}/{task}.")
        unknown_tasks = sorted(set(actual_by_task) - set(TASKS))
        if unknown_tasks:
            raise AssertionError(f"Unknown task(s) in {split}.jsonl: {unknown_tasks}")
        actual = {task: actual_by_task.get(task, 0) for task in TASKS}
        if actual != expected_by_task:
            raise AssertionError(
                f"{split} task counts differ from dataset manifest: "
                f"expected={expected_by_task}, actual={actual}"
            )
    else:
        if split != "test":
            raise AssertionError("A manifest is required to validate validation-split counts.")
        expected_by_task = dict(EXPECTED_TEST_BY_TASK)
        actual = {task: actual_by_task.get(task, 0) for task in TASKS}
        if len(rows) != EXPECTED_TEST_TOTAL or actual != expected_by_task:
            raise AssertionError(
                f"Legacy test count mismatch: expected total={EXPECTED_TEST_TOTAL}, "
                f"by_task={expected_by_task}; got total={len(rows)}, by_task={actual}"
            )

    total = sum(actual.values()) if manifest is not None else EXPECTED_TEST_TOTAL
    if len(rows) != total:
        raise AssertionError(f"{split} row total expected {total}, got {len(rows)}")
    return {"row_count": len(rows), "by_task": actual}


def assert_no_split_leakage(data_dir: Path) -> dict[str, Any]:
    train = read_jsonl(data_dir / "train.jsonl")
    val = read_jsonl(data_dir / "validation.jsonl")
    test = read_jsonl(data_dir / "test.jsonl")
    train_k = collect_keys(train)
    val_k = collect_keys(val)
    test_k = collect_keys(test)
    overlaps = {
        "train∩test": len(train_k & test_k),
        "validation∩test": len(val_k & test_k),
        "train∩validation": len(train_k & val_k),
    }
    if any(v > 0 for v in overlaps.values()):
        raise AssertionError(f"Leakage detected between splits: {overlaps}")
    bloom_train_groups = {
        str(r.get("group_id") or r.get("source_id") or r["source_question"]).lower()
        for r in train
        if r["task"] == TASK_BLOOM
    }
    bloom_val_groups = {
        str(r.get("group_id") or r.get("source_id") or r["source_question"]).lower()
        for r in val
        if r["task"] == TASK_BLOOM
    }
    bloom_test_groups = {
        str(r.get("group_id") or r.get("source_id") or r["source_question"]).lower()
        for r in test
        if r["task"] == TASK_BLOOM
    }
    bloom_overlaps = {
        "train_groups∩validation": len(bloom_train_groups & bloom_val_groups),
        "train_groups∩test": len(bloom_train_groups & bloom_test_groups),
        "validation_groups∩test": len(bloom_val_groups & bloom_test_groups),
    }
    if any(v > 0 for v in bloom_overlaps.values()):
        raise AssertionError(f"Bloom group leakage: {bloom_overlaps}")
    return {"key_overlaps": overlaps, "bloom_group_overlaps": bloom_overlaps}


def load_evaluation_split(
    data_dir: Path | None = None, *, split: str = "validation"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load an explicitly selected held-out split, checking manifest hashes/counts/leakage."""
    if split not in {"validation", "test"}:
        raise ValueError(f"Evaluation split must be validation or test, got {split!r}")
    data_dir = data_dir or MULTITASK_FINAL_DATA_DIR
    split_path = data_dir / f"{split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split} split: {split_path}")
    manifest = load_manifest(data_dir)
    rows = read_jsonl(split_path)
    assert_split_matches(rows, path=split_path, expected_split=split)
    counts = assert_split_counts(rows, manifest=manifest, split=split)

    expected_hash = manifest.get("generated_files_sha256", {}).get(split_path.name)
    actual_hash = sha256_file(split_path)
    if not expected_hash:
        raise AssertionError(
            f"Manifest has no SHA-256 for {split_path.name}; regenerate the corpus."
        )
    if expected_hash != actual_hash:
        raise AssertionError(
            f"SHA-256 mismatch for {split_path.name}; dataset may be stale or edited."
        )

    leakage = assert_no_split_leakage(data_dir)
    manifest_hash = sha256_file(data_dir / "dataset_manifest.json")
    meta = {
        "split": split,
        "dataset_version": manifest.get("version"),
        "manifest_path": str(data_dir / "dataset_manifest.json"),
        "manifest_sha256": manifest_hash,
        "dataset_hash": actual_hash,
        "split_path": str(split_path),
        "file_sha256": actual_hash,
        "counts": counts,
        "leakage": leakage,
        "bloom_annotation_provenance": manifest.get("bloom_source", {}).get(
            "heldout_annotation_provenance"
        ),
    }
    return rows, meta


def load_test_split(
    data_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backward-compatible wrapper. New experiment runs should use validation by default."""
    return load_evaluation_split(data_dir, split="test")
