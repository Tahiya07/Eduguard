"""Dataset loading and held-out test assertions for multitask evaluation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from paths import MULTITASK_FINAL_DATA_DIR, TASK_BLOOM, TASK_QA, TASK_SUMMARIZATION

EXPECTED_TASKS = {TASK_BLOOM, TASK_QA, TASK_SUMMARIZATION}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or MULTITASK_FINAL_DATA_DIR
    path = data_dir / "dataset_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _leakage_key(row: dict[str, Any]) -> tuple[str, str]:
    task = row["task"]
    if task == TASK_BLOOM:
        key = str(row.get("group_id") or row.get("source_id") or row["source_question"]).lower()
    else:
        key = str(row.get("source_id") or row.get("id") or row.get("example_id"))
    return task, key


def collect_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {_leakage_key(r) for r in rows}


def assert_split_is_test(rows: list[dict[str, Any]], *, path: Path) -> None:
    if path.name != "test.jsonl":
        raise AssertionError(
            f"Evaluator must use held-out TEST split; got file {path.name}. "
            "Do not evaluate on validation.jsonl."
        )
    non_test = [r for r in rows if r.get("split") != "test"]
    if non_test:
        raise AssertionError(
            f"{len(non_test)} records are missing split=test tag in {path}"
        )


def assert_test_counts(rows: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if manifest is None:
        raise AssertionError("Dataset manifest is required to validate test counts.")

    by_task = Counter(r.get("task") for r in rows)
    unknown = set(by_task) - EXPECTED_TASKS
    if unknown:
        raise AssertionError(f"Unexpected test tasks: {sorted(map(str, unknown))}")

    reported = manifest.get("counts", {}).get("test", {})
    if not isinstance(reported, dict) or not reported:
        raise AssertionError("Manifest is missing counts.test.")

    expected = {task: int(count) for task, count in reported.items()}
    if set(expected) != EXPECTED_TASKS:
        raise AssertionError(
            f"Manifest test tasks do not match expected task set: {sorted(expected)}"
        )

    expected_total = sum(expected.values())
    if expected_total != len(rows):
        raise AssertionError(
            f"Manifest test task counts sum to {expected_total}, "
            f"but test.jsonl contains {len(rows)} rows."
        )

    for task in sorted(EXPECTED_TASKS):
        got = by_task.get(task, 0)
        if got != expected[task]:
            raise AssertionError(
                f"test task {task}: manifest expects {expected[task]}, got {got}"
            )

    return {"test_count": len(rows), "by_task": dict(by_task)}


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
        "train_groups∩test": len(bloom_train_groups & bloom_test_groups),
        "val_groups∩test": len(bloom_val_groups & bloom_test_groups),
    }
    if any(v > 0 for v in bloom_overlaps.values()):
        raise AssertionError(f"Bloom group leakage: {bloom_overlaps}")
    return {"key_overlaps": overlaps, "bloom_group_overlaps": bloom_overlaps}


def load_test_split(data_dir: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_dir = data_dir or MULTITASK_FINAL_DATA_DIR
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}")
    manifest = load_manifest(data_dir)
    rows = read_jsonl(test_path)
    assert_split_is_test(rows, path=test_path)
    counts = assert_test_counts(rows, manifest)
    leakage = assert_no_split_leakage(data_dir)
    meta = {
        "test_manifest_path": str(test_path),
        "dataset_hash": manifest.get("corpus_hash"),
        "file_sha256": sha256_file(test_path),
        "counts": counts,
        "leakage": leakage,
    }
    return rows, meta
