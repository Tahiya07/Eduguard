#!/usr/bin/env python
"""Validate the active task-separated 1.5B training configuration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CONFIG_DIR = EXPERIMENT_DIR / "configs"
DATA_DIR = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate"
ACTIVE_CONFIG = CONFIG_DIR / "qwen15b_multitask_separate_20261016.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_io import validate_separate_dataset  # noqa: E402


def main() -> None:
    errors: list[str] = []
    config: dict = {}
    manifest: dict = {}
    try:
        config = json.loads(ACTIVE_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot load active training config: {type(exc).__name__}: {exc}")
    try:
        manifest = json.loads((DATA_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
        integrity = validate_separate_dataset(DATA_DIR, REPO_ROOT)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        integrity = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"task-separated dataset validation failed: {exc}")

    if config:
        if config.get("model_id") != "Qwen/Qwen2.5-1.5B-Instruct":
            errors.append("active config must use Qwen/Qwen2.5-1.5B-Instruct")
        if config.get("dataset_dir", "").replace("\\", "/") != "data/multitask_bloom_rewrite_separate":
            errors.append("active config does not point to the task-separated dataset")
        if config.get("train_tasks") != manifest.get("tasks"):
            errors.append("active config train_tasks differ from manifest tasks")
        if config.get("max_seq_length") != manifest.get("tokenizer", {}).get("max_seq_length"):
            errors.append("active config max_seq_length differs from manifest tokenizer limit")
        for split in ("validation", "test"):
            key = f"{split}_task_limits"
            if config.get(key) != manifest.get("counts", {}).get(split):
                errors.append(f"active config {key} differs from manifest counts")

    report = {
        "ok": not errors,
        "active_config": str(ACTIVE_CONFIG.relative_to(REPO_ROOT)).replace("\\", "/"),
        "model_id": config.get("model_id"),
        "dataset_dir": config.get("dataset_dir"),
        "dataset_integrity": integrity,
        "errors": errors,
    }
    out = EXPERIMENT_DIR / "reports" / "training_config_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
