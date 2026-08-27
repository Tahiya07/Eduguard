#!/usr/bin/env python
"""Validate training configs are identical except model_id/output paths."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
CONFIG_DIR = EXPERIMENT_DIR / "configs"

IGNORE_KEYS = {
    "experiment_name",
    "model_id",
    "model_key",
    "output_dir",
    "results_dir",
    "notes",
}


def main() -> None:
    a = json.loads((CONFIG_DIR / "qwen05b_multitask.json").read_text(encoding="utf-8"))
    b = json.loads((CONFIG_DIR / "qwen15b_multitask.json").read_text(encoding="utf-8"))
    keys = set(a) | set(b)
    diffs = []
    for k in sorted(keys):
        if k in IGNORE_KEYS:
            continue
        if a.get(k) != b.get(k):
            diffs.append({"key": k, "0.5b": a.get(k), "1.5b": b.get(k)})
    decision = json.loads((CONFIG_DIR / "decision_rule.json").read_text(encoding="utf-8"))
    assert decision.get("frozen_before_test_evaluation") is True
    assert decision.get("do_not_change_after_seeing_test_results") is True
    report = {
        "hyperparameter_diffs_outside_allowed": diffs,
        "ok": len(diffs) == 0,
        "decision_rule_loaded": True,
        "model_ids": {"0.5b": a["model_id"], "1.5b": b["model_id"]},
    }
    out = EXPERIMENT_DIR / "reports" / "training_config_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if diffs:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
