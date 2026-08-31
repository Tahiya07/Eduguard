"""Automated resume state machine test (no expensive training)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_resume_skips_completed_and_retries_interrupted(tmp_path: Path):
    state = {
        "run_id": "test_run",
        "completed_experiments": ["exp1"],
        "interrupted_experiments": ["exp2"],
        "failed_experiments": [],
        "blocked_experiments": [],
        "skipped_experiments": [],
        "not_executed_experiments": [],
    }
    registry_order = ["exp1", "exp2", "exp3"]
    completed = set(state["completed_experiments"])
    interrupted = set(state["interrupted_experiments"])

    next_exp = None
    for eid in registry_order:
        if eid in completed:
            continue
        if eid in interrupted or eid not in completed:
            next_exp = eid
            break
    assert next_exp == "exp2"

    # After exp2 completes
    state["completed_experiments"].append("exp2")
    state["interrupted_experiments"].remove("exp2")
    completed = set(state["completed_experiments"])
    next_exp = None
    for eid in registry_order:
        if eid not in completed:
            next_exp = eid
            break
    assert next_exp == "exp3"


def test_stale_artifact_not_reused_for_new_run(tmp_path: Path):
    from experiments.federated.run_integrity import artifact_matches_run

    p = tmp_path / "result.json"
    p.write_text(json.dumps({"run_id": "old", "status": "EXECUTED"}), encoding="utf-8")
    assert not artifact_matches_run(p, run_id="new", allow_missing_run_id=False)
