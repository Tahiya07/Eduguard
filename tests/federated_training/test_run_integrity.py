"""Run integrity and stale-result prevention tests."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.federated.run_integrity import (
    artifact_matches_run,
    config_hash,
    result_is_placeholder,
)


def test_placeholder_not_valid_completion(tmp_path: Path):
    p = tmp_path / "out.json"
    p.write_text(json.dumps({"status": "NOT_EXECUTED"}), encoding="utf-8")
    assert result_is_placeholder(json.loads(p.read_text()))
    assert not artifact_matches_run(p, run_id="run_a", allow_missing_run_id=False)


def test_stale_run_id_rejected(tmp_path: Path):
    p = tmp_path / "out.json"
    p.write_text(
        json.dumps({"run_id": "old_run", "status": "EXECUTED", "metrics": {}}),
        encoding="utf-8",
    )
    assert not artifact_matches_run(p, run_id="new_run", allow_missing_run_id=False)


def test_matching_run_id_accepted(tmp_path: Path):
    p = tmp_path / "out.json"
    p.write_text(
        json.dumps({"run_id": "run_x", "status": "EXECUTED"}),
        encoding="utf-8",
    )
    assert artifact_matches_run(p, run_id="run_x", allow_missing_run_id=False)


def test_config_hash_mismatch_reported(tmp_path: Path):
    from experiments.federated.run_integrity import artifact_match_failure_reason

    p = tmp_path / "out.json"
    p.write_text(
        json.dumps({"run_id": "run_x", "status": "EXECUTED", "config_hash": "aaa"}),
        encoding="utf-8",
    )
    reason = artifact_match_failure_reason(
        p,
        run_id="run_x",
        config_hash_expected="bbb",
        allow_missing_run_id=False,
    )
    assert reason is not None
    assert "config_hash mismatch" in reason


def test_config_hash_stable():
    a = config_hash({"seed": 42, "model": "m"})
    b = config_hash({"model": "m", "seed": 42})
    assert a == b
