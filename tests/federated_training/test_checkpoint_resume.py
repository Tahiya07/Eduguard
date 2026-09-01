"""Round checkpoint resume logic (no model training)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.federated.checkpoint import resolve_round_resume, write_round_checkpoint


def test_round_checkpoint_resume_start_round():
    ckpt = {"last_completed_round": 3, "config_hash": "x", "history": []}
    start_round = int(ckpt.get("last_completed_round", 0)) + 1
    assert start_round == 4


def test_auto_resume_from_partial_checkpoint(tmp_path: Path):
    ckpt_path = tmp_path / "round_checkpoint.json"
    write_round_checkpoint(
        ckpt_path,
        last_completed_round=5,
        config_hash="abc123",
        history=[{"round": i, "status": "COMPLETE"} for i in range(1, 6)],
        total_upload=100,
        total_download=200,
        trainable_parameters=1000,
        trainable_param_breakdown=None,
        adapter_bytes=500,
        global_adapter=str(tmp_path / "adapter"),
        start_time="2026-01-01T00:00:00+00:00",
    )
    state = resolve_round_resume(
        ckpt_path,
        config_hash="abc123",
        configured_rounds=20,
        resume_requested=False,
        fresh_requested=False,
    )
    assert state.should_resume is True
    assert state.start_round == 6
    assert state.last_completed_round == 5
    assert len(state.history) == 5
    assert state.total_upload == 100
    assert state.training_complete is False


def test_auto_resume_without_explicit_resume_flag(tmp_path: Path):
    ckpt_path = tmp_path / "round_checkpoint.json"
    write_round_checkpoint(
        ckpt_path,
        last_completed_round=2,
        config_hash="same",
        history=[{"round": 1}, {"round": 2}],
        total_upload=10,
        total_download=20,
        trainable_parameters=1,
        trainable_param_breakdown=None,
        adapter_bytes=1,
        global_adapter="/tmp/adapter",
        start_time="t0",
    )
    state = resolve_round_resume(
        ckpt_path,
        config_hash="same",
        configured_rounds=5,
        resume_requested=False,
        fresh_requested=False,
    )
    assert state.start_round == 3


def test_fresh_start_ignores_checkpoint(tmp_path: Path):
    ckpt_path = tmp_path / "round_checkpoint.json"
    write_round_checkpoint(
        ckpt_path,
        last_completed_round=4,
        config_hash="abc",
        history=[{"round": 1}],
        total_upload=1,
        total_download=1,
        trainable_parameters=1,
        trainable_param_breakdown=None,
        adapter_bytes=1,
        global_adapter="/tmp/adapter",
        start_time="t0",
    )
    state = resolve_round_resume(
        ckpt_path,
        config_hash="abc",
        configured_rounds=20,
        resume_requested=False,
        fresh_requested=True,
    )
    assert state.should_resume is False
    assert state.start_round == 1
    assert state.history == []


def test_config_hash_mismatch_blocks_silent_restart(tmp_path: Path):
    ckpt_path = tmp_path / "round_checkpoint.json"
    ckpt_path.write_text(
        json.dumps({"last_completed_round": 3, "config_hash": "old", "history": []}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="Refusing to start fresh"):
        resolve_round_resume(
            ckpt_path,
            config_hash="new",
            configured_rounds=20,
            resume_requested=False,
            fresh_requested=False,
        )


def test_training_complete_skips_remaining_rounds(tmp_path: Path):
    ckpt_path = tmp_path / "round_checkpoint.json"
    write_round_checkpoint(
        ckpt_path,
        last_completed_round=20,
        config_hash="done",
        history=[{"round": i} for i in range(1, 21)],
        total_upload=1,
        total_download=1,
        trainable_parameters=1,
        trainable_param_breakdown=None,
        adapter_bytes=1,
        global_adapter="/tmp/adapter",
        start_time="t0",
    )
    state = resolve_round_resume(
        ckpt_path,
        config_hash="done",
        configured_rounds=20,
        resume_requested=False,
        fresh_requested=False,
    )
    assert state.training_complete is True
    assert state.start_round == 21
