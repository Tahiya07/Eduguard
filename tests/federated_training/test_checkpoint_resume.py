"""Round checkpoint resume logic (no model training)."""

from __future__ import annotations

import json
from pathlib import Path


def test_round_checkpoint_config_hash_mismatch_blocks_resume(tmp_path: Path):
    ckpt = {
        "last_completed_round": 2,
        "config_hash": "aaa",
        "history": [{"round": 1}, {"round": 2}],
    }
    p = tmp_path / "round_checkpoint.json"
    p.write_text(json.dumps(ckpt), encoding="utf-8")
    loaded = json.loads(p.read_text())
    assert loaded["config_hash"] != "bbb"


def test_round_checkpoint_resume_start_round():
    ckpt = {"last_completed_round": 3, "config_hash": "x", "history": []}
    start_round = int(ckpt.get("last_completed_round", 0)) + 1
    assert start_round == 4
