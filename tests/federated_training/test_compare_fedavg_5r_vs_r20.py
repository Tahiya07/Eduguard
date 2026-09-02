"""Tests for fedavg 5r vs r20 comparison script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_compare_script_picks_r20_from_artifacts():
    from experiments.federated.scripts import compare_fedavg_5r_vs_r20 as cmp

    result_5r = ROOT / "artifacts/federated/results/federated_lora_fedavg_iid.json"
    result_r20 = ROOT / "artifacts/federated/results/federated_lora_fedavg_iid_r20.json"
    if not result_5r.is_file() or not result_r20.is_file():
        pytest.skip("federated result artifacts missing")

    m5 = cmp._metrics_from_fl_result(json.loads(result_5r.read_text(encoding="utf-8")))
    m20 = cmp._metrics_from_fl_result(json.loads(result_r20.read_text(encoding="utf-8")))
    assert cmp._pick_winner(m5, m20) == "fedavg_iid_r20"
    assert m20["accuracy"] > m5["accuracy"]
    assert m20["optimizer_steps"] == 6240
    assert m5["optimizer_steps"] == 1560


def test_comparison_report_written():
    out = ROOT / "artifacts/evaluation/fedavg_5r_vs_r20_comparison.json"
    if not out.is_file():
        pytest.skip("comparison report not generated yet")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["winner"] == "fedavg_iid_r20"
    assert data["comparison_is_fair"] is False
    assert data["deltas"]["accuracy"] > 0
