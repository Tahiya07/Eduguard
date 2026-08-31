"""Scientific gate enforcement tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_federated_dp_blocked_without_lock():
    from training.federated import dp as fed_dp

    lock = ROOT / "artifacts" / "privacy" / "dp_bloom_validated_v1.json"
    if lock.is_file():
        data = json.loads(lock.read_text(encoding="utf-8"))
        if data.get("validation_gate_passed"):
            pytest.skip("DP lock already passed")
    with pytest.raises(SystemExit):
        fed_dp.load_dp_lock()


def test_dp_gate_blocks_federated_dp_spec():
    from experiments.federated.experiment_registry import build_registry, experiment_by_id
    from experiments.federated.run_all_experiments import _dp_gate_open, _gate_blocked
    import sys

    spec = experiment_by_id(build_registry(str(ROOT), sys.executable), "federated_dp")
    if _dp_gate_open():
        pytest.skip("DP gate open")
    reason = _gate_blocked(spec)
    assert reason is not None
    assert "DP validation" in reason


def test_laptop_mode_blocks_gpu_resource():
    from experiments.federated.experiment_registry import build_registry, experiment_by_id
    from experiments.federated.run_all_experiments import _resource_blocked
    import sys

    spec = experiment_by_id(build_registry(str(ROOT), sys.executable), "fedavg_iid")
    reason = _resource_blocked(spec, allow_gpu=False, laptop_mode=True)
    assert reason is not None
    assert "laptop mode" in reason


def test_multitask_gate_closed_by_default():
    from experiments.federated.run_all_experiments import _multitask_gate_open

    marker = ROOT / "artifacts" / "federated" / "results" / "centralized_multitask_success.json"
    if marker.is_file() and json.loads(marker.read_text()).get("success"):
        pytest.skip("multitask gate open")
    assert _multitask_gate_open() is False


def test_dp_gate_opens_with_mock_lock(tmp_path, monkeypatch):
    from experiments.federated import run_all_experiments as runner
    from experiments.federated.experiment_registry import build_registry, experiment_by_id
    import sys

    lock = tmp_path / "dp_bloom_validated_v1.json"
    lock.write_text(json.dumps({"validation_gate_passed": True}), encoding="utf-8")
    monkeypatch.setattr(runner, "DP_LOCK", lock)
    assert runner._dp_gate_open() is True
    spec = experiment_by_id(build_registry(str(ROOT), sys.executable), "federated_dp")
    assert runner._gate_blocked(spec) is None
