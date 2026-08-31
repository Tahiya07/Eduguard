"""Runner dependency order for post-FedAvg diagnostic experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.federated.experiment_registry import build_registry, experiment_by_id


def _index(registry, experiment_id: str) -> int:
    for i, spec in enumerate(registry):
        if spec.experiment_id == experiment_id:
            return i
    raise KeyError(experiment_id)


def test_diagnostic_experiment_order():
    registry = build_registry(str(ROOT), sys.executable)
    order = [
        "fedavg_iid",
        "framework_parity_audit",
        "fedavg_iid_r20",
        "fedavg_iid_localepoch1",
        "fedprox_iid",
        "fl_baseline_diagnosis",
        "utility_gap_analysis",
    ]
    indices = [_index(registry, eid) for eid in order]
    assert indices == sorted(indices), f"expected monotonic order, got {order}"

    fedavg = experiment_by_id(registry, "fedavg_iid")
    assert "fl_smoke_fedavg_iid" in fedavg.prerequisites

    audit = experiment_by_id(registry, "framework_parity_audit")
    assert audit.prerequisites == ["fedavg_iid"]

    r20 = experiment_by_id(registry, "fedavg_iid_r20")
    assert r20.prerequisites == ["framework_parity_audit"]
    assert "fedavg_iid_r20" in " ".join(r20.command)

    localepoch = experiment_by_id(registry, "fedavg_iid_localepoch1")
    assert localepoch.prerequisites == ["fedavg_iid_r20"]

    fedprox = experiment_by_id(registry, "fedprox_iid")
    assert fedprox.prerequisites == ["fedavg_iid_localepoch1"]

    diagnosis = experiment_by_id(registry, "fl_baseline_diagnosis")
    assert diagnosis.prerequisites == ["fedprox_iid"]


def test_targeted_experiments_use_unique_tags():
    registry = build_registry(str(ROOT), sys.executable)
    tags = []
    for eid in ("fedavg_iid_r20", "fedavg_iid_localepoch1", "fedprox_iid"):
        spec = experiment_by_id(registry, eid)
        cmd = " ".join(spec.command)
        assert "--experiment-tag" in cmd
        tag = cmd.split("--experiment-tag", 1)[1].strip().split()[0]
        tags.append(tag)
    assert len(tags) == len(set(tags))
