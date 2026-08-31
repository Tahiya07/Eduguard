#!/usr/bin/env python
"""Pre-flight checks before expensive GPU research execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.federated.check_gpu_environment import check_environment
from experiments.federated.experiment_registry import build_registry
from experiments.federated.run_integrity import (
    DATASET_FILES,
    config_hash,
    dataset_hashes,
    file_sha256,
    git_revision,
    load_json,
)

DP_LOCK = ROOT / "artifacts" / "privacy" / "dp_bloom_validated_v1.json"
STATE_FILE = ROOT / "experiments" / "federated" / "state" / "run_state.json"
CONFIG_DIR = ROOT / "experiments" / "federated" / "configs"


def _dp_gate_state() -> dict:
    if not DP_LOCK.is_file():
        return {"open": False, "reason": "lock file missing"}
    data = load_json(DP_LOCK) or {}
    return {
        "open": bool(data.get("validation_gate_passed")),
        "validation_gate_passed": bool(data.get("validation_gate_passed")),
        "path": str(DP_LOCK),
    }


def _multitask_gate_state() -> dict:
    marker = ROOT / "artifacts" / "federated" / "results" / "centralized_multitask_success.json"
    if not marker.is_file():
        return {"open": False, "reason": "centralized multitask marker missing"}
    data = load_json(marker) or {}
    return {"open": bool(data.get("success")), "path": str(marker)}


def run_preflight(*, require_gpu: bool = True) -> dict:
    report: dict = {
        "format": "preflight_check_v1",
        "ready": False,
        "git_revision": git_revision(),
        "dataset_hashes": dataset_hashes(),
        "blocking_issues": [],
        "warnings": [],
        "sections": {},
    }

    def block(msg: str) -> None:
        report["blocking_issues"].append(msg)

    # GPU environment
    gpu = check_environment()
    report["sections"]["gpu_environment"] = gpu
    if require_gpu and not gpu.get("ready"):
        block("GPU environment check failed")

    # Registry + configs
    registry = build_registry(str(ROOT), sys.executable)
    config_audit = []
    for spec in registry:
        if not spec.config_path:
            continue
        p = Path(spec.config_path)
        if not p.is_file():
            block(f"missing config for {spec.experiment_id}: {p}")
            continue
        cfg = load_json(p) or {}
        missing_fields = [f for f in (
            "model", "dataset", "seed", "algorithm", "resource_class"
        ) if f not in cfg]
        if missing_fields:
            block(f"{spec.experiment_id} config missing fields: {missing_fields}")
        cfg["config_hash"] = config_hash(cfg)
        config_audit.append({"experiment_id": spec.experiment_id, "path": str(p), "config_hash": cfg["config_hash"]})
    report["sections"]["experiment_configs"] = config_audit

    # Output directories writable
    for rel in (
        "artifacts/federated/results",
        "artifacts/federated/models",
        "artifacts/privacy",
        "artifacts/evaluation",
        "experiments/federated/logs",
        "experiments/federated/state",
    ):
        d = ROOT / rel
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".preflight_write_test"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as exc:
            block(f"cannot write to {rel}: {exc}")

    # State file
    if STATE_FILE.is_file():
        state = load_json(STATE_FILE) or {}
        report["sections"]["run_state"] = {
            "exists": True,
            "run_id": state.get("run_id"),
            "status": state.get("status"),
            "completed": len(state.get("completed_experiments", [])),
        }
    else:
        report["sections"]["run_state"] = {"exists": False}

    # Scientific gates
    report["sections"]["dp_gate"] = _dp_gate_state()
    report["sections"]["multitask_gate"] = _multitask_gate_state()

    # Production isolation
    try:
        import backend.service  # noqa: F401

        report["sections"]["production_import"] = {"passed": True}
    except Exception as exc:
        block(f"production backend import failed: {exc}")

    # Baseline model path must not be federated overwrite target
    locked = [
        ROOT / "models" / "qwen_bloom_trained0.5B",
        ROOT / "models" / "qwen_bloom_merged0.5B",
    ]
    for p in locked:
        if p.is_dir():
            report.setdefault("sections", {}).setdefault("baseline_artifacts", []).append(str(p))

    report["ready"] = len(report["blocking_issues"]) == 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight checks for federated research")
    parser.add_argument("--json-out", default=str(ROOT / "artifacts" / "evaluation" / "preflight_report.json"))
    parser.add_argument("--allow-no-gpu", action="store_true", help="Do not require CUDA (laptop dev only)")
    args = parser.parse_args()

    report = run_preflight(require_gpu=not args.allow_no_gpu)
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("EduGuard Pre-flight Check")
    print("=========================")
    print(f"READY: {'YES' if report['ready'] else 'NO'}")
    print(f"DP gate: {'OPEN' if report['sections']['dp_gate'].get('open') else 'CLOSED'}")
    print(f"Multitask gate: {'OPEN' if report['sections']['multitask_gate'].get('open') else 'CLOSED'}")
    for issue in report["blocking_issues"]:
        print(f"  BLOCKING: {issue}")
    print(f"\nReport: {out}")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
