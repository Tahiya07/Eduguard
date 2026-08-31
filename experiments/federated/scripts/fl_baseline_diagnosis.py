#!/usr/bin/env python
"""FL baseline diagnosis report across targeted FedAvg/FedProx experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
OUT_JSON = ROOT / "artifacts" / "evaluation" / "fl_baseline_diagnosis.json"
OUT_MD = ROOT / "artifacts" / "evaluation" / "fl_baseline_diagnosis.md"
BASELINE_LOCK = ROOT / "artifacts" / "evaluation" / "fedavg_iid_baseline_lock.json"

EXPERIMENTS = [
    ("framework_reference", None, "historical Framework FedAvg+IID"),
    ("fedavg_iid", ROOT / "artifacts" / "federated" / "results" / "federated_lora_fedavg_iid.json", "5 rounds x 3 local epochs (immutable baseline)"),
    ("fedavg_iid_r20", ROOT / "artifacts" / "federated" / "results" / "federated_lora_fedavg_iid_r20.json", "20 rounds x 3 local epochs"),
    ("fedavg_iid_localepoch1", ROOT / "artifacts" / "federated" / "results" / "federated_lora_fedavg_iid_localepoch1.json", "5 rounds x 1 local epoch"),
    ("fedprox_iid", ROOT / "artifacts" / "federated" / "results" / "federated_lora_fedprox_iid.json", "FedProx mu=0.01, 5 rounds x 3 local epochs"),
]

FRAMEWORK_REF_CANDIDATES = [
    ROOT.parent / "Framework" / "results" / "federated_lora_fedavg_iid1.json",
    ROOT / "artifacts" / "evaluation" / "framework_reference_fedavg_iid.json",
]


def _load(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_result(name: str, data: Optional[dict], description: str) -> dict:
    if data is None:
        return {
            "experiment": name,
            "description": description,
            "status": "NOT_EXECUTED",
        }
    test = data.get("final_test_metrics") or data.get("metrics") or {}
    training = data.get("training") or {}
    comm = data.get("communication") or {}
    return {
        "experiment": name,
        "description": description,
        "status": data.get("status", "UNKNOWN"),
        "accuracy": test.get("accuracy"),
        "macro_f1": test.get("macro_f1"),
        "quadratic_weighted_kappa": test.get("quadratic_weighted_kappa"),
        "within_one_level_accuracy": test.get("within_one_level_accuracy"),
        "severe_error_rate": test.get("severe_error_rate"),
        "ece": test.get("ece"),
        "per_class_f1": test.get("per_class_f1") or test.get("per_class"),
        "runtime_seconds": data.get("runtime_seconds"),
        "communication": {
            "upload_bytes": comm.get("upload_bytes"),
            "download_bytes": comm.get("download_bytes"),
            "total_bytes": comm.get("total_bytes"),
            "trainable_parameter_count": comm.get("trainable_parameter_count"),
            "adapter_size_bytes": comm.get("adapter_size_bytes"),
        },
        "optimizer_steps": {
            "configured_estimate": training.get("total_optimizer_steps_estimate"),
            "actual_completed": (training.get("actual") or {}).get("total_optimizer_steps_completed"),
        },
        "training_budget_comparison": training.get("training_budget_comparison"),
        "artifact_path": (data.get("artifact_paths") or {}).get("results_json"),
    }


def _framework_reference_row() -> dict:
    fw_path = next((p for p in FRAMEWORK_REF_CANDIDATES if p.is_file()), None)
    data = _load(fw_path)
    if data:
        row = _row_from_result("framework_reference", data, "Framework FedAvg+IID reference JSON")
        row["artifact_path"] = str(fw_path)
        return row
    lock = _load(BASELINE_LOCK)
    return {
        "experiment": "framework_reference",
        "description": "Historical Framework FedAvg+IID (documented)",
        "status": "DOCUMENTED_ONLY",
        "accuracy": 0.503,
        "artifact_path": "Framework/results/federated_lora_fedavg_iid1.json",
        "note": "Local Framework JSON not found; using documented ~50.3% test accuracy.",
        "framework_reference_available": False,
        "baseline_lock_gap": (lock or {}).get("absolute_gap_vs_framework"),
    }


def _select_best(rows: List[dict]) -> dict:
    executed = [
        r
        for r in rows
        if r.get("experiment") != "framework_reference"
        and r.get("accuracy") is not None
        and r.get("status") not in {"NOT_EXECUTED", "STRUCTURE_ONLY"}
    ]
    if not executed:
        return {
            "best_observed_configuration": None,
            "reason": "No executed EduGuard FL experiments with test metrics yet.",
            "comparison_is_fair": False,
            "test_set_used_for_selection": False,
        }
    best = max(executed, key=lambda r: float(r["accuracy"]))
    return {
        "best_observed_configuration": best["experiment"],
        "best_accuracy": best.get("accuracy"),
        "reason": "Highest test accuracy among executed targeted experiments (not claimed optimal).",
        "comparison_is_fair": False,
        "comparison_fairness_note": (
            "Experiments differ by rounds/local epochs/algorithm; budgets are not matched to centralized training."
        ),
        "test_set_used_for_selection": False,
        "selection_policy": "Report-only ranking; validation set should be used for configuration selection.",
    }


def build_report() -> dict:
    rows = [_framework_reference_row()]
    for name, path, desc in EXPERIMENTS[1:]:
        rows.append(_row_from_result(name, _load(path), desc))

    # Prefer immutable baseline lock for fedavg_iid if result file missing or overwritten
    lock = _load(BASELINE_LOCK)
    for i, row in enumerate(rows):
        if row.get("experiment") == "fedavg_iid" and lock and row.get("status") == "NOT_EXECUTED":
            rows[i] = {
                **_row_from_result("fedavg_iid", {"final_test_metrics": lock["final_test_metrics"], "status": "IMMUTABLE_BASELINE", "runtime_seconds": lock.get("runtime_seconds")}, row["description"]),
                "immutable_baseline_lock": str(BASELINE_LOCK),
            }

    return {
        "run_id": os.environ.get("EDUGUARD_RUN_ID"),
        "status": "COMPUTED",
        "dp_status": "NOT VALIDATED",
        "experiments": rows,
        "best_observed": _select_best(rows),
        "immutable_baseline": {
            "experiment_id": "fedavg_iid",
            "accuracy": 0.5686,
            "macro_f1": 0.4962,
            "lock_file": str(BASELINE_LOCK),
        },
    }


def write_markdown(report: dict) -> str:
    lines = [
        "# FL Baseline Diagnosis",
        "",
        f"**DP status:** {report.get('dp_status')}",
        "",
        "## Experiment comparison",
        "",
        "| Experiment | Status | Accuracy | Macro-F1 | QWK | Within-1 | Severe err | ECE | Runtime (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("experiments") or []:
        lines.append(
            f"| {row.get('experiment')} | {row.get('status')} | {row.get('accuracy')} | "
            f"{row.get('macro_f1')} | {row.get('quadratic_weighted_kappa')} | "
            f"{row.get('within_one_level_accuracy')} | {row.get('severe_error_rate')} | "
            f"{row.get('ece')} | {row.get('runtime_seconds')} |"
        )
    best = report.get("best_observed") or {}
    lines.extend(
        [
            "",
            "## Best observed configuration",
            "",
            f"- Configuration: {best.get('best_observed_configuration')}",
            f"- Reason: {best.get('reason')}",
            f"- Fair comparison: {best.get('comparison_is_fair')}",
            f"- Test set used for selection: {best.get('test_set_used_for_selection')}",
            "",
            "## Immutable baseline",
            "",
            f"- fedavg_iid accuracy: {report.get('immutable_baseline', {}).get('accuracy')}",
            f"- Lock file: `{report.get('immutable_baseline', {}).get('lock_file')}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(write_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
