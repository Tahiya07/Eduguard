#!/usr/bin/env python
"""Framework FL parity gate — compare EduGuard FedAvg+IID to reference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "evaluation" / "framework_parity_gate.json"
FED_RESULT = ROOT / "artifacts" / "federated" / "results" / "federated_lora_fedavg_iid.json"
FRAMEWORK_REF_CANDIDATES = [
    ROOT.parent / "Framework" / "results" / "federated_lora_fedavg_iid1.json",
    ROOT / "artifacts" / "evaluation" / "framework_reference_fedavg_iid.json",
]

# Material gap threshold (absolute accuracy) — stop FL matrix if exceeded
MAX_ABS_GAP = 0.05


def main() -> int:
    report = {
        "status": "NOT_EXECUTED",
        "eduguard_source": str(FED_RESULT),
        "framework_reference": None,
        "eduguard_test_accuracy": None,
        "framework_test_accuracy": None,
        "absolute_gap": None,
        "parity_passed": False,
        "action": "Run fedavg_iid on GPU first.",
    }

    if not FED_RESULT.is_file():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    fed = json.loads(FED_RESULT.read_text(encoding="utf-8"))
    test = fed.get("final_test_metrics") or {}
    edu_acc = test.get("accuracy")
    report["eduguard_test_accuracy"] = edu_acc
    report["eduguard_macro_f1"] = test.get("macro_f1")
    report["status"] = "COMPUTED"

    fw_acc = None
    fw_path = next((p for p in FRAMEWORK_REF_CANDIDATES if p.is_file()), None)
    if fw_path:
        report["framework_reference"] = str(fw_path)
        fw = json.loads(fw_path.read_text(encoding="utf-8"))
        fw_test = fw.get("final_test_metrics") or fw.get("test_metrics") or fw
        fw_acc = fw_test.get("accuracy")
        report["framework_test_accuracy"] = fw_acc
        report["framework_macro_f1"] = fw_test.get("macro_f1")

    if edu_acc is not None and fw_acc is not None:
        gap = abs(float(edu_acc) - float(fw_acc))
        report["absolute_gap"] = gap
        report["parity_passed"] = gap <= MAX_ABS_GAP
        if not report["parity_passed"]:
            report["action"] = (
                f"STOP: accuracy gap {gap:.4f} exceeds {MAX_ABS_GAP}. Diagnose before continuing FL matrix."
            )
        else:
            report["action"] = "Parity within tolerance — proceed with FL experiments."
    else:
        report["action"] = "Framework reference not found locally — record EduGuard result only; compare manually."

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report.get("parity_passed") is False and report.get("absolute_gap") is not None:
        return 1
    return 0 if FED_RESULT.is_file() else 2


if __name__ == "__main__":
    raise SystemExit(main())
