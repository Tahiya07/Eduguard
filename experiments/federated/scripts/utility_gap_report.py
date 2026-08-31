#!/usr/bin/env python
"""Compare federated vs centralized Bloom metrics (utility gap diagnosis)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_JSON = ROOT / "artifacts" / "evaluation" / "utility_gap_report.json"
OUT_MD = ROOT / "artifacts" / "evaluation" / "utility_gap_report.md"

CENTRALIZED_CANDIDATES = [
    ROOT / "results" / "bloom_lora_eval_0.5B.json",
    ROOT / "artifacts" / "evaluation" / "bloom_centralized_eval.json",
]
FED_CANDIDATES = [
    ("fedavg_iid", ROOT / "artifacts" / "federated" / "results" / "federated_lora_fedavg_iid.json"),
    ("fedprox_iid", ROOT / "artifacts" / "federated" / "results" / "federated_lora_fedprox_iid.json"),
    ("fedavg_noniid_a05", ROOT / "artifacts" / "federated" / "results" / "federated_lora_fedavg_noniid_a0.5.json"),
]


def _metrics_from_result(data: dict) -> dict:
    test = data.get("final_test_metrics") or data.get("test_metrics") or data
    return {
        "accuracy": test.get("accuracy"),
        "macro_f1": test.get("macro_f1"),
    }


def main() -> int:
    central_path = next((p for p in CENTRALIZED_CANDIDATES if p.is_file()), None)
    central = json.loads(central_path.read_text()) if central_path else None

    rows = []
    for name, path in FED_CANDIDATES:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        m = _metrics_from_result(data)
        gap = None
        if central and m.get("accuracy") is not None:
            c_acc = central.get("test_accuracy") or central.get("accuracy")
            if c_acc is not None:
                gap = float(c_acc) - float(m["accuracy"])
        rows.append({"experiment": name, **m, "accuracy_gap_vs_central": gap})

    report = {
        "run_id": os.environ.get("EDUGUARD_RUN_ID"),
        "status": "COMPUTED" if rows else "NOT_EXECUTED",
        "centralized_source": str(central_path) if central_path else None,
        "centralized_metrics": _metrics_from_result(central) if central else None,
        "comparisons": rows,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = ["# Utility Gap Report", ""]
    if central:
        md_lines.append(f"Centralized source: `{central_path}`")
        cm = report["centralized_metrics"]
        md_lines.append(f"- Centralized accuracy: {cm.get('accuracy')}")
        md_lines.append(f"- Centralized macro-F1: {cm.get('macro_f1')}")
        md_lines.append("")
    md_lines.append("| Experiment | Accuracy | Macro-F1 | Gap vs Central |")
    md_lines.append("|---|---:|---:|---:|")
    for row in rows:
        md_lines.append(
            f"| {row['experiment']} | {row.get('accuracy')} | {row.get('macro_f1')} | {row.get('accuracy_gap_vs_central')} |"
        )
    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
