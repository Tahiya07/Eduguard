"""Run federated FL matrix and compare against Framework baseline references."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from training.paths import ARTIFACTS_FEDERATED, ROOT

FRAMEWORK_BASELINE = Path(r"C:\Users\tahiy\PycharmProjects\Framework\results\federated_lora_fedavg_iid1.json")

MATRIX = [
    {"algorithm": "fedavg", "partition": "iid", "alpha": 0.5},
    {"algorithm": "fedprox", "partition": "iid", "alpha": 0.5},
    {"algorithm": "fedavg", "partition": "non_iid_label", "alpha": 0.5},
    {"algorithm": "fedprox", "partition": "non_iid_label", "alpha": 0.5},
]


def _tag(spec: dict) -> str:
    from training.federated.config import setting_tag

    return setting_tag(
        algorithm=spec["algorithm"],
        partition=spec["partition"],
        alpha=spec["alpha"] if spec["partition"] == "non_iid_label" else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default=None, help="Run single tag e.g. fedavg_iid")
    args = parser.parse_args()

    py = sys.executable
    sim = ROOT / "training" / "federated" / "simulation.py"
    reports = []

    for spec in MATRIX:
        tag = _tag(spec)
        if args.only and args.only not in (tag, spec["algorithm"]):
            continue
        cmd = [
            py,
            str(sim),
            "--clients",
            str(args.clients),
            "--rounds",
            str(args.rounds),
            "--algorithm",
            spec["algorithm"],
            "--partition",
            spec["partition"],
            "--alpha",
            str(spec["alpha"]),
        ]
        print("[matrix]", tag, "->", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), check=True)
        result_path = ARTIFACTS_FEDERATED / "results" / f"federated_lora_{tag}.json"
        if result_path.is_file():
            reports.append(json.loads(result_path.read_text(encoding="utf-8")))

    parity = {"eduguard_runs": len(reports), "framework_baseline": None}
    if FRAMEWORK_BASELINE.is_file():
        parity["framework_baseline"] = json.loads(FRAMEWORK_BASELINE.read_text(encoding="utf-8"))

    out = ARTIFACTS_FEDERATED / "results" / "framework_parity_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"parity": parity, "reports": reports}, indent=2), encoding="utf-8")
    print(f"[matrix] parity report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
