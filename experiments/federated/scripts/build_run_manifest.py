#!/usr/bin/env python
"""Build or refresh the latest run manifest from run_state.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "experiments" / "federated" / "state" / "run_state.json"
OUT_DIR = ROOT / "experiments" / "federated" / "results" / "runs"


def main() -> int:
    if not STATE.is_file():
        print("No run_state.json found.")
        return 1
    state = json.loads(STATE.read_text(encoding="utf-8"))
    run_id = state.get("run_id", "unknown")
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "git_revision": state.get("git_revision"),
        "environment": state.get("environment"),
        "completed_experiments": state.get("completed_experiments", []),
        "failed_experiments": state.get("failed_experiments", []),
        "blocked_experiments": state.get("blocked_experiments", []),
        "status": state.get("status"),
        "artifact_index": {
            "federated_results": str(ROOT / "artifacts" / "federated" / "results"),
            "privacy": str(ROOT / "artifacts" / "privacy"),
            "evaluation": str(ROOT / "artifacts" / "evaluation"),
        },
    }
    out = run_dir / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
