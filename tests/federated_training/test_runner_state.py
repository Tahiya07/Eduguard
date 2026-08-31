"""Master runner state and dry-run behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "federated" / "run_all_experiments.py"


def test_dry_run_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--dry-run", "--new-run", "--experiment", "env_validation"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_status_without_run():
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--status"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
