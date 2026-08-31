#!/usr/bin/env python
"""Federated DP training — BLOCKED until Phase 2A validation lock exists.

Does NOT add server-side Gaussian noise. Client updates must come from the
validated centralized DP procedure referenced in dp_bloom_validated_v1.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.paths import ARTIFACTS_PRIVACY, ROOT

DP_LOCK = ARTIFACTS_PRIVACY / "dp_bloom_validated_v1.json"


def load_dp_lock() -> dict:
    if not DP_LOCK.is_file():
        raise SystemExit(
            "Federated DP is BLOCKED: missing artifacts/privacy/dp_bloom_validated_v1.json\n"
            "Run Phase 2A: python -m training.centralized.validate_dp_bloom"
        )
    data = json.loads(DP_LOCK.read_text(encoding="utf-8"))
    if not data.get("validation_gate_passed"):
        raise SystemExit(
            "Federated DP is BLOCKED: validation_gate_passed=false in dp_bloom_validated_v1.json\n"
            f"See: {ARTIFACTS_PRIVACY / 'dp_validation_failed_latest.json'}"
        )
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Federated DP Bloom training (gated)")
    parser.add_argument("--mode", choices=("federated_train", "check"), default="check")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "federated" / "results" / "federated_dp_fedavg_iid.json"))
    args = parser.parse_args()

    lock = load_dp_lock()
    print(f"[dp] DP procedure lock loaded (accountant={lock.get('accountant', 'unknown')})")
    print(
        "[dp] Privacy accounting must use training.federated.execution_stats."
        "get_privacy_accounting_steps(report) — actual global_step counts only."
    )

    if args.mode == "check":
        print("[dp] Gate open — federated DP implementation may proceed on GPU.")
        return 0

    raise SystemExit(
        "Federated DP training orchestration is not yet wired to the validated client loop.\n"
        "Phase 2A must pass on GPU first; then implement client DP using locked hyperparameters.\n"
        f"Locked procedure hash: {lock.get('training_script_sha256', 'n/a')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
