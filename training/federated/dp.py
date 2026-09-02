#!/usr/bin/env python
"""Federated DP training — BLOCKED until Phase 2A validation lock exists.

Does NOT add server-side Gaussian noise. Client updates must come from the
validated centralized DP procedure referenced in a dp_bloom_*_validated_v1.json lock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.paths import ARTIFACTS_PRIVACY, ROOT

DP_LOCK_FULL = ARTIFACTS_PRIVACY / "dp_bloom_validated_v1.json"
DP_LOCK_SCORE_HEAD = ARTIFACTS_PRIVACY / "dp_bloom_score_head_validated_v1.json"
DP_SCOPE_FULL = "full"
DP_SCOPE_SCORE_HEAD = "score-head-only"
DP_SCOPE_AUTO = "auto"


def resolve_dp_lock_path(scope: str = DP_SCOPE_AUTO) -> Path:
    if scope == DP_SCOPE_FULL:
        return DP_LOCK_FULL
    if scope == DP_SCOPE_SCORE_HEAD:
        return DP_LOCK_SCORE_HEAD
    if DP_LOCK_FULL.is_file():
        data = json.loads(DP_LOCK_FULL.read_text(encoding="utf-8"))
        if data.get("validation_gate_passed"):
            return DP_LOCK_FULL
    return DP_LOCK_SCORE_HEAD


def load_dp_lock(scope: str = DP_SCOPE_AUTO) -> dict:
    path = resolve_dp_lock_path(scope)
    if not path.is_file():
        raise SystemExit(
            f"Federated DP is BLOCKED: missing {path}\n"
            "Run Phase 2A:\n"
            "  python -m training.centralized.validate_dp_bloom\n"
            "  python -m training.centralized.validate_dp_bloom --dp-mode score-head-only"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("validation_gate_passed"):
        raise SystemExit(
            f"Federated DP is BLOCKED: validation_gate_passed=false in {path}\n"
            f"See: {ARTIFACTS_PRIVACY / 'dp_validation_failed_latest.json'}"
        )
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Federated DP Bloom training (gated)")
    parser.add_argument("--mode", choices=("federated_train", "check"), default="check")
    parser.add_argument(
        "--dp-scope",
        choices=(DP_SCOPE_AUTO, DP_SCOPE_FULL, DP_SCOPE_SCORE_HEAD),
        default=DP_SCOPE_AUTO,
        help="Which validation lock to require (auto prefers full LoRA+score if present).",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "federated" / "results" / "federated_dp_fedavg_iid.json"),
    )
    args = parser.parse_args()

    lock = load_dp_lock(args.dp_scope)
    lock_path = resolve_dp_lock_path(args.dp_scope)
    dp_mode = lock.get("dp_mode", lock.get("locked_procedure", {}).get("dp_scope", "unknown"))
    print(f"[dp] DP procedure lock loaded from {lock_path}")
    print(f"[dp] dp_mode={dp_mode} accountant={lock.get('locked_procedure', {}).get('accountant', 'unknown')}")
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
        f"Locked procedure hash: {lock.get('train_script_sha256', 'n/a')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
