#!/usr/bin/env python
"""Federated DP training — gated on Phase 2A validation lock.

Client updates use Opacus DP-SGD per the locked centralized procedure.
The server performs FedAvg aggregation (FedProx is a client objective).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from training.paths import ARTIFACTS_FEDERATED, ROOT

from training.federated.config import DEFAULT_PROX_MU

DP_LOCK_FULL = ROOT / "artifacts" / "privacy" / "dp_bloom_validated_v1.json"
DP_LOCK_SCORE_HEAD = ROOT / "artifacts" / "privacy" / "dp_bloom_score_head_validated_v1.json"
DP_SCOPE_FULL = "full"
DP_SCOPE_SCORE_HEAD = "score-head-only"
DP_SCOPE_AUTO = "auto"

DEFAULT_TAG = "fedprox_iid_dp"
DEFAULT_OUTPUT = ROOT / "artifacts" / "federated" / "results" / "federated_dp_fedprox_iid.json"
DEFAULT_ADAPTER = (
    ROOT / "artifacts" / "federated" / "models" / "qwen_bloom_federated0.5B_fedprox_iid_dp"
)


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
            f"See: {ROOT / 'artifacts' / 'privacy' / 'dp_validation_failed_latest.json'}"
        )
    return data


def run_federated_dp_training(
    *,
    output: Path,
    dp_scope: str,
    noise_multiplier: float,
    target_delta: float,
    rounds: int,
    clients: int,
    local_epochs: float,
    experiment_tag: str,
    global_adapter: Path,
    fresh: bool,
    algorithm: str = "fedprox",
    prox_mu: float = DEFAULT_PROX_MU,
) -> int:
    lock_path = resolve_dp_lock_path(dp_scope)
    lock = load_dp_lock(dp_scope)
    scope_flag = dp_scope if dp_scope != DP_SCOPE_AUTO else (
        "score-head-only" if lock.get("dp_mode") == "score-head-only" else "full"
    )

    cmd = [
        sys.executable,
        "-m",
        "training.federated.simulation",
        "--clients",
        str(clients),
        "--rounds",
        str(rounds),
        "--local-epochs",
        str(local_epochs),
        "--algorithm",
        algorithm,
        "--prox-mu",
        str(prox_mu if algorithm == "fedprox" else 0.0),
        "--partition",
        "iid",
        "--seed",
        "42",
        "--experiment-tag",
        experiment_tag,
        "--global-adapter",
        str(global_adapter),
        "--results-json",
        str(output),
        "--enable-dp",
        "--dp-scope",
        scope_flag,
        "--dp-noise-multiplier",
        str(noise_multiplier),
        "--dp-delta",
        str(target_delta),
    ]
    if fresh:
        cmd.append("--fresh")
    cmd.append("--save-best-checkpoint")

    print("[dp] launching federated DP simulation:")
    print(" ", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    if not output.is_file():
        raise SystemExit(f"Federated DP finished but result JSON missing: {output}")
    print(f"[dp] wrote {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Federated DP Bloom training (gated)")
    parser.add_argument("--mode", choices=("federated_train", "check"), default="check")
    parser.add_argument(
        "--dp-scope",
        choices=(DP_SCOPE_AUTO, DP_SCOPE_FULL, DP_SCOPE_SCORE_HEAD),
        default=DP_SCOPE_AUTO,
        help="Which validation lock to require (auto prefers full LoRA+score if present).",
    )
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--local-epochs", type=float, default=3.0)
    parser.add_argument("--experiment-tag", default=DEFAULT_TAG)
    parser.add_argument("--global-adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedprox")
    parser.add_argument("--prox-mu", type=float, default=DEFAULT_PROX_MU)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    lock = load_dp_lock(args.dp_scope)
    lock_path = resolve_dp_lock_path(args.dp_scope)
    dp_mode = lock.get("dp_mode", lock.get("locked_procedure", {}).get("dp_scope", "unknown"))
    print(f"[dp] DP procedure lock loaded from {lock_path}")
    print(f"[dp] dp_mode={dp_mode} accountant={lock.get('locked_procedure', {}).get('accountant', 'unknown')}")
    print(
        "[dp] Privacy accounting uses per-client Opacus epsilon plus naive federated composition."
    )

    if args.mode == "check":
        print("[dp] Gate open — federated DP implementation may proceed on GPU.")
        return 0

    return run_federated_dp_training(
        output=Path(args.output),
        dp_scope=args.dp_scope,
        noise_multiplier=args.noise_multiplier,
        target_delta=args.target_delta,
        rounds=args.rounds,
        clients=args.clients,
        local_epochs=args.local_epochs,
        experiment_tag=args.experiment_tag,
        global_adapter=Path(args.global_adapter),
        fresh=bool(args.fresh),
        algorithm=args.algorithm,
        prox_mu=float(args.prox_mu),
    )


if __name__ == "__main__":
    raise SystemExit(main())
