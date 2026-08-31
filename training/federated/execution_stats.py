"""Federated training execution statistics — configured estimates vs actual counters."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def read_trainer_execution_stats(trainer) -> Dict[str, Any]:
    """Read actual optimizer steps from a completed Hugging Face Trainer."""
    state = trainer.state
    global_step = int(getattr(state, "global_step", 0) or 0)
    return {
        "optimizer_steps_completed": global_step,
        "epochs_completed": float(getattr(state, "epoch", 0.0) or 0.0),
        "max_steps": getattr(state, "max_steps", None),
        "logging_steps": getattr(state, "logging_steps", None),
    }


def execution_from_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract execution block written by client.py into a transport bundle."""
    block = bundle.get("execution") or {}
    steps = block.get("optimizer_steps_completed")
    return {
        "client_id": bundle.get("client_id"),
        "round": int(bundle.get("round", 0)),
        "optimizer_steps_completed": int(steps) if steps is not None else None,
        "epochs_completed": block.get("epochs_completed"),
        "n_samples": int(bundle.get("n_samples", 0)),
    }


def summarize_round_bundles(bundles: List[Mapping[str, Any]], round_idx: int) -> Dict[str, Any]:
    per_client: Dict[str, int] = {}
    for bundle in bundles:
        ex = execution_from_bundle(bundle)
        cid = ex.get("client_id")
        steps = ex.get("optimizer_steps_completed")
        if cid and steps is not None:
            per_client[str(cid)] = int(steps)
    round_total = int(sum(per_client.values()))
    return {
        "round": int(round_idx),
        "optimizer_steps_per_client": per_client,
        "optimizer_steps_round_total": round_total,
        "clients_reporting": len(per_client),
    }


def summarize_execution_actual(
    history: List[Mapping[str, Any]],
    *,
    configured_rounds: int,
) -> Dict[str, Any]:
    """Aggregate actual execution counters from per-round history records."""
    per_round: List[Dict[str, Any]] = []
    per_client_per_round: Dict[str, Dict[str, int]] = {}
    total_steps = 0
    rounds_completed = 0

    for rec in history:
        ex = rec.get("execution") or {}
        if not ex or ex.get("optimizer_steps_round_total") is None:
            continue
        round_idx = int(rec.get("round", ex.get("round", 0)))
        rounds_completed += 1
        round_total = int(ex.get("optimizer_steps_round_total") or 0)
        total_steps += round_total
        per_client = dict(ex.get("optimizer_steps_per_client") or {})
        per_round.append(
            {
                "round": round_idx,
                "optimizer_steps_per_client": per_client,
                "optimizer_steps_round_total": round_total,
            }
        )
        per_client_per_round[f"round_{round_idx:02d}"] = per_client

    if rounds_completed == 0:
        status = "NOT_EXECUTED"
    elif rounds_completed < int(configured_rounds):
        status = "PARTIAL"
    else:
        status = "COMPLETE"

    return {
        "federated_rounds_configured": int(configured_rounds),
        "federated_rounds_completed": rounds_completed,
        "optimizer_steps_per_client_per_round": per_client_per_round,
        "optimizer_steps_per_round_all_clients": [r["optimizer_steps_round_total"] for r in per_round],
        "total_optimizer_steps_completed": total_steps,
        "per_round": per_round,
        "execution_status": status,
        "privacy_accounting_steps_source": (
            "actual_global_step" if total_steps > 0 else "none"
        ),
        "privacy_accounting_steps": total_steps if total_steps > 0 else None,
    }


def get_privacy_accounting_steps(report: Mapping[str, Any]) -> Optional[int]:
    """Return privacy-relevant step count: actual execution only, never estimates."""
    actual = (report.get("training") or {}).get("actual") or {}
    steps = actual.get("privacy_accounting_steps")
    if steps is not None:
        return int(steps)
    total = actual.get("total_optimizer_steps_completed")
    return int(total) if total else None
