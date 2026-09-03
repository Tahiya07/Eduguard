"""Federated round checkpoint load/save helpers for simulation resume."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RoundResumeState:
    """Resolved simulation resume state from round_checkpoint.json."""

    should_resume: bool
    start_round: int
    history: List[Dict[str, Any]]
    total_upload: int
    total_download: int
    trainable_parameters: Optional[int]
    trainable_param_breakdown: Optional[Dict[str, Any]]
    adapter_bytes: Optional[int]
    start_time: Optional[str]
    last_completed_round: int
    training_complete: bool
    global_adapter: Optional[str]
    best_checkpoint: Optional[Dict[str, Any]] = None


def load_round_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_round_resume(
    checkpoint_path: Path,
    *,
    config_hash: str,
    configured_rounds: int,
    resume_requested: bool,
    fresh_requested: bool,
) -> RoundResumeState:
    """Decide whether to resume FL training from an on-disk round checkpoint.

    Auto-resumes when a compatible partial checkpoint exists (last_completed_round
    in ``1..configured_rounds-1``) unless ``fresh_requested`` is set.
    """
    empty = RoundResumeState(
        should_resume=False,
        start_round=1,
        history=[],
        total_upload=0,
        total_download=0,
        trainable_parameters=None,
        trainable_param_breakdown=None,
        adapter_bytes=None,
        start_time=None,
        last_completed_round=0,
        training_complete=False,
        global_adapter=None,
        best_checkpoint=None,
    )
    if fresh_requested:
        return empty

    ckpt = load_round_checkpoint(checkpoint_path)
    if not ckpt:
        return empty

    ckpt_hash = ckpt.get("config_hash")
    if ckpt_hash != config_hash:
        if resume_requested:
            raise SystemExit(
                f"Refusing resume: config_hash mismatch ({ckpt_hash} vs {config_hash}). "
                "Use --fresh to discard the checkpoint and restart."
            )
        raise SystemExit(
            f"Refusing to start fresh: existing checkpoint at {checkpoint_path} "
            f"has config_hash {ckpt_hash!r} but current run is {config_hash!r}. "
            "Use --fresh to discard the checkpoint and restart, or match the original config."
        )

    last_completed = int(ckpt.get("last_completed_round", 0) or 0)
    comm_acc = ckpt.get("communication_accum") or {}
    best_checkpoint = ckpt.get("best_checkpoint")

    if last_completed >= int(configured_rounds):
        return RoundResumeState(
            should_resume=True,
            start_round=int(configured_rounds) + 1,
            history=list(ckpt.get("history", [])),
            total_upload=int(comm_acc.get("total_upload", 0)),
            total_download=int(comm_acc.get("total_download", 0)),
            trainable_parameters=ckpt.get("trainable_parameters"),
            trainable_param_breakdown=ckpt.get("trainable_param_breakdown"),
            adapter_bytes=ckpt.get("adapter_bytes"),
            start_time=ckpt.get("start_time"),
            last_completed_round=last_completed,
            training_complete=True,
            global_adapter=ckpt.get("global_adapter"),
            best_checkpoint=best_checkpoint,
        )

    if last_completed <= 0:
        return empty

    return RoundResumeState(
        should_resume=True,
        start_round=last_completed + 1,
        history=list(ckpt.get("history", [])),
        total_upload=int(comm_acc.get("total_upload", 0)),
        total_download=int(comm_acc.get("total_download", 0)),
        trainable_parameters=ckpt.get("trainable_parameters"),
        trainable_param_breakdown=ckpt.get("trainable_param_breakdown"),
        adapter_bytes=ckpt.get("adapter_bytes"),
        start_time=ckpt.get("start_time"),
        last_completed_round=last_completed,
        training_complete=False,
        global_adapter=ckpt.get("global_adapter"),
        best_checkpoint=best_checkpoint,
    )


def write_round_checkpoint(
    path: Path,
    *,
    last_completed_round: int,
    config_hash: str,
    history: List[Dict[str, Any]],
    total_upload: int,
    total_download: int,
    trainable_parameters: Optional[int],
    trainable_param_breakdown: Optional[Dict[str, Any]],
    adapter_bytes: Optional[int],
    global_adapter: str,
    start_time: Optional[str],
    best_checkpoint: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_completed_round": int(last_completed_round),
        "config_hash": config_hash,
        "history": history,
        "communication_accum": {
            "total_upload": int(total_upload),
            "total_download": int(total_download),
        },
        "trainable_parameters": trainable_parameters,
        "trainable_param_breakdown": trainable_param_breakdown,
        "adapter_bytes": adapter_bytes,
        "global_adapter": global_adapter,
        "start_time": start_time,
        "best_checkpoint": best_checkpoint,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
