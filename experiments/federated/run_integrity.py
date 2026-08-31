"""Run integrity helpers: hashes, result validation, metadata envelopes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

DATASET_FILES = (
    "data/figshare_bloom_v1_train.csv",
    "data/figshare_bloom_v1_val.csv",
    "data/figshare_bloom_v1_test.csv",
)


def git_revision(repo: Path | None = None) -> str:
    repo = repo or ROOT
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_hashes(repo: Path | None = None) -> Dict[str, Optional[str]]:
    repo = repo or ROOT
    out: Dict[str, Optional[str]] = {}
    for rel in DATASET_FILES:
        out[rel] = file_sha256(repo / rel)
    return out


def config_hash(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def result_is_placeholder(data: Dict[str, Any]) -> bool:
    status = str(data.get("status", "")).upper()
    if status in {"NOT_EXECUTED", "NOT_IMPLEMENTED", "PLACEHOLDER"}:
        return True
    if data.get("validation_gate_passed") is False and "gates" in data:
        return False  # DP failure report is a valid artifact, not a placeholder success
    return False


def artifact_match_failure_reason(
    path: Path,
    *,
    run_id: str,
    git_rev: Optional[str] = None,
    config_hash_expected: Optional[str] = None,
    allow_missing_run_id: bool = False,
) -> Optional[str]:
    """Return a failure reason, or None if the artifact matches the active run."""
    data = load_json(path)
    if data is None:
        return "invalid or missing JSON"
    if result_is_placeholder(data):
        return f"placeholder status={data.get('status')!r}"

    artifact_run = data.get("run_id")
    if artifact_run is not None:
        if artifact_run != run_id:
            return f"run_id mismatch (artifact={artifact_run!r}, expected={run_id!r})"
    elif not allow_missing_run_id:
        return "missing run_id"

    if git_rev and data.get("git_revision") not in (None, git_rev):
        return (
            f"git_revision mismatch (artifact={data.get('git_revision')!r}, "
            f"expected={git_rev!r})"
        )

    if config_hash_expected and data.get("config_hash") not in (None, config_hash_expected):
        return (
            f"config_hash mismatch (artifact={data.get('config_hash')!r}, "
            f"expected={config_hash_expected!r})"
        )

    return None


def artifact_matches_run(
    path: Path,
    *,
    run_id: str,
    git_rev: Optional[str] = None,
    config_hash_expected: Optional[str] = None,
    allow_missing_run_id: bool = False,
) -> bool:
    """Return True only if artifact belongs to the active run and is not a placeholder."""
    return (
        artifact_match_failure_reason(
            path,
            run_id=run_id,
            git_rev=git_rev,
            config_hash_expected=config_hash_expected,
            allow_missing_run_id=allow_missing_run_id,
        )
        is None
    )


def build_result_envelope(
    *,
    run_id: str,
    experiment_id: str,
    config_payload: Dict[str, Any],
    model_identifier: str,
    seed: int,
    status: str = "EXECUTED",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from datetime import datetime, timezone

    ds_hashes = dataset_hashes()
    envelope = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "git_revision": git_revision(),
        "config_hash": config_hash(config_payload),
        "dataset_hashes": ds_hashes,
        "model_identifier": model_identifier,
        "seed": seed,
        "status": status,
        "start_time": extra.pop("start_time", None) if extra else None,
        "end_time": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        envelope.update(extra)
    return envelope
