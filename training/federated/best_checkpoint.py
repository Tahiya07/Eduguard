"""Best FL checkpoint selection helpers (validation-accuracy based)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

METRIC_KEYS = ("accuracy", "macro_f1", "quadratic_weighted_kappa")


def metric_tuple(row: Mapping[str, Any]) -> Tuple[float, float, float]:
    return (
        float(row.get("accuracy") or 0.0),
        float(row.get("macro_f1") or 0.0),
        float(row.get("quadratic_weighted_kappa") or 0.0),
    )


def is_better_metrics(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    return metric_tuple(candidate) > metric_tuple(incumbent)


def pick_best_history_round(history: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the history row with best validation accuracy (tie-break F1, QWK)."""
    best: Optional[Dict[str, Any]] = None
    for row in history:
        if row.get("accuracy") is None:
            continue
        if is_better_metrics(row, best):
            best = dict(row)
    return best


def best_adapter_dir(global_dir: Path) -> Path:
    return Path(f"{global_dir}_best")


def copy_adapter_dir(src: Path, dst: Path) -> Path:
    """Replace dst with a full copy of src adapter directory."""
    if not (src / "adapter_config.json").is_file():
        raise FileNotFoundError(f"LoRA adapter missing at {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def maybe_save_best_checkpoint(
    *,
    global_dir: Path,
    round_idx: int,
    metrics: Mapping[str, Any],
    current_best: Optional[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    """If metrics beat current_best, copy global_dir to sibling `_best` and return updated record."""
    if not is_better_metrics(metrics, (current_best or {}).get("best_val_metrics")):
        return False, current_best or {}

    dest = best_adapter_dir(global_dir)
    copy_adapter_dir(global_dir, dest)
    record = {
        "best_round": int(round_idx),
        "best_val_metrics": {
            k: metrics.get(k)
            for k in (
                "accuracy",
                "macro_f1",
                "quadratic_weighted_kappa",
                "within_one_level_accuracy",
                "severe_error_rate",
                "ece",
                "n_eval",
            )
            if metrics.get(k) is not None
        },
        "best_adapter_path": str(dest),
    }
    return True, record


def select_best_across_runs(
    candidates: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pick the overall best among run summaries.

    Each candidate should include:
      experiment_id, best_round, best_val_metrics, best_adapter_path (optional),
      final_test_metrics (optional), results_json (optional).
    """
    winner: Optional[Dict[str, Any]] = None
    for c in candidates:
        metrics = c.get("best_val_metrics") or {}
        if metrics.get("accuracy") is None:
            continue
        if winner is None or is_better_metrics(metrics, winner.get("best_val_metrics")):
            winner = dict(c)
    return winner
