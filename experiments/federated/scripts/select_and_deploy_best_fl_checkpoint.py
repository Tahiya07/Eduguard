#!/usr/bin/env python
"""Select best FedAvg/FedProx r20 val checkpoint, merge, and write deployment recommendation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.federated.best_checkpoint import (  # noqa: E402
    best_adapter_dir,
    pick_best_history_round,
    select_best_across_runs,
)

RESULT_FEDAVG = ROOT / "artifacts/federated/results/federated_lora_fedavg_iid_r20.json"
RESULT_FEDPROX = ROOT / "artifacts/federated/results/federated_lora_fedprox_iid_r20.json"
LORA_FEDAVG = ROOT / "artifacts/federated/models/qwen_bloom_federated0.5B_fedavg_iid_r20"
LORA_FEDPROX = ROOT / "artifacts/federated/models/qwen_bloom_federated0.5B_fedprox_iid_r20"
OUT_DIR = ROOT / "artifacts/evaluation"
SELECTION_JSON = OUT_DIR / "best_fl_checkpoint_selection.json"
DEPLOYMENT_JSON = OUT_DIR / "deployment_recommendation.json"
GLOBAL_DIR = ROOT / "artifacts/federated/global"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_from_result(
    *,
    experiment_id: str,
    result_path: Path,
    lora_dir: Path,
) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    data = _load(result_path)
    best = data.get("best_checkpoint")
    if not best or best.get("best_val_metrics") is None:
        picked = pick_best_history_round(data.get("history") or [])
        if picked is None:
            return None
        best = {
            "best_round": int(picked.get("round") or 0),
            "best_val_metrics": {
                k: picked.get(k)
                for k in (
                    "accuracy",
                    "macro_f1",
                    "quadratic_weighted_kappa",
                    "within_one_level_accuracy",
                    "severe_error_rate",
                    "ece",
                    "n_eval",
                )
                if picked.get(k) is not None
            },
            "best_adapter_path": str(best_adapter_dir(lora_dir)),
            "recovered_from_history": True,
        }
    adapter = Path(best.get("best_adapter_path") or best_adapter_dir(lora_dir))
    final_row = None
    history = data.get("history") or []
    if history:
        final_row = history[-1]
    return {
        "experiment_id": experiment_id,
        "results_json": str(result_path),
        "lora_dir": str(lora_dir),
        "best_round": best.get("best_round"),
        "best_val_metrics": best.get("best_val_metrics"),
        "best_adapter_path": str(adapter),
        "best_adapter_exists": (adapter / "adapter_config.json").is_file(),
        "best_test_metrics": data.get("best_test_metrics"),
        "final_test_metrics": data.get("final_test_metrics") or data.get("metrics"),
        "final_round_val": {
            "round": final_row.get("round") if final_row else None,
            "accuracy": final_row.get("accuracy") if final_row else None,
            "macro_f1": final_row.get("macro_f1") if final_row else None,
            "quadratic_weighted_kappa": final_row.get("quadratic_weighted_kappa") if final_row else None,
        },
        "recovered_from_history": bool(best.get("recovered_from_history")),
    }


def _merge(lora_dir: Path, out_dir: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "training.centralized.merge_model",
        "--model-size",
        "0.5b",
        "--lora-dir",
        str(lora_dir),
        "--output-dir",
        str(out_dir),
        "--force",
    ]
    print(f"[select] merging: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    ok = proc.returncode == 0 and (out_dir / "config.json").is_file()
    return {
        "status": "MERGED" if ok else "FAILED",
        "lora_dir": str(lora_dir),
        "merged_dir": str(out_dir),
        "exit_code": proc.returncode,
        "command": cmd,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select and deploy best FL checkpoint across FedAvg/FedProx r20.")
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args()

    candidates = []
    for eid, result, lora in (
        ("fedavg_iid_r20", RESULT_FEDAVG, LORA_FEDAVG),
        ("fedprox_iid_r20", RESULT_FEDPROX, LORA_FEDPROX),
    ):
        c = _candidate_from_result(experiment_id=eid, result_path=result, lora_dir=lora)
        if c:
            candidates.append(c)

    if not candidates:
        print("No FedAvg/FedProx r20 result JSONs found.", file=sys.stderr)
        return 1

    winner = select_best_across_runs(candidates)
    if winner is None:
        print("Could not select a winner (missing val accuracy).", file=sys.stderr)
        return 1

    adapter = Path(winner["best_adapter_path"])
    round_n = int(winner.get("best_round") or 0)
    tag = winner["experiment_id"]
    merged_dir = GLOBAL_DIR / f"qwen_bloom_federated0.5B_{tag}_best_r{round_n}_merged"

    merge_info: dict[str, Any]
    if args.skip_merge:
        merge_info = {"status": "SKIPPED", "reason": "--skip-merge", "merged_dir": str(merged_dir)}
    elif not winner.get("best_adapter_exists"):
        merge_info = {
            "status": "BLOCKED",
            "reason": (
                f"Best adapter missing at {adapter}. "
                "Re-run simulation with --save-best-checkpoint --fresh to recover round weights."
            ),
            "merged_dir": str(merged_dir),
            "lora_dir": str(adapter),
        }
    else:
        merge_info = _merge(adapter, merged_dir)

    report = {
        "format": "best_fl_checkpoint_selection_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "max validation accuracy; ties by macro_f1 then quadratic_weighted_kappa",
        "candidates": candidates,
        "winner": {
            "experiment_id": winner["experiment_id"],
            "best_round": winner.get("best_round"),
            "best_val_metrics": winner.get("best_val_metrics"),
            "best_test_metrics": winner.get("best_test_metrics"),
            "best_adapter_path": winner.get("best_adapter_path"),
            "final_test_metrics": winner.get("final_test_metrics"),
        },
        "merge": merge_info,
        "deployment": {
            "bloom_model_dir": merge_info.get("merged_dir"),
            "start_offline_env": f'$env:BLOOM_MODEL_DIR = "{merge_info.get("merged_dir")}"',
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    DEPLOYMENT_JSON.write_text(
        json.dumps(
            {
                "winner": winner["experiment_id"],
                "best_round": winner.get("best_round"),
                "selection_rule": report["selection_rule"],
                "bloom_model_dir": merge_info.get("merged_dir"),
                "lora_dir": winner.get("best_adapter_path"),
                "merge": merge_info,
                "best_val_metrics": winner.get("best_val_metrics"),
                "best_test_metrics": winner.get("best_test_metrics"),
                "rationale": "Highest validation accuracy across FedAvg/FedProx r20 best checkpoints.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"[select] wrote {SELECTION_JSON}")
    print(f"[select] wrote {DEPLOYMENT_JSON}")
    if merge_info.get("status") == "BLOCKED":
        return 2
    if merge_info.get("status") == "FAILED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
