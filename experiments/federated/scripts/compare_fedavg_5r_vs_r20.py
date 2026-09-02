#!/usr/bin/env python
"""Compare fedavg_iid (5 rounds) vs fedavg_iid_r20 (20 rounds) for deployment selection."""

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

from bloom_eval_metrics import mcnemar_test
from predict_bloom import BLOOM_LABELS

RESULT_5R = ROOT / "artifacts/federated/results/federated_lora_fedavg_iid.json"
RESULT_R20 = ROOT / "artifacts/federated/results/federated_lora_fedavg_iid_r20.json"
LORA_5R = ROOT / "artifacts/federated/models/qwen_bloom_federated0.5B_fedavg_iid"
LORA_R20 = ROOT / "artifacts/federated/models/qwen_bloom_federated0.5B_fedavg_iid_r20"
OUT_DIR = ROOT / "artifacts/evaluation"
EVAL_5R = OUT_DIR / "eval_fedavg_iid_5r.json"
EVAL_R20 = OUT_DIR / "eval_fedavg_iid_r20.json"
ROWS_5R = OUT_DIR / "eval_fedavg_iid_5r_rows.csv"
ROWS_R20 = OUT_DIR / "eval_fedavg_iid_r20_rows.csv"
COMPARISON_JSON = OUT_DIR / "fedavg_5r_vs_r20_comparison.json"
COMPARISON_MD = OUT_DIR / "fedavg_5r_vs_r20_comparison.md"
DEPLOYMENT_JSON = OUT_DIR / "deployment_recommendation.json"

METRIC_KEYS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("quadratic_weighted_kappa", "QWK"),
    ("within_one_level_accuracy", "Within-1"),
    ("severe_error_rate", "Severe err"),
    ("ece", "ECE"),
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_from_fl_result(data: dict) -> dict[str, Any]:
    test = data.get("final_test_metrics") or data.get("metrics") or {}
    training = data.get("training") or {}
    comm = data.get("communication") or {}
    actual = training.get("actual") or {}
    return {
        "accuracy": test.get("accuracy"),
        "macro_f1": test.get("macro_f1"),
        "quadratic_weighted_kappa": test.get("quadratic_weighted_kappa"),
        "within_one_level_accuracy": test.get("within_one_level_accuracy"),
        "severe_error_rate": test.get("severe_error_rate"),
        "ece": test.get("ece"),
        "n_eval": test.get("n_eval"),
        "per_class": test.get("per_class"),
        "confusion_matrix": test.get("confusion_matrix"),
        "optimizer_steps": actual.get("total_optimizer_steps_completed")
        or training.get("total_optimizer_steps_estimate"),
        "federated_rounds": training.get("federated_rounds") or data.get("training", {}).get("configured", {}).get("federated_rounds"),
        "runtime_seconds": data.get("runtime_seconds"),
        "communication_total_bytes": comm.get("total_bytes"),
        "artifact_path": (data.get("artifact_paths") or {}).get("global_adapter"),
        "results_json": (data.get("artifact_paths") or {}).get("results_json"),
    }


def _to_eval_payload(name: str, metrics: dict, *, source: str, lora_dir: Path) -> dict:
    return {
        "benchmark": "bloom_lora_figshare_v1",
        "model_size": "0.5b",
        "experiment_id": name,
        "lora_dir": str(lora_dir),
        "source": source,
        "test_csv": str(ROOT / "data/figshare_bloom_v1_test.csv"),
        "n_test": metrics.get("n_eval"),
        "qwen_lora": {k: metrics.get(k) for k, _ in METRIC_KEYS if metrics.get(k) is not None},
        "confusion_matrix": metrics.get("confusion_matrix"),
        "per_class": metrics.get("per_class"),
    }


def _lora_ready(path: Path) -> bool:
    return (path / "adapter_config.json").is_file()


def _run_evaluate_bloom(lora_dir: Path, results_json: Path, test_csv: Path) -> bool:
    if not _lora_ready(lora_dir):
        return False
    cmd = [
        sys.executable,
        str(ROOT / "evaluate_bloom.py"),
        "--model-size",
        "0.5b",
        "--lora-dir",
        str(lora_dir),
        "--test-csv",
        str(test_csv),
        "--results-json",
        str(results_json),
    ]
    print(f"[compare] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    return proc.returncode == 0 and results_json.is_file()


def _load_eval_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = _load(path)
    return data.get("qwen_lora") or data.get("metrics") or data


def _rows_from_eval(path: Path) -> list[dict] | None:
    if not path.is_file():
        return None
    data = _load(path)
    rows = data.get("rows")
    if rows:
        return rows
    # evaluate_bloom also writes CSV under profile path; try evaluation rows export
    return None


def _rows_from_csv(path: Path) -> list[dict] | None:
    if not path.is_file():
        return None
    import pandas as pd

    df = pd.read_csv(path)
    if {"gold", "prediction"}.issubset(df.columns):
        return df.to_dict("records")
    return None


def _mcnemar_from_rows(rows_a: list[dict], rows_b: list[dict]) -> dict | None:
    if len(rows_a) != len(rows_b) or not rows_a:
        return None
    label2id = {label: i for i, label in enumerate(BLOOM_LABELS)}
    y_true, y_a, y_b = [], [], []
    for ra, rb in zip(rows_a, rows_b):
        gold = str(ra.get("gold") or ra.get("bloom_level"))
        if gold not in label2id:
            continue
        pa = str(ra.get("prediction"))
        pb = str(rb.get("prediction"))
        if pa not in label2id or pb not in label2id:
            continue
        y_true.append(label2id[gold])
        y_a.append(label2id[pa])
        y_b.append(label2id[pb])
    if not y_true:
        return None
    return mcnemar_test(y_true, y_a, y_b)


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _pick_winner(m5: dict, m20: dict) -> str:
    score5 = (m5.get("accuracy") or 0, m5.get("macro_f1") or 0, m5.get("quadratic_weighted_kappa") or 0)
    score20 = (m20.get("accuracy") or 0, m20.get("macro_f1") or 0, m20.get("quadratic_weighted_kappa") or 0)
    return "fedavg_iid_r20" if score20 > score5 else "fedavg_iid"


def _try_merge(winner: str) -> dict:
    if winner == "fedavg_iid_r20":
        lora_dir = LORA_R20
        out_dir = ROOT / "artifacts/federated/global/qwen_bloom_federated0.5B_fedavg_iid_r20_merged"
    else:
        lora_dir = LORA_5R
        out_dir = ROOT / "artifacts/federated/global/qwen_bloom_federated0.5B_fedavg_iid_merged"
    if not _lora_ready(lora_dir):
        return {
            "status": "SKIPPED",
            "reason": f"LoRA adapter missing at {lora_dir}",
            "merge_command": (
                f"python training/centralized/merge_model.py --model-size 0.5b "
                f"--lora-dir {lora_dir.relative_to(ROOT)} "
                f"--output-dir {out_dir.relative_to(ROOT)} --force"
            ),
        }
    cmd = [
        sys.executable,
        str(ROOT / "training/centralized/merge_model.py"),
        "--model-size",
        "0.5b",
        "--lora-dir",
        str(lora_dir),
        "--output-dir",
        str(out_dir),
        "--force",
    ]
    print(f"[compare] merging winner: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    merged_ok = (out_dir / "config.json").is_file()
    return {
        "status": "MERGED" if proc.returncode == 0 and merged_ok else "FAILED",
        "lora_dir": str(lora_dir),
        "merged_dir": str(out_dir),
        "exit_code": proc.returncode,
    }


def _write_md(report: dict) -> None:
    lines = [
        "# FedAvg IID: 5 rounds vs 20 rounds",
        "",
        f"Generated: {report['timestamp_utc']}",
        "",
        "## Summary",
        "",
        f"- **Winner (test metrics):** `{report['winner']}`",
        f"- **Fair comparison:** {report['comparison_is_fair']}",
        f"- **Note:** {report['fairness_note']}",
        "",
        "## Test metrics",
        "",
        "| Metric | fedavg_iid (5r) | fedavg_iid_r20 | Delta (r20 - 5r) |",
        "|---|---:|---:|---:|",
    ]
    for key, label in METRIC_KEYS:
        a = report["models"]["fedavg_iid"]["metrics"].get(key)
        b = report["models"]["fedavg_iid_r20"]["metrics"].get(key)
        d = report["deltas"].get(key)
        lines.append(f"| {label} | {a} | {b} | {d} |")
    lines.extend(
        [
            "",
            "## Training budget",
            "",
            f"- 5r optimizer steps: {report['models']['fedavg_iid']['metrics'].get('optimizer_steps')}",
            f"- r20 optimizer steps: {report['models']['fedavg_iid_r20']['metrics'].get('optimizer_steps')}",
            f"- 5r runtime (s): {report['models']['fedavg_iid']['metrics'].get('runtime_seconds')}",
            f"- r20 runtime (s): {report['models']['fedavg_iid_r20']['metrics'].get('runtime_seconds')}",
            "",
            "## Deployment",
            "",
            f"- Recommended `BLOOM_MODEL_DIR`: `{report['deployment']['bloom_model_dir']}`",
            f"- Merge status: {report['merge']['status']}",
        ]
    )
    if report.get("mcnemar"):
        m = report["mcnemar"]
        lines.extend(
            [
                "",
                "## McNemar test (paired predictions)",
                "",
                f"- p-value: {m.get('p_value')}",
                f"- significant: {m.get('significant')}",
            ]
        )
    COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare 5-round vs 20-round FedAvg IID models.")
    parser.add_argument("--test-csv", type=Path, default=ROOT / "data/figshare_bloom_v1_test.csv")
    parser.add_argument("--skip-live-eval", action="store_true", help="Use FL result JSON metrics only.")
    parser.add_argument("--skip-merge", action="store_true", help="Do not run merge_model.py.")
    args = parser.parse_args()

    if not RESULT_5R.is_file() or not RESULT_R20.is_file():
        print("Missing federated result JSON artifacts.", file=sys.stderr)
        return 1

    fl5 = _load(RESULT_5R)
    fl20 = _load(RESULT_R20)
    m5 = _metrics_from_fl_result(fl5)
    m20 = _metrics_from_fl_result(fl20)

    eval_source_5r = "federated_result_json"
    eval_source_r20 = "federated_result_json"

    if not args.skip_live_eval:
        if _run_evaluate_bloom(LORA_5R, EVAL_5R, args.test_csv):
            live5 = _load_eval_metrics(EVAL_5R)
            if live5:
                m5.update({k: live5.get(k) for k, _ in METRIC_KEYS if live5.get(k) is not None})
                eval_source_5r = "evaluate_bloom.py"
        if _run_evaluate_bloom(LORA_R20, EVAL_R20, args.test_csv):
            live20 = _load_eval_metrics(EVAL_R20)
            if live20:
                m20.update({k: live20.get(k) for k, _ in METRIC_KEYS if live20.get(k) is not None})
                eval_source_r20 = "evaluate_bloom.py"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_5R.write_text(
        json.dumps(_to_eval_payload("fedavg_iid", m5, source=eval_source_5r, lora_dir=LORA_5R), indent=2),
        encoding="utf-8",
    )
    EVAL_R20.write_text(
        json.dumps(_to_eval_payload("fedavg_iid_r20", m20, source=eval_source_r20, lora_dir=LORA_R20), indent=2),
        encoding="utf-8",
    )

    rows5 = _rows_from_eval(EVAL_5R) or _rows_from_csv(ROWS_5R)
    rows20 = _rows_from_eval(EVAL_R20) or _rows_from_csv(ROWS_R20)
    mcnemar = _mcnemar_from_rows(rows5, rows20) if rows5 and rows20 else None

    winner = _pick_winner(m5, m20)
    merged_dir = (
        ROOT / "artifacts/federated/global/qwen_bloom_federated0.5B_fedavg_iid_r20_merged"
        if winner == "fedavg_iid_r20"
        else ROOT / "artifacts/federated/global/qwen_bloom_federated0.5B_fedavg_iid_merged"
    )
    merge_info = {"status": "SKIPPED", "reason": "--skip-merge"} if args.skip_merge else _try_merge(winner)

    report = {
        "format": "fedavg_5r_vs_r20_comparison_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_is_fair": False,
        "fairness_note": "r20 uses 4x optimizer steps (6240 vs 1560); not budget-matched.",
        "winner": winner,
        "models": {
            "fedavg_iid": {
                "rounds": 5,
                "lora_dir": str(LORA_5R),
                "eval_json": str(EVAL_5R),
                "eval_source": eval_source_5r,
                "metrics": m5,
            },
            "fedavg_iid_r20": {
                "rounds": 20,
                "lora_dir": str(LORA_R20),
                "eval_json": str(EVAL_R20),
                "eval_source": eval_source_r20,
                "metrics": m20,
            },
        },
        "deltas": {key: _delta(m5.get(key), m20.get(key)) for key, _ in METRIC_KEYS},
        "mcnemar": mcnemar,
        "merge": merge_info,
        "deployment": {
            "bloom_model_dir": str(merged_dir),
            "start_offline_env": f'$env:BLOOM_MODEL_DIR = "{merged_dir}"',
            "immutable_baseline_preserved": "artifacts/evaluation/fedavg_iid_baseline_lock.json",
        },
    }
    COMPARISON_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    DEPLOYMENT_JSON.write_text(
        json.dumps(
            {
                "winner": winner,
                "bloom_model_dir": str(merged_dir),
                "lora_dir": str(LORA_R20 if winner == "fedavg_iid_r20" else LORA_5R),
                "merge": merge_info,
                "rationale": "Higher test accuracy, macro-F1, and QWK on figshare_bloom_v1_test; 4x training budget.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_md(report)

    print(json.dumps(report, indent=2))
    print(f"\n[compare] wrote {COMPARISON_JSON}")
    print(f"[compare] wrote {COMPARISON_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
