#!/usr/bin/env python
"""Paper-ready evaluation for the final deployable FL Bloom checkpoint.

Writes tables (.md/.tex/.csv) and figures (.png/.pdf) under artifacts/evaluation/paper/.
Selection uses validation accuracy; headline metrics are held-out test.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bloom_eval_metrics import evaluate_predictions  # noqa: E402
from predict_bloom import BLOOM_LABELS, QwenBloomPredictor, is_lora_adapter  # noqa: E402
from training.federated.best_checkpoint import pick_best_history_round  # noqa: E402

DEFAULT_OUT = ROOT / "artifacts/evaluation/paper"
SELECTION_JSON = ROOT / "artifacts/evaluation/best_fl_checkpoint_selection.json"
DEPLOYMENT_JSON = ROOT / "artifacts/evaluation/deployment_recommendation.json"
RESULT_FEDAVG = ROOT / "artifacts/federated/results/federated_lora_fedavg_iid_r20.json"
RESULT_FEDPROX = ROOT / "artifacts/federated/results/federated_lora_fedprox_iid_r20.json"
TEST_CSV = ROOT / "data/figshare_bloom_v1_test.csv"

MAIN_KEYS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("quadratic_weighted_kappa", "QWK"),
    ("within_one_level_accuracy", "Within-1"),
    ("severe_error_rate", "Severe err"),
    ("ece", "ECE"),
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_deployable(args: argparse.Namespace) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "model_dir": args.model_dir,
        "lora_dir": args.lora_dir,
        "experiment_id": None,
        "best_round": None,
        "selection_rule": "max validation accuracy; ties by macro_f1 then QWK",
    }
    if SELECTION_JSON.is_file():
        sel = _load(SELECTION_JSON)
        winner = sel.get("winner") or {}
        merge = sel.get("merge") or {}
        meta["experiment_id"] = winner.get("experiment_id")
        meta["best_round"] = winner.get("best_round")
        meta["best_val_metrics"] = winner.get("best_val_metrics")
        meta["best_test_metrics"] = winner.get("best_test_metrics")
        meta["selection_rule"] = sel.get("selection_rule", meta["selection_rule"])
        if not meta["model_dir"]:
            meta["model_dir"] = merge.get("merged_dir") or (sel.get("deployment") or {}).get("bloom_model_dir")
        if not meta["lora_dir"]:
            meta["lora_dir"] = winner.get("best_adapter_path")
    elif DEPLOYMENT_JSON.is_file():
        dep = _load(DEPLOYMENT_JSON)
        meta["experiment_id"] = dep.get("winner")
        meta["best_round"] = dep.get("best_round")
        meta["best_val_metrics"] = dep.get("best_val_metrics")
        meta["best_test_metrics"] = dep.get("best_test_metrics")
        if not meta["model_dir"]:
            meta["model_dir"] = dep.get("bloom_model_dir")
        if not meta["lora_dir"]:
            meta["lora_dir"] = dep.get("lora_dir")
    return meta


def _history_summary(path: Path, experiment_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = _load(path)
    history = data.get("history") or []
    best = data.get("best_checkpoint") or {}
    if not best.get("best_val_metrics"):
        picked = pick_best_history_round(history)
        if picked:
            best = {
                "best_round": picked.get("round"),
                "best_val_metrics": {
                    k: picked.get(k)
                    for k in ("accuracy", "macro_f1", "quadratic_weighted_kappa")
                    if picked.get(k) is not None
                },
            }
    final = history[-1] if history else {}
    return {
        "experiment_id": experiment_id,
        "history": history,
        "best_round": best.get("best_round"),
        "best_val": best.get("best_val_metrics") or {},
        "best_test": data.get("best_test_metrics") or {},
        "final_val": {
            "round": final.get("round"),
            "accuracy": final.get("accuracy"),
            "macro_f1": final.get("macro_f1"),
            "quadratic_weighted_kappa": final.get("quadratic_weighted_kappa"),
        },
        "final_test": data.get("final_test_metrics") or data.get("metrics") or {},
    }


def _run_live_eval(model_dir: Path, test_csv: Path, bootstrap: int) -> dict[str, Any]:
    df = pd.read_csv(test_csv).dropna()
    text_col = "question" if "question" in df.columns else df.columns[0]
    label_col = "bloom_level" if "bloom_level" in df.columns else df.columns[1]
    label2id = {label: i for i, label in enumerate(BLOOM_LABELS)}
    base = "Qwen/Qwen2.5-0.5B-Instruct" if is_lora_adapter(model_dir) else None
    predictor = QwenBloomPredictor(model_dir=str(model_dir), base_model=base, prefer_merged=True)
    rows = []
    y_true, y_pred, confidences = [], [], []
    for _, row in df.iterrows():
        text = str(row[text_col])
        gold = str(row[label_col])
        out = predictor.predict(text)
        pred = out["prediction"]
        rows.append({"question": text, "gold": gold, "prediction": pred, "confidence": out["confidence"]})
        y_true.append(label2id[gold])
        y_pred.append(label2id[pred])
        confidences.append(float(out["confidence"]))
        if len(rows) % 50 == 0:
            print(f"[paper-eval] {len(rows)}/{len(df)}")
    metrics = evaluate_predictions(
        y_true, y_pred, confidences=confidences, bootstrap_samples=bootstrap
    )
    from sklearn.metrics import classification_report, confusion_matrix

    return {
        "metrics": metrics,
        "rows": rows,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confidences,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=BLOOM_LABELS, digits=4, zero_division=0
        ),
        "n_test": len(df),
        "test_csv": str(test_csv),
        "model_dir": str(model_dir),
    }


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _write_table(path_stem: Path, headers: list[str], rows: list[list[str]], caption: str) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    # CSV
    pd.DataFrame(rows, columns=headers).to_csv(path_stem.with_suffix(".csv"), index=False)
    # Markdown
    md = [f"# {caption}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        md.append("| " + " | ".join(row) + " |")
    path_stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    # LaTeX
    cols = "l" + "r" * (len(headers) - 1)
    lines = [
        "% " + caption,
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path_stem.with_suffix(".tex").write_text("\n".join(lines), encoding="utf-8")


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _try_import_pyplot():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("[paper-eval] matplotlib not installed; skipping figure generation")
        return None


def _fig_confusion(cm: list[list[int]], title: str, path: Path) -> None:
    plt = _try_import_pyplot()
    if plt is None:
        return
    arr = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(BLOOM_LABELS)), BLOOM_LABELS, rotation=40, ha="right")
    ax.set_yticks(range(len(BLOOM_LABELS)), BLOOM_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    _savefig(fig, path)


def _fig_per_class_f1(per_class: dict, title: str, path: Path) -> None:
    plt = _try_import_pyplot()
    if plt is None:
        return
    labels = list(BLOOM_LABELS)
    vals = [float((per_class.get(l) or {}).get("f1") or 0.0) for l in labels]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, vals, color="#2c5f8a")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    _savefig(fig, path)


def _fig_reliability(calibration: dict, title: str, path: Path) -> None:
    plt = _try_import_pyplot()
    if plt is None:
        return
    bins = calibration.get("reliability_bins") or {}
    conf = bins.get("bin_confidence") or []
    acc = bins.get("bin_accuracy") or []
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#888888", label="Perfect")
    if conf and acc:
        ax.plot(conf, acc, "o-", color="#2c5f8a", label="Model")
    ece = calibration.get("ece")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title + (f" (ECE={ece:.3f})" if ece is not None else ""))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _savefig(fig, path)


def _fig_learning_curves(summaries: list[dict], path: Path) -> None:
    plt = _try_import_pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = {"fedavg_iid_r20": "#2c5f8a", "fedprox_iid_r20": "#8a4b2c"}
    for s in summaries:
        hist = s.get("history") or []
        if not hist:
            continue
        xs = [h.get("round") for h in hist if h.get("accuracy") is not None]
        ys = [h.get("accuracy") for h in hist if h.get("accuracy") is not None]
        eid = s["experiment_id"]
        ax.plot(xs, ys, "-o", markersize=3, label=eid, color=colors.get(eid))
        br = s.get("best_round")
        bv = (s.get("best_val") or {}).get("accuracy")
        if br is not None and bv is not None:
            ax.scatter([br], [bv], s=80, marker="*", zorder=5, color=colors.get(eid), edgecolors="black")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation learning curves (star = best-val round)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, path)


def _fig_best_vs_final(summaries: list[dict], path: Path) -> None:
    plt = _try_import_pyplot()
    if plt is None:
        return
    labels = []
    best_vals, final_vals, best_tests, final_tests = [], [], [], []
    for s in summaries:
        labels.append(s["experiment_id"].replace("_iid_r20", ""))
        best_vals.append(float((s.get("best_val") or {}).get("accuracy") or 0))
        final_vals.append(float((s.get("final_val") or {}).get("accuracy") or 0))
        best_tests.append(float((s.get("best_test") or {}).get("accuracy") or 0))
        final_tests.append(float((s.get("final_test") or {}).get("accuracy") or 0))
    x = np.arange(len(labels))
    w = 0.18
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - 1.5 * w, best_vals, w, label="Best val", color="#2c5f8a")
    ax.bar(x - 0.5 * w, final_vals, w, label="Final val", color="#7aa0c4")
    ax.bar(x + 0.5 * w, best_tests, w, label="Best test", color="#8a4b2c")
    ax.bar(x + 1.5 * w, final_tests, w, label="Final test", color="#c49a7a")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Best-val checkpoint vs final round")
    ax.legend(ncol=2)
    fig.tight_layout()
    _savefig(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper evaluation for deployable FL Bloom model.")
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--lora-dir", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--skip-live-eval", action="store_true")
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    meta = _resolve_deployable(args)
    summaries = [
        s
        for s in (
            _history_summary(RESULT_FEDAVG, "fedavg_iid_r20"),
            _history_summary(RESULT_FEDPROX, "fedprox_iid_r20"),
        )
        if s
    ]

    live = None
    model_path = Path(meta["model_dir"]) if meta.get("model_dir") else None
    lora_path = Path(meta["lora_dir"]) if meta.get("lora_dir") else None
    eval_path = None
    if not args.skip_live_eval:
        if model_path and (model_path / "config.json").is_file():
            eval_path = model_path
        elif lora_path and (lora_path / "adapter_config.json").is_file():
            eval_path = lora_path
        if eval_path is not None:
            print(f"[paper-eval] live evaluating {eval_path}")
            live = _run_live_eval(eval_path, args.test_csv, args.bootstrap_samples)
        else:
            print("[paper-eval] no deployable checkpoint on disk; writing curve/comparison tables only")

    # Prefer live metrics; else best_test from selection / FL JSON
    headline = None
    if live:
        headline = live["metrics"]
    elif meta.get("best_test_metrics"):
        headline = meta["best_test_metrics"]
    else:
        for s in summaries:
            if s.get("experiment_id") == meta.get("experiment_id") and s.get("best_test"):
                headline = s["best_test"]
                break

    # Table 1
    t1_rows = []
    if headline:
        boot = (headline.get("bootstrap") or {}) if isinstance(headline, dict) else {}
        for key, label in MAIN_KEYS:
            val = headline.get(key)
            ci = boot.get(key) or {}
            ci_s = (
                f"[{_fmt(ci.get('ci_low'))}, {_fmt(ci.get('ci_high'))}]"
                if ci.get("ci_low") is not None
                else "—"
            )
            t1_rows.append([label, _fmt(val), ci_s])
    else:
        t1_rows.append(["(no deployable test metrics yet)", "—", "—"])
    _write_table(
        out / "table1_main_metrics",
        ["Metric", "Value", "95% bootstrap CI"],
        t1_rows,
        "Main test metrics for deployable FL Bloom checkpoint",
    )

    # Table 2 algorithm comparison
    t2_rows = []
    for s in summaries:
        t2_rows.append(
            [
                s["experiment_id"],
                str(s.get("best_round") or "—"),
                _fmt((s.get("best_val") or {}).get("accuracy")),
                _fmt((s.get("final_val") or {}).get("accuracy")),
                _fmt((s.get("best_test") or {}).get("accuracy")),
                _fmt((s.get("final_test") or {}).get("accuracy")),
            ]
        )
    _write_table(
        out / "table2_algorithm_comparison",
        ["Experiment", "Best round", "Best val", "Final val", "Best test", "Final test"],
        t2_rows or [["—", "—", "—", "—", "—", "—"]],
        "FedAvg vs FedProx r20: best-val checkpoint vs final round",
    )

    # Table 3 per-class
    per_class = (headline or {}).get("per_class") or {}
    t3_rows = []
    for label in BLOOM_LABELS:
        stats = per_class.get(label) or {}
        t3_rows.append(
            [
                label,
                _fmt(stats.get("precision")),
                _fmt(stats.get("recall")),
                _fmt(stats.get("f1")),
                str(int(stats.get("support") or 0)) if stats else "—",
            ]
        )
    _write_table(
        out / "table3_per_class",
        ["Bloom level", "Precision", "Recall", "F1", "Support"],
        t3_rows,
        "Per-class test metrics for deployable FL model",
    )

    # Figures
    if live and live.get("confusion_matrix"):
        _fig_confusion(
            live["confusion_matrix"],
            f"Confusion matrix ({meta.get('experiment_id')} best r{meta.get('best_round')})",
            out / "fig_confusion_matrix",
        )
    if per_class:
        _fig_per_class_f1(per_class, "Per-class F1 (deployable model)", out / "fig_per_class_f1")
    calib = (headline or {}).get("calibration")
    if calib:
        _fig_reliability(calib, "Reliability diagram", out / "fig_reliability")
    if summaries:
        _fig_learning_curves(summaries, out / "fig_learning_curves")
        _fig_best_vs_final(summaries, out / "fig_best_vs_final")

    master = {
        "format": "paper_fl_deployable_results_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": meta.get("selection_rule"),
        "deployable": meta,
        "live_eval": {
            "ran": live is not None,
            "model_dir": str(eval_path) if eval_path else None,
            "n_test": (live or {}).get("n_test"),
            "metrics": headline,
            "confusion_matrix": (live or {}).get("confusion_matrix"),
        },
        "algorithm_summaries": [
            {
                "experiment_id": s["experiment_id"],
                "best_round": s.get("best_round"),
                "best_val": s.get("best_val"),
                "best_test": s.get("best_test"),
                "final_val": s.get("final_val"),
                "final_test": s.get("final_test"),
            }
            for s in summaries
        ],
        "artifacts": {
            "tables": [
                "table1_main_metrics.{md,tex,csv}",
                "table2_algorithm_comparison.{md,tex,csv}",
                "table3_per_class.{md,tex,csv}",
            ],
            "figures": [
                "fig_confusion_matrix.{png,pdf}",
                "fig_per_class_f1.{png,pdf}",
                "fig_reliability.{png,pdf}",
                "fig_learning_curves.{png,pdf}",
                "fig_best_vs_final.{png,pdf}",
            ],
        },
        "limitations": [
            "Checkpoint selected by validation accuracy; headline metrics are held-out test.",
            "Best-round adapters require --save-best-checkpoint re-runs; history-only peaks are not deployable.",
            "r20 vs 5-round comparisons are not budget-matched unless stated separately.",
        ],
    }
    (out / "paper_main_results.json").write_text(json.dumps(master, indent=2), encoding="utf-8")
    if live and live.get("rows"):
        pd.DataFrame(live["rows"]).to_csv(out / "deployable_test_predictions.csv", index=False)

    md_lines = [
        "# Paper results — deployable FL Bloom model",
        "",
        f"Generated: {master['timestamp_utc']}",
        "",
        f"- Winner: `{meta.get('experiment_id')}` best round `{meta.get('best_round')}`",
        f"- Selection: {meta.get('selection_rule')}",
        f"- Model dir: `{meta.get('model_dir')}`",
        f"- Live eval: {live is not None}",
        "",
        "## Headline test metrics",
        "",
    ]
    if headline:
        for key, label in MAIN_KEYS:
            md_lines.append(f"- {label}: {_fmt(headline.get(key))}")
    else:
        md_lines.append("- Not available yet (run after best-checkpoint merge).")
    md_lines.extend(
        [
            "",
            "## Insert into manuscript",
            "",
            f"- Tables: `{out}/table1_main_metrics.tex`, `table2_algorithm_comparison.tex`, `table3_per_class.tex`",
            f"- Figures: `{out}/fig_*.png` (also `.pdf`)",
            "",
            "## Honesty notes",
            "",
            "- Selection used **validation** accuracy; reported headline metrics are **held-out test**.",
            "- Best-round ≠ last-round when curves overshoot.",
            "- Do not claim deployable round-N weights without `_best` adapter on disk.",
            "",
        ]
    )
    (out / "PAPER_RESULTS.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[paper-eval] wrote {out}")
    print(f"[paper-eval] summary -> {out / 'PAPER_RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
