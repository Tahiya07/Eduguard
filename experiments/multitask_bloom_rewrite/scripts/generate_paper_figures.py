#!/usr/bin/env python
"""Publication figures/tables for the FINAL DEPLOYED MODEL only (code only).

Reads metrics from saved evaluation JSON — does not hard-code values.
Does NOT create baseline comparison / ablation figures.

Outputs under experiments/multitask_bloom_rewrite/paper_figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_meta(out_dir: Path, figure_id: str, meta: dict) -> None:
    (out_dir / f"{figure_id}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final-model paper figures/tables")
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to final deployed model metrics.json",
    )
    parser.add_argument(
        "--confusion",
        default=None,
        help="Optional confusion_matrix.json (default: sibling of metrics)",
    )
    parser.add_argument(
        "--failure-analysis",
        default=None,
        help="Optional failure_analysis.json",
    )
    parser.add_argument(
        "--human-results",
        default=None,
        help="Optional scored human eval JSON; skips Figure 7 if absent",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments/multitask_bloom_rewrite/paper_figures"),
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.is_absolute():
        metrics_path = REPO_ROOT / metrics_path
    metrics = load_json(metrics_path)
    conf_path = Path(args.confusion) if args.confusion else metrics_path.parent / "confusion_matrix.json"
    if not conf_path.is_absolute():
        conf_path = REPO_ROOT / conf_path
    fail_path = (
        Path(args.failure_analysis)
        if args.failure_analysis
        else metrics_path.parent / "failure_analysis.json"
    )
    if not fail_path.is_absolute():
        fail_path = REPO_ROOT / fail_path

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    ensure_out(out_dir)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "matplotlib/numpy required for figure generation. "
            f"Install in the GPU env before running. ({exc})"
        ) from exc

    bloom = metrics.get("bloom", {})
    per = bloom.get("classification", {}).get("per_level", {})
    rates = bloom.get("rates", {})
    model_id = metrics.get("model_id", "unknown")
    n_bloom = bloom.get("classification", {}).get("n")
    dataset_hash = (metrics.get("dataset") or {}).get("dataset_hash")

    # FIGURE 1 — per-level F1
    f1s = [float(per.get(lvl, {}).get("f1") or 0.0) for lvl in LEVELS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(LEVELS, f1s, color="#2F4B7C")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Bloom per-level F1 (final deployed model)")
    acc = bloom.get("classification", {}).get("accuracy")
    macro = bloom.get("classification", {}).get("macro_f1")
    ax.text(
        0.02,
        0.98,
        f"Accuracy={acc:.3f}\nMacro-F1={macro:.3f}" if isinstance(acc, float) else "",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"fig1_bloom_per_level_f1.{ext}", dpi=300)
    plt.close(fig)
    save_meta(
        out_dir,
        "fig1",
        {
            "figure_id": "fig1_bloom_per_level_f1",
            "data_source": str(metrics_path),
            "model": model_id,
            "sample_count": n_bloom,
            "metric": "per_level_f1",
            "dataset_hash": dataset_hash,
            "experiment": metrics.get("checkpoint"),
        },
    )

    # FIGURE 2 — confusion matrix
    if conf_path.is_file():
        cm = load_json(conf_path)
        mat = np.zeros((6, 6), dtype=float)
        for i, src in enumerate(LEVELS):
            row = cm.get(src, {})
            total = sum(float(row.get(t, 0) or 0) for t in LEVELS) or 1.0
            for j, tgt in enumerate(LEVELS):
                mat[i, j] = float(row.get(tgt, 0) or 0) / total
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(6), LEVELS, rotation=45, ha="right")
        ax.set_yticks(range(6), LEVELS)
        ax.set_xlabel("Predicted Bloom level")
        ax.set_ylabel("True Bloom level")
        ax.set_title("Bloom confusion matrix (row-normalized)")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{mat[i, j]*100:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out_dir / f"fig2_bloom_confusion_matrix.{ext}", dpi=300)
        plt.close(fig)
        save_meta(
            out_dir,
            "fig2",
            {
                "figure_id": "fig2_bloom_confusion_matrix",
                "data_source": str(conf_path),
                "model": model_id,
                "sample_count": n_bloom,
                "metric": "row_normalized_confusion",
                "dataset_hash": dataset_hash,
            },
        )

    # FIGURE 3 — rewrite validation rates
    keys = [
        "format_valid_rate",
        "semantic_valid_rate",
        "cognitive_valid_rate",
        "classifier_match_rate",
        "fully_validated_rewrite_rate",
    ]
    vals = [float(rates.get(k) or 0.0) for k in keys]
    labels = ["Format", "Semantic", "Cognitive", "Classifier match", "Fully validated"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, vals, color="#4C78A8")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Bloom rewrite validation rates")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"fig3_rewrite_validation_rates.{ext}", dpi=300)
    plt.close(fig)
    save_meta(
        out_dir,
        "fig3",
        {
            "figure_id": "fig3_rewrite_validation_rates",
            "data_source": str(metrics_path),
            "model": model_id,
            "sample_count": n_bloom,
            "metric": "rewrite_validation_rates",
            "dataset_hash": dataset_hash,
        },
    )

    # FIGURE 4 — mutually exclusive failure breakdown
    if fail_path.is_file():
        fails = load_json(fail_path).get("failure_counts", {})
        # Prefer exclusive categories if present; else use provided counts
        labels4 = list(fails.keys())
        vals4 = [float(fails[k]) for k in labels4]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(labels4, vals4, color="#F58518")
        ax.set_xlabel("Count")
        ax.set_title("Bloom output failure breakdown")
        fig.tight_layout()
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out_dir / f"fig4_failure_breakdown.{ext}", dpi=300)
        plt.close(fig)
        save_meta(
            out_dir,
            "fig4",
            {
                "figure_id": "fig4_failure_breakdown",
                "data_source": str(fail_path),
                "model": model_id,
                "sample_count": n_bloom,
                "metric": "failure_counts",
                "dataset_hash": dataset_hash,
            },
        )

    # FIGURE 5 — summarization
    summ = metrics.get("summarization", {})
    s_labels, s_vals = [], []
    for k, lab in (("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")):
        if summ.get(k) is not None:
            s_labels.append(lab)
            s_vals.append(float(summ[k]))
    bert = metrics.get("bertscore") or {}
    if bert.get("available") and bert.get("f1") is not None:
        s_labels.append("BERTScore-F1")
        s_vals.append(float(bert["f1"]))
    if s_labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(s_labels, s_vals, color="#54A24B")
        ax.set_ylim(0, max(1.0, max(s_vals) * 1.15))
        ax.set_ylabel("Score")
        ax.set_title("Summarization performance")
        fig.tight_layout()
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out_dir / f"fig5_summarization.{ext}", dpi=300)
        plt.close(fig)
        save_meta(
            out_dir,
            "fig5",
            {
                "figure_id": "fig5_summarization",
                "data_source": str(metrics_path),
                "model": model_id,
                "sample_count": summ.get("n"),
                "metric": "rouge_optional_bertscore",
                "dataset_hash": dataset_hash,
            },
        )

    # FIGURE 6 — deployment efficiency
    lat = (metrics.get("latency") or {}).get("per_example") or {}
    res = metrics.get("resources") or {}
    labels6, vals6, units = [], [], []
    if res.get("rss_mb") is not None:
        labels6.append("RSS")
        vals6.append(float(res["rss_mb"]))
        units.append("MB")
    if res.get("uss_mb") is not None:
        labels6.append("USS")
        vals6.append(float(res["uss_mb"]))
        units.append("MB")
    if lat.get("p50") is not None:
        labels6.append("p50 latency")
        vals6.append(float(lat["p50"]))
        units.append("s")
    if lat.get("p95") is not None:
        labels6.append("p95 latency")
        vals6.append(float(lat["p95"]))
        units.append("s")
    if labels6:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels6, vals6, color="#B279A2")
        ax.set_title("Deployment resource / latency measurements")
        ax.set_ylabel("Value (mixed units — see metadata)")
        fig.tight_layout()
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out_dir / f"fig6_deployment_efficiency.{ext}", dpi=300)
        plt.close(fig)
        save_meta(
            out_dir,
            "fig6",
            {
                "figure_id": "fig6_deployment_efficiency",
                "data_source": str(metrics_path),
                "model": model_id,
                "labels": labels6,
                "values": vals6,
                "units": units,
                "gpu_name": res.get("gpu_name"),
                "gpu_memory_allocated_mb": res.get("gpu_memory_allocated_mb"),
                "dataset_hash": dataset_hash,
            },
        )

    # FIGURE 7 — human eval (optional)
    if args.human_results:
        human_path = Path(args.human_results)
        if not human_path.is_absolute():
            human_path = REPO_ROOT / human_path
        if human_path.is_file():
            human = load_json(human_path)
            criteria = human.get("criteria_means") or human.get("means") or {}
            if criteria:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.bar(list(criteria.keys()), [float(v) for v in criteria.values()], color="#E45756")
                ax.set_title("Human evaluation (final deployed model)")
                ax.tick_params(axis="x", rotation=25)
                fig.tight_layout()
                for ext in ("pdf", "svg", "png"):
                    fig.savefig(out_dir / f"fig7_human_evaluation.{ext}", dpi=300)
                plt.close(fig)
                save_meta(
                    out_dir,
                    "fig7",
                    {
                        "figure_id": "fig7_human_evaluation",
                        "data_source": str(human_path),
                        "model": model_id,
                        "metric": "human_criteria",
                        "dataset_hash": dataset_hash,
                    },
                )

    # TABLES
    table1 = {
        "title": "Final Bloom per-level Precision / Recall / F1",
        "rows": [
            {
                "level": lvl,
                "precision": per.get(lvl, {}).get("precision"),
                "recall": per.get(lvl, {}).get("recall"),
                "f1": per.get(lvl, {}).get("f1"),
                "support": per.get(lvl, {}).get("support"),
            }
            for lvl in LEVELS
        ],
        "accuracy": bloom.get("classification", {}).get("accuracy"),
        "macro_f1": bloom.get("classification", {}).get("macro_f1"),
    }
    table2 = {"title": "Final rewrite-validation metrics", "rates": rates}
    table3 = {
        "title": "Final QA and summarization metrics",
        "qa": metrics.get("qa"),
        "summarization": metrics.get("summarization"),
        "bertscore": metrics.get("bertscore"),
    }
    table4 = {
        "title": "Final deployment resource metrics",
        "latency": metrics.get("latency"),
        "resources": metrics.get("resources"),
    }
    for name, payload in (
        ("table1_bloom_per_level", table1),
        ("table2_rewrite_validation", table2),
        ("table3_qa_summarization", table3),
        ("table4_deployment_resources", table4),
    ):
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_md = out_dir / "final_deployed_model_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Final deployed model — essential figures/tables",
                "",
                f"Model: `{model_id}`",
                f"Metrics source: `{metrics_path}`",
                f"Bloom N: {n_bloom}",
                f"Dataset hash: `{dataset_hash}`",
                "",
                "Figures: fig1–fig6 (fig7 only if human results provided).",
                "Tables: table1–table4 JSON sidecars.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("Wrote figures/tables to", out_dir)


if __name__ == "__main__":
    main()
