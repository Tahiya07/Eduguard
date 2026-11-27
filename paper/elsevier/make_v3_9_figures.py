"""Regenerate the v3.9 figures for the Elsevier manuscript from stored artifacts only.

Usage: .venv/Scripts/python.exe paper/elsevier/make_v3_9_figures.py
Writes PNG (300 dpi) + PDF into paper/elsevier/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "multitask_bloom_rewrite" / "results"
LORA = RES / "qwen15b_multitask_eduguard_v3_9_20261127"
OUT = Path(__file__).resolve().parent / "figures"

# validated default categorical order (slots 1-3), text inks, recessive grid
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Values copied from evaluation_report_extended.md of each run (validation split).
LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
BASE_ACC = [0.0164, 0.7119, 0.7692, 0.4038, 0.8246, 0.9833]
LORA_ACC = [0.8197, 0.9831, 0.8846, 0.9808, 1.0, 0.9]
REF_ACC = [0.4426, 0.8475, 0.9231, 0.7692, 0.9825, 0.85]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def bars(ax, groups, series, colors, labels, ylim=(0, 1.05)):
    n = len(series)
    w = 0.8 / n
    for i, (vals, c, lab) in enumerate(zip(series, colors, labels)):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        ax.bar(xs, vals, width=w - 0.03, color=c, label=lab, zorder=2)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)


def fig_levels():
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    bars(ax, LEVELS, [BASE_ACC, LORA_ACC, REF_ACC], [ORANGE, BLUE, AQUA],
         ["Base Qwen2.5-1.5B (prompt only)", "EduGuard v3.9 LoRA", "Reference rewrites (classifier ceiling)"])
    ax.set_ylabel("Classifier agrees with target level")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=8)
    save(fig, "fig09_v3_9_per_level_accuracy")


def fig_tasks():
    groups = ["Bloom target\naccuracy", "Bloom fully\nvalidated", "QA\nBERTScore F1", "QA\nbge-m3",
              "Summ.\nBERTScore F1", "Summ.\nbge-m3", "Summ.\nROUGE-Lsum"]
    base = [0.6158, 0.6012, 0.1276, 0.803, -0.0013, 0.7975, 0.2866]
    lora = [0.9267, 0.8035, 0.3434, 0.8202, 0.2676, 0.8351, 0.3571]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    bars(ax, groups, [base, lora], [ORANGE, BLUE], ["Base (prompt only)", "EduGuard v3.9 LoRA"], ylim=(-0.05, 1.05))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Score (validation)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8)
    ax.tick_params(axis="x", labelsize=7.5)
    save(fig, "fig10_v3_9_base_vs_lora")


def fig_loss():
    hist = json.loads((LORA / "trainer_log_history.json").read_text(encoding="utf-8"))["log_history"]
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    best = min(ev, key=lambda p: p[1])
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(*zip(*tr), color=BLUE, lw=2, label="Training loss")
    ax.plot(*zip(*ev), color=ORANGE, lw=2, marker="o", ms=4, label="Validation loss")
    ax.axvline(best[0], color=INK2, lw=0.8, ls="--")
    ax.annotate(f"best step {best[0]}\nval loss {best[1]:.4f}", (best[0], best[1]), xytext=(8, 18),
                textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Masked cross-entropy")
    ax.yaxis.grid(True, color=GRID)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig11_v3_9_training_curve")


if __name__ == "__main__":
    fig_levels()
    fig_tasks()
    fig_loss()
    print("wrote figures to", OUT)
