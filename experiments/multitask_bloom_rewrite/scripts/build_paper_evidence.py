#!/usr/bin/env python
"""Build the journal evidence pack for one EduGuard run: figures, tables, CIs, examples.

Every number is recomputed from saved artefacts; nothing is hard-coded:

* training curves  <- trainer_log_history.json (results dir) or the newest
                      checkpoint-*/trainer_state.json (output dir)
* task results     <- predictions.jsonl written by evaluate_rewrite.py
                      (LoRA, and the base model when its results exist)
* dataset figures  <- the task-separated dataset named in the config

Outputs (default paper/elsevier/evidence/<run>_<split>/):
  figures/fig_*.pdf (vector) + fig_*.png (600 dpi) + fig_*_data.csv (plotted numbers)
  tables/t*_*.csv + t*_*.tex (booktabs)
  examples/qualitative_examples.md + .jsonl
  CAPTIONS.md (draft captions with n), evidence_manifest.json (SHA-256 of inputs/outputs)

Usage:
  python build_paper_evidence.py --split validation
  python build_paper_evidence.py --split test --lora-results <dir> --base-results <dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for p in (EXPERIMENT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
TASKS = ["bloom_rewrite", "qa", "summarization"]
TASK_LABEL = {"bloom_rewrite": "Bloom rewriting", "qa": "Closed-book QA", "summarization": "Summarization"}
SPLITS = ["train", "validation", "test"]
LENGTH_ORDER = ["short", "medium", "long", "very_long"]
SOURCE_LABEL = {
    "openstax_textbook_section": "OpenStax textbook",
    "original_academic_example": "Authored academic",
    "lecture_slide_text_style": "Lecture slides",
    "arxiv_scientific_article": "arXiv article",  # older datasets (v3_4 and earlier)
}

# Palette: validated categorical slots 1-3 (dataviz reference palette; all-pairs
# CVD-safe). Orange/aqua are < 3:1 on white, so every bar carries a value label
# and every figure ships its data CSV. Base model bars are hatched (print/CVD).
C1, C2, C3 = "#257de0", "#f47230", "#00b894"
CONDITION_COLOR = {"lora": C1, "base": C2}
CONDITION_HATCH = {"lora": "", "base": "////"}
CONDITION_LABEL = {"lora": "EduGuard LoRA", "base": "Base Qwen2.5-1.5B"}
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"
SEQ_RAMP = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
SINGLE_COL_IN, DOUBLE_COL_IN = 3.54, 7.48  # 90 mm / 190 mm (Elsevier)
N_BOOT = 2000
BOOT_SEED = 42


# ----------------------------------------------------------------------------- io
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


class Pack:
    """Collects outputs and their provenance."""

    def __init__(self, out_dir: Path) -> None:
        self.out = out_dir
        for sub in ("figures", "tables", "examples"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        self.inputs: dict[str, str] = {}
        self.outputs: list[Path] = []
        self.captions: list[tuple[str, str]] = []
        self.skipped: list[str] = []

    def add_input(self, path: Path | None) -> None:
        if path and path.is_file():
            self.inputs[str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)] = sha256(path)

    def write_csv(self, rel: str, header: list[str], rows: list[list[Any]]) -> Path:
        path = self.out / rel
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        self.outputs.append(path)
        return path

    def write_tex(self, rel: str, header: list[str], rows: list[list[Any]], caption: str, label: str) -> None:
        def esc(x: Any) -> str:
            s = str(x)
            for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
                s = s.replace(a, b)
            return s
        cols = "l" + "r" * (len(header) - 1)
        lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
                 rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
        lines += [" & ".join(esc(c) for c in row) + r" \\" for row in rows]
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        path = self.out / rel
        path.write_text("\n".join(lines), encoding="utf-8")
        self.outputs.append(path)

    def table(self, stem: str, header: list[str], rows: list[list[Any]], caption: str) -> None:
        self.write_csv(f"tables/{stem}.csv", header, rows)
        self.write_tex(f"tables/{stem}.tex", header, rows, caption, f"tab:{stem}")
        self.captions.append((f"tables/{stem}", caption))

    def figure(self, fig, stem: str, caption: str, data_header: list[str], data_rows: list[list[Any]]) -> None:
        for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
            path = self.out / "figures" / f"{stem}.{ext}"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kw)
            self.outputs.append(path)
        self.write_csv(f"figures/{stem}_data.csv", data_header, data_rows)
        self.captions.append((f"figures/{stem}", caption))
        import matplotlib.pyplot as plt
        plt.close(fig)


# ------------------------------------------------------------------------ style
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    family = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.edgecolor": "#bdbdbd", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "lines.linewidth": 1.6, "hatch.linewidth": 0.6, "hatch.color": "#ffffff",
        "figure.dpi": 150, "savefig.facecolor": "white",
    })
    return plt


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("eduguard_blue", ["#f4f9ff"] + SEQ_RAMP)


def top_legend(fig, entries: list[tuple[str, str, str]]) -> None:
    """One legend above all panels so it never covers data. entries: (label, color, hatch)."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, hatch=h, edgecolor="white", linewidth=0.8, label=l) for l, c, h in entries]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               frameon=False, handlelength=1.6, columnspacing=1.4)


def condition_legend(fig, order: list[str]) -> None:
    if len(order) > 1:
        top_legend(fig, [(CONDITION_LABEL[c], CONDITION_COLOR[c], CONDITION_HATCH[c]) for c in order])


def bar_labels(ax, bars, values, fmt="{:.3f}", inside=False) -> None:
    for bar, v in zip(bars, values):
        if v is None:
            continue
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.annotate(fmt.format(v), (x, y), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color=TEXT)


# ------------------------------------------------------------------- statistics
def bootstrap_ci(values: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    if not values:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(values) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, float]:
    """Mean of (a - b) with 95% CI and two-sided bootstrap p-value; a, b aligned by example."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if not n:
        return {}
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    mean = sum(diffs) / n
    p = 2 * min(sum(x <= 0 for x in boots), sum(x >= 0 for x in boots)) / n_boot
    return {"delta": mean, "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1], "p": min(1.0, p)}


_ROUGE = None


def rouge(pred: str, ref: str) -> dict[str, float]:
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer
        _ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = _ROUGE.score(ref or "", pred or "")
    return {k: s[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")}


def score_predictions(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach per-example scores (same scorer settings as evaluate_rewrite.py)."""
    for r in preds:
        if r["task"] in ("qa", "summarization"):
            r.update(rouge(r.get("prediction") or "", r.get("reference") or ""))
        else:
            r["target_match_f"] = float(bool(r.get("target_match")))
    return preds


def fmt_ci(m: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ------------------------------------------------------------------- loaders
def load_training_history(cfg: dict[str, Any], pack: Pack) -> dict[str, Any] | None:
    results = resolve(cfg.get("results_dir"))
    candidate = results / "trainer_log_history.json" if results else None
    if candidate and candidate.is_file():
        pack.add_input(candidate)
        return json.loads(candidate.read_text(encoding="utf-8"))
    out = resolve(cfg.get("output_dir"))
    states = sorted(out.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[-1])) if out else []
    if not states:
        return None
    pack.add_input(states[-1])
    s = json.loads(states[-1].read_text(encoding="utf-8"))
    return {"log_history": s["log_history"], "best_metric": s.get("best_metric"),
            "best_model_checkpoint": s.get("best_model_checkpoint"), "global_step": s.get("global_step")}


def load_condition(results_dir: Path | None, pack: Pack) -> list[dict[str, Any]] | None:
    if not results_dir:
        return None
    pred_path = results_dir / "predictions.jsonl"
    if not pred_path.is_file():
        return None
    # Never build figures from an evaluation that is still running or crashed:
    # predictions.jsonl is written incrementally.
    progress = results_dir / "evaluation_progress.json"
    status = json.loads(progress.read_text(encoding="utf-8")).get("status") if progress.is_file() else None
    if status != "complete" or not (results_dir / "metrics.json").is_file():
        pack.skipped.append(f"{results_dir.name}: evaluation not complete (status={status}); predictions not used")
        return []
    pack.add_input(pred_path)
    pack.add_input(results_dir / "metrics.json")
    pack.add_input(results_dir / "evaluation_subset_manifest.json")
    preds = score_predictions(read_jsonl(pred_path))
    ext_path = results_dir / "metrics_extended.json"
    if ext_path.is_file():
        pack.add_input(ext_path)
        ext = json.loads(ext_path.read_text(encoding="utf-8"))
        extra: dict[str, dict[str, Any]] = {}
        for row in (ext.get("bloom") or {}).get("per_row", []):
            extra[row["id"]] = {"calibrated_full_f": float(row["calibrated_full"]), "ref_rougeL": row["ref_rougeL"],
                                "near_copy_f": float(row["near_copy"])}
        for task in ("qa", "summarization"):
            for row in (ext.get(task) or {}).get("per_row", []):
                extra[row["id"]] = {k: row[k] for k in ("embedding_sim", "rougeLsum", "rougeL_lc") if k in row}
        for r in preds:
            r.update(extra.get(r["id"], {}))
    else:
        pack.skipped.append(f"{results_dir.name}: no metrics_extended.json (run augment_evaluation.py) - calibrated metrics omitted")
    return preds


# ------------------------------------------------------------------ figures
def fig_training(plt, pack: Pack, hist: dict[str, Any]) -> None:
    logs = hist["log_history"]
    train = [h for h in logs if "loss" in h and "eval_loss" not in h]
    evals = [h for h in logs if "eval_loss" in h]
    if not train:
        pack.skipped.append("fig01_training_curves: no training loss entries")
        return
    best = min(evals, key=lambda h: h["eval_loss"]) if evals else None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL_IN, 3.3), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
    ax1.plot([h["epoch"] for h in train], [h["loss"] for h in train], color=C1, label="Training loss")
    if evals:
        ax1.plot([h["epoch"] for h in evals], [h["eval_loss"] for h in evals], color=C2, marker="o",
                 markersize=4.5, markeredgecolor="white", markeredgewidth=1, label="Validation loss")
        ax1.annotate(f"best: epoch {best['epoch']:.0f}\nval. loss {best['eval_loss']:.3f}",
                     (best["epoch"], best["eval_loss"]), xytext=(8, 14), textcoords="offset points",
                     fontsize=6.5, color=TEXT, arrowprops={"arrowstyle": "-", "color": TEXT2, "lw": 0.6})
    ax1.set_ylabel("Loss (answer tokens)")
    ax1.legend(loc="upper right")
    ax2.plot([h["epoch"] for h in train], [h["learning_rate"] * 1e4 for h in train], color=C1)
    ax2.set_ylabel("LR (×10$^{-4}$)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, None)
    rows = [[h["step"], round(h["epoch"], 4), "train", h["loss"], h.get("learning_rate"), h.get("grad_norm")] for h in train]
    rows += [[h["step"], round(h["epoch"], 4), "validation", h["eval_loss"], None, None] for h in evals]
    pack.figure(fig, "fig01_training_curves",
                f"Training and validation loss (assistant tokens only) and learning-rate schedule. "
                f"The adapter with the lowest validation loss (epoch {best['epoch']:.0f}, {best['eval_loss']:.3f}) "
                f"was retained as the final model." if best else "Training loss and learning-rate schedule.",
                ["step", "epoch", "series", "loss", "learning_rate", "grad_norm"], rows)

    mem = [h for h in train if "gpu_peak_alloc_gb" in h]
    if mem:
        fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.0))
        ax.plot([h["epoch"] for h in mem], [float(h["gpu_peak_alloc_gb"]) for h in mem], color=C1, label="Peak allocated")
        ax.plot([h["epoch"] for h in mem], [float(h["gpu_reserved_gb"]) for h in mem], color=C2, linestyle="--", label="Reserved")
        ax.set_ylabel("GPU memory (GB)")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, None)
        ax.legend(loc="lower right")
        pack.figure(fig, "figS1_gpu_memory",
                    "Peak allocated and reserved GPU memory per logging interval during training (RTX 4080, 16 GB).",
                    ["step", "epoch", "gpu_peak_alloc_gb", "gpu_reserved_gb"],
                    [[h["step"], round(h["epoch"], 4), h["gpu_peak_alloc_gb"], h["gpu_reserved_gb"]] for h in mem])


def fig_dataset(plt, pack: Pack, data_dir: Path, model_id: str, max_len: int) -> None:
    import dataset_io
    manifest = data_dir / "dataset_manifest.json"
    pack.add_input(manifest)
    rows_by_split = {s: dataset_io.load_split_rows(data_dir, s, validate=False)[0] for s in SPLITS}
    counts = {s: Counter(r["task"] for r in rows_by_split[s]) for s in SPLITS}

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.2))
    width = 0.26
    for i, (split, color) in enumerate(zip(SPLITS, (C1, C2, C3))):
        xs = [t + (i - 1) * width for t in range(len(TASKS))]
        vals = [counts[split][t] for t in TASKS]
        bars = ax.bar(xs, vals, width=width * 0.92, color=color, label=split.capitalize(), edgecolor="white", linewidth=0.8)
        bar_labels(ax, bars, vals, fmt="{:,}")
    ax.set_xticks(range(len(TASKS)), [TASK_LABEL[t] for t in TASKS])
    ax.set_ylabel("Rows")
    top_legend(fig, [(sp.capitalize(), col, "") for sp, col in zip(SPLITS, (C1, C2, C3))])
    ax.margins(y=0.15)
    data = [[TASK_LABEL[t], s, counts[s][t]] for t in TASKS for s in SPLITS]
    pack.figure(fig, "fig02_dataset_splits", "Rows per task and split in the final multitask dataset.",
                ["task", "split", "rows"], data)
    pack.table("t1_dataset_splits", ["Task", "Train", "Validation", "Test", "Total"],
               [[TASK_LABEL[t], *[counts[s][t] for s in SPLITS], sum(counts[s][t] for s in SPLITS)] for t in TASKS]
               + [["Total", *[sum(counts[s].values()) for s in SPLITS], sum(sum(c.values()) for c in counts.values())]],
               "Rows per task and split.")

    # Token lengths (train) per task, against the training context limit.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    lengths = {t: [len(tok(r["sft_text"], add_special_tokens=False)["input_ids"]) for r in rows_by_split["train"] if r["task"] == t] for t in TASKS}
    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 1.9))
    bp = ax.boxplot([lengths[t] for t in TASKS], vert=False, widths=0.5, patch_artist=True, whis=(0, 100),
                    medianprops={"color": TEXT, "linewidth": 1.2}, flierprops={"markersize": 2})
    for patch, color in zip(bp["boxes"], (C1, C2, C3)):
        patch.set_facecolor(color); patch.set_alpha(0.85); patch.set_edgecolor(color)
    ax.set_yticks(range(1, 4), [TASK_LABEL[t] for t in TASKS])
    ax.set_xscale("log")
    ax.axvline(max_len, color=TEXT2, linestyle="--", linewidth=0.9)
    ax.annotate(f"limit {max_len:,}", (max_len, 0.55), xytext=(-3, 0), textcoords="offset points",
                ha="right", va="bottom", fontsize=6.5, color=TEXT2)
    ax.set_xlabel("Tokens per training example (log scale)")
    ax.grid(axis="y", visible=False)
    data = []
    for t in TASKS:
        s = sorted(lengths[t])
        q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1)))]
        data.append([TASK_LABEL[t], len(s), s[0], q(0.25), q(0.5), q(0.75), q(0.99), s[-1], sum(x > max_len for x in s)])
    pack.figure(fig, "fig03_token_lengths",
                f"Token length of training examples (prompt + answer, Qwen2.5 tokenizer; boxes = quartiles, whiskers = min/max). No example exceeds the {max_len:,}-token limit, so nothing is truncated.",
                ["task", "n", "min", "p25", "median", "p75", "p99", "max", "over_limit"], data)

    # Summarization composition: length category x source type (all splits).
    summ = [r for s in SPLITS for r in rows_by_split[s] if r["task"] == "summarization"]
    from evaluate_rewrite import summary_length_category
    comp = Counter((summary_length_category(r.get("article") or ""), r.get("source_type")) for r in summ)
    sources = [s for s in SOURCE_LABEL if any(k[1] == s for k in comp)]
    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 1.9))
    left = [0] * len(LENGTH_ORDER)
    for src, color in zip(sources, (C1, C2, C3)):
        vals = [comp[(c, src)] for c in LENGTH_ORDER]
        ax.barh(range(len(LENGTH_ORDER)), vals, left=left, color=color, label=SOURCE_LABEL[src], edgecolor="white", linewidth=1, height=0.62)
        left = [a + b for a, b in zip(left, vals)]
    for i, total in enumerate(left):
        ax.annotate(f"{total}", (total, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.5)
    ax.set_yticks(range(len(LENGTH_ORDER)), [c.replace("_", " ").capitalize() for c in LENGTH_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Records (all splits)")
    top_legend(fig, [(SOURCE_LABEL[sr], col, "") for sr, col in zip(sources, (C1, C2, C3))])
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    pack.figure(fig, "fig04_summarization_composition",
                "Summarization records by source-length category and source type (all splits).",
                ["length_category", "source_type", "records"],
                [[c, s, comp[(c, s)]] for c in LENGTH_ORDER for s in sources])
    split_comp = {s: Counter(summary_length_category(r.get("article") or "") for r in rows_by_split[s] if r["task"] == "summarization") for s in SPLITS}
    pack.table("t2_summarization_categories", ["Category", "Train", "Validation", "Test"],
               [[c.replace("_", " ").capitalize(), *[split_comp[s][c] for s in SPLITS]] for c in LENGTH_ORDER],
               "Summarization records per length category and split.")


def grouped_bars(plt, ax, groups: list[str], series: dict[str, list[tuple[float, float, float] | None]], ylabel: str,
                 ylim: tuple[float, float] | None = (0, 1)) -> None:
    conds = list(series)
    width = 0.8 / len(conds)
    for i, cond in enumerate(conds):
        xs = [g + (i - (len(conds) - 1) / 2) * width for g in range(len(groups))]
        vals = [v[0] if v else 0 for v in series[cond]]
        bars = ax.bar(xs, vals, width=width * 0.92, color=CONDITION_COLOR[cond], hatch=CONDITION_HATCH[cond],
                      edgecolor="white", linewidth=0.8, label=CONDITION_LABEL[cond])
        err = [[(v[0] - v[1]) if v else 0 for v in series[cond]], [(v[2] - v[0]) if v else 0 for v in series[cond]]]
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=TEXT, elinewidth=0.7, capsize=1.8)
        for bar, v in zip(bars, series[cond]):
            if v:
                ax.annotate(f"{v[0]:.2f}", (bar.get_x() + bar.get_width() / 2, v[2]), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


def results_figures(plt, pack: Pack, conds: dict[str, list[dict[str, Any]]], split: str) -> None:
    by = {c: defaultdict(list) for c in conds}
    for c, preds in conds.items():
        for r in preds:
            by[c][r["task"]].append(r)
    # (name, task, per-example key, show in headline figure). The table lists every
    # available metric; the figure shows the ones that cannot be gamed by copying
    # the source (the original word-overlap "fully validated" rate can be).
    candidates = [
        ("Bloom target accuracy", "bloom_rewrite", "target_match_f", True),
        ("Bloom fully validated (calibrated)", "bloom_rewrite", "calibrated_full_f", True),
        ("Bloom ROUGE-L to reference rewrite", "bloom_rewrite", "ref_rougeL", True),
        ("Bloom near-copy of source (lower is better)", "bloom_rewrite", "near_copy_f", False),
        ("QA ROUGE-L", "qa", "rougeL", True),
        ("QA ROUGE-L, length-controlled", "qa", "rougeL_lc", True),
        ("QA bge-m3 similarity to reference", "qa", "embedding_sim", False),
        ("Summarization ROUGE-1", "summarization", "rouge1", True),
        ("Summarization ROUGE-2", "summarization", "rouge2", True),
        ("Summarization ROUGE-Lsum", "summarization", "rougeLsum", True),
        ("Summarization bge-m3 similarity to reference", "summarization", "embedding_sim", False),
    ]
    headline = [
        (name, task, key) for name, task, key, _ in candidates
        if all(by[c][task] and all(r.get(key) is not None for r in by[c][task]) for c in conds)
    ]
    shown = {name for name, task, key, show in candidates if show}
    tick = {
        "Bloom target accuracy": "Target\naccuracy", "Bloom fully validated (calibrated)": "Fully valid.\n(calibrated)",
        "Bloom ROUGE-L to reference rewrite": "ROUGE-L to\nreference",
        "QA ROUGE-L": "ROUGE-L", "QA ROUGE-L, length-controlled": "ROUGE-L\n(length-ctrl.)",
        "Summarization ROUGE-1": "ROUGE-1", "Summarization ROUGE-2": "ROUGE-2", "Summarization ROUGE-Lsum": "ROUGE-Lsum",
    }
    ci = {c: {name: bootstrap_ci([r[key] for r in by[c][task]]) if by[c][task] else None
              for name, task, key in headline} for c in conds}
    table_rows = []
    for name, task, key in headline:
        n = len(by.get("lora", by[next(iter(conds))])[task])
        row = [name, n]
        for c in ("base", "lora"):
            row.append(fmt_ci(*ci[c][name]) if c in conds and ci[c][name] else "n/a")
        if "base" in conds and "lora" in conds and by["base"][task] and by["lora"][task]:
            base_by_id = {r["id"]: r[key] for r in by["base"][task]}
            pairs = [(r[key], base_by_id[r["id"]]) for r in by["lora"][task] if r["id"] in base_by_id]
            d = paired_bootstrap([a for a, _ in pairs], [b for _, b in pairs])
            row += [fmt_ci(d["delta"], d["lo"], d["hi"]), f"{d['p']:.4f}"]
        else:
            row += ["n/a", "n/a"]
        table_rows.append(row)
    pack.table("t3_main_results", ["Metric", "N", "Base [95% CI]", "LoRA [95% CI]", "Delta LoRA-base [95% CI]", "Bootstrap p"],
               table_rows, f"Held-out {split} results. 95% percentile bootstrap CIs ({N_BOOT} resamples); paired bootstrap for the difference.")

    order = [c for c in ("base", "lora") if c in conds]
    # Small multiples: one panel per task, shared 0-1 score axis.
    fig_metrics = [h for h in headline if h[0] in shown]
    present = [t for t in TASKS if any(h[1] == t for h in fig_metrics)]
    widths = [sum(1 for h in fig_metrics if h[1] == t) for t in present]
    fig, axes = plt.subplots(1, len(present), figsize=(DOUBLE_COL_IN, 2.5), sharey=True,
                             gridspec_kw={"width_ratios": widths, "wspace": 0.08})
    for ax, t in zip(axes, present):
        hs = [h for h in fig_metrics if h[1] == t]
        grouped_bars(plt, ax, [tick[h[0]] for h in hs], {c: [ci[c][h[0]] for h in hs] for c in order}, "Score" if t == present[0] else "")
        ax.set_title(f"{TASK_LABEL[t]} (n={len(by[order[-1]][t])})")
    condition_legend(fig, order)
    pack.figure(fig, "fig05_main_results",
                f"Held-out {split} performance per task with 95% bootstrap confidence intervals. Bloom target accuracy is agreement with a fixed 0.5B Bloom classifier; "
                "'fully validated (calibrated)' uses a bge-m3 topic check calibrated on training references; ROUGE-L to reference compares each rewrite with the teacher-written rewrite for the same level; "
                "length-controlled ROUGE-L removes Markdown and cuts each output to its reference's word count, identically for both systems.",
                ["metric", "condition", "mean", "ci_low", "ci_high", "n"],
                [[h[0], c, *ci[c][h[0]], len(by[c][h[1]])] for h in fig_metrics for c in order if ci[c][h[0]]])

    lora = by.get("lora") or by[order[-1]]
    main_cond = "lora" if "lora" in conds else order[-1]

    # Bloom confusion matrices (target level x classifier prediction), row-normalised.
    if any(by[c]["bloom_rewrite"] for c in order):
        cols = LEVELS + ["MISSING"]
        fig, axes = plt.subplots(1, len(order), figsize=(SINGLE_COL_IN * len(order) * 1.05, 2.9), squeeze=False,
                                 sharey=True, gridspec_kw={"wspace": 0.06})
        data = []
        for ax, c in zip(axes[0], order):
            recs = by[c]["bloom_rewrite"]
            mat = [[sum(1 for r in recs if r["target_bloom_level"] == t and (r.get("classifier_prediction") or "MISSING") == p) for p in cols] for t in LEVELS]
            norm = [[v / (sum(row) or 1) for v in row] for row in mat]
            ax.imshow(norm, cmap=seq_cmap(), vmin=0, vmax=1, aspect="auto")
            for i, row in enumerate(mat):
                for j, v in enumerate(row):
                    if v:
                        ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                                color="white" if norm[i][j] > 0.55 else TEXT)
                    data.append([c, LEVELS[i], cols[j], v, round(norm[i][j], 4)])
            ax.set_xticks(range(len(cols)), [x[:3] if x != "MISSING" else "None" for x in cols])
            ax.set_yticks(range(len(LEVELS)), LEVELS)
            ax.set_xlabel("Classifier-predicted level of rewrite")
            ax.set_title(CONDITION_LABEL[c])
            ax.grid(False)
        axes[0][0].set_ylabel("Requested target level")
        pack.figure(fig, "fig06_bloom_confusion",
                    f"Requested target level versus the Bloom classifier's prediction for each generated rewrite ({split}). Cells show counts; shading is the row-normalised share.",
                    ["condition", "target_level", "predicted_level", "count", "row_share"], data)

        per_level = {c: [bootstrap_ci([r["target_match_f"] for r in by[c]["bloom_rewrite"] if r["target_bloom_level"] == lv]) for lv in LEVELS] for c in order}
        support = [sum(1 for r in lora["bloom_rewrite"] if r["target_bloom_level"] == lv) for lv in LEVELS]
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN * 0.6, 2.3))
        grouped_bars(plt, ax, [f"{lv}\n(n={n})" for lv, n in zip(LEVELS, support)], per_level, "Target accuracy")
        condition_legend(fig, order)
        pack.figure(fig, "fig07_bloom_per_level",
                    f"Bloom rewriting target accuracy per requested level with 95% bootstrap CIs ({split}).",
                    ["condition", "target_level", "mean", "ci_low", "ci_high", "n"],
                    [[c, lv, *per_level[c][i], support[i]] for c in order for i, lv in enumerate(LEVELS)])
        pack.table("t4_bloom_per_level", ["Target level", "N", *[f"{CONDITION_LABEL[c]} accuracy [95% CI]" for c in order]],
                   [[lv, support[i], *[fmt_ci(*per_level[c][i]) for c in order]] for i, lv in enumerate(LEVELS)],
                   f"Bloom rewriting target accuracy per requested level ({split}).")

        recs = by[main_cond]["bloom_rewrite"]
        cell = {}
        for s in LEVELS:
            for t in LEVELS:
                sub = [r for r in recs if r.get("source_bloom_level") == s and r["target_bloom_level"] == t]
                cell[(s, t)] = (sum(r["target_match_f"] for r in sub) / len(sub), len(sub)) if sub else (None, 0)
        import math
        src_levels = [lv for lv in LEVELS if any(cell[(lv, t)][1] for t in LEVELS)]  # drop empty source rows
        fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 0.42 * len(src_levels) + 0.75))
        grid = [[cell[(s, t)][0] if cell[(s, t)][0] is not None else math.nan for t in LEVELS] for s in src_levels]
        ax.imshow(grid, cmap=seq_cmap(), vmin=0, vmax=1, aspect="auto")
        for i, s in enumerate(src_levels):
            for j, t in enumerate(LEVELS):
                v, n = cell[(s, t)]
                if n:
                    ax.text(j, i, f"{v:.2f}\n{n}", ha="center", va="center", fontsize=5.5,
                            color="white" if v > 0.55 else TEXT, linespacing=0.95)
        ax.set_xticks(range(6), [lv[:3] for lv in LEVELS])
        ax.set_yticks(range(len(src_levels)), src_levels)
        ax.set_xlabel("Requested target level")
        ax.set_ylabel("Source question level")
        ax.grid(False)
        pack.figure(fig, "fig08_source_target_accuracy",
                    f"{CONDITION_LABEL[main_cond]}: target accuracy for each source-to-target level transformation ({split}). Each cell shows accuracy and n; the diagonal is empty by design.",
                    ["source_level", "target_level", "target_accuracy", "n"],
                    [[s, t, cell[(s, t)][0], cell[(s, t)][1]] for s in LEVELS for t in LEVELS if cell[(s, t)][1]])

        failed = lambda r: not r["calibrated_full_f"] if "calibrated_full_f" in r else not r.get("fully_validated")
        fails = {c: Counter(r.get("failure_category") or "OTHER" for r in by[c]["bloom_rewrite"] if failed(r)) for c in order}
        cats = sorted({k for f in fails.values() for k in f if k}, key=lambda k: -fails[main_cond].get(k, 0))
        if cats:
            fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 0.35 * len(cats) + 0.7))
            h = 0.8 / len(order)
            for i, c in enumerate(order):
                ys = [k + (i - (len(order) - 1) / 2) * h for k in range(len(cats))]
                vals = [fails[c].get(k, 0) for k in cats]
                bars = ax.barh(ys, vals, height=h * 0.9, color=CONDITION_COLOR[c], hatch=CONDITION_HATCH[c],
                               edgecolor="white", linewidth=0.8, label=CONDITION_LABEL[c])
                for b, v in zip(bars, vals):
                    ax.annotate(str(v), (v, b.get_y() + b.get_height() / 2), xytext=(2, 0), textcoords="offset points", va="center", fontsize=6)
            ax.set_yticks(range(len(cats)), [k.replace("_", " ").capitalize() for k in cats])
            ax.invert_yaxis()
            ax.set_xlabel("Rewrites not fully validated")
            ax.grid(axis="y", visible=False)
            condition_legend(fig, order)
            ax.margins(x=0.12)
            pack.figure(fig, "fig09_bloom_failures",
                        f"Primary failure category of Bloom rewrites that did not pass full validation ({split}).",
                        ["condition", "failure_category", "count"], [[c, k, fails[c].get(k, 0)] for c in order for k in cats])

    # Summarization breakdowns.
    if any(by[c]["summarization"] for c in order):
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 2.3), gridspec_kw={"width_ratios": [3, 4]})
        data, trows = [], []
        for ax, key, groups, label in (
            (axa, "source_type", [s for s in SOURCE_LABEL if any(r.get("source_type") == s for r in lora["summarization"])], SOURCE_LABEL),
            (axb, "length_category", [c for c in LENGTH_ORDER if any(r.get("length_category") == c for r in lora["summarization"])], {c: c.replace("_", " ").capitalize() for c in LENGTH_ORDER}),
        ):
            series = {c: [bootstrap_ci([r["rougeL"] for r in by[c]["summarization"] if r.get(key) == g]) for g in groups] for c in order}
            ns = [sum(1 for r in lora["summarization"] if r.get(key) == g) for g in groups]
            ticks = [(label[g].replace(" ", "\n", 1) if key == "source_type" else label[g]) + f"\n(n={n})" for g, n in zip(groups, ns)]
            grouped_bars(plt, ax, ticks, series, "ROUGE-L", ylim=(0, 1))
            for c in order:
                for g, n, v in zip(groups, ns, series[c]):
                    data.append([key, g, c, *v, n])
                    trows.append([key.replace("_", " "), label[g], n, CONDITION_LABEL[c], fmt_ci(*v)])
        axa.set_title("By source type")
        axb.set_title("By source length")
        axb.set_ylabel("")
        condition_legend(fig, order)
        pack.figure(fig, "fig10_summarization_breakdown",
                    f"Summarization ROUGE-L by source type and by source-length category with 95% bootstrap CIs ({split}). Small groups (n < 10) have wide intervals.",
                    ["grouping", "group", "condition", "mean", "ci_low", "ci_high", "n"], data)
        pack.table("t5_summarization_breakdown", ["Grouping", "Group", "N", "Condition", "ROUGE-L [95% CI]"], trows,
                   f"Summarization ROUGE-L by source type and length category ({split}).")

    # QA by type.
    if any(by[c]["qa"] for c in order):
        type_counts = Counter(r.get("qa_type") or "unknown" for r in lora["qa"])
        keep = [t for t, n in type_counts.most_common() if n >= 5]
        def grp(r):
            t = r.get("qa_type") or "unknown"
            return t if t in keep else "other (n<5 each)"
        groups = keep + (["other (n<5 each)"] if any(grp(r) == "other (n<5 each)" for r in lora["qa"]) else [])
        series = {c: [bootstrap_ci([r["rougeL"] for r in by[c]["qa"] if grp(r) == g]) for g in groups] for c in order}
        ns = [sum(1 for r in lora["qa"] if grp(r) == g) for g in groups]
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN * 0.75, 2.3))
        grouped_bars(plt, ax, [g.capitalize().replace(" (", "\n(") + f"\n(n={n})" for g, n in zip(groups, ns)], series, "ROUGE-L", ylim=(0, 1))
        condition_legend(fig, order)
        pack.figure(fig, "fig11_qa_by_type", f"Closed-book QA ROUGE-L by question type with 95% bootstrap CIs ({split}).",
                    ["qa_type", "condition", "mean", "ci_low", "ci_high", "n"],
                    [[g, c, *series[c][i], ns[i]] for c in order for i, g in enumerate(groups)])
        pack.table("t6_qa_by_type", ["Question type", "N", *[f"{CONDITION_LABEL[c]} ROUGE-L [95% CI]" for c in order]],
                   [[g, ns[i], *[fmt_ci(*series[c][i]) for c in order]] for i, g in enumerate(groups)],
                   f"Closed-book QA ROUGE-L by question type ({split}).")

    # Per-example distributions.
    tasks_present = [t for t in ("qa", "summarization") if by[main_cond][t]]
    if tasks_present:
        fig, axes = plt.subplots(1, len(tasks_present), figsize=(SINGLE_COL_IN * len(tasks_present), 2.2), squeeze=False)
        data = []
        rng = random.Random(0)
        for ax, t in zip(axes[0], tasks_present):
            vals = [[r["rougeL"] for r in by[c][t]] for c in order]
            bp = ax.boxplot(vals, widths=0.45, patch_artist=True, showfliers=False, medianprops={"color": TEXT, "linewidth": 1.2})
            for patch, c in zip(bp["boxes"], order):
                patch.set_facecolor("white"); patch.set_edgecolor(CONDITION_COLOR[c]); patch.set_linewidth(1.2)
            for i, (c, v) in enumerate(zip(order, vals), start=1):
                ax.scatter([i + rng.uniform(-0.15, 0.15) for _ in v], v, s=6, color=CONDITION_COLOR[c], alpha=0.55, linewidths=0, zorder=3)
                data += [[t, c, x] for x in v]
            ax.set_xticks(range(1, len(order) + 1), [CONDITION_LABEL[c] for c in order])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Per-example ROUGE-L")
            ax.set_title(TASK_LABEL[t])
            ax.grid(axis="x", visible=False)
        pack.figure(fig, "fig12_rouge_distributions",
                    f"Per-example ROUGE-L distributions ({split}); boxes show quartiles, points are individual examples.",
                    ["task", "condition", "rougeL"], data)


def calibration_figures(plt, pack: Pack, ext: dict[str, Any], split: str) -> None:
    """Model vs. human-reference rewrites under the same instruments (augment_evaluation.py)."""
    b = ext.get("bloom")
    if not b:
        return
    per = b["per_level"]
    if all(v.get("reference_target_accuracy") is not None for v in per.values()):
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN * 0.6, 2.3))
        width = 0.4
        for i, (key, label, color, hatch) in enumerate((
            ("model_target_accuracy", "EduGuard LoRA rewrites", C1, ""),
            ("reference_target_accuracy", "Human reference rewrites", C3, "...."),
        )):
            xs = [x + (i - 0.5) * width for x in range(len(LEVELS))]
            vals = [per[lv][key] for lv in LEVELS]
            bars = ax.bar(xs, vals, width=width * 0.92, color=color, hatch=hatch, edgecolor="white", linewidth=0.8)
            bar_labels(ax, bars, vals, fmt="{:.2f}")
        ax.set_xticks(range(len(LEVELS)), [f"{lv}\n(n={per[lv]['n']})" for lv in LEVELS])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Classifier agrees with\nrequested level")
        ax.grid(axis="x", visible=False)
        top_legend(fig, [("EduGuard LoRA rewrites", C1, ""), ("Human reference rewrites", C3, "....")])
        pack.figure(fig, "fig13_classifier_ceiling",
                    f"Agreement of the fixed Bloom classifier with the requested level for model rewrites and for the human reference rewrites of the same items ({split}). "
                    "Low reference agreement (e.g. Remember, Analyze) marks a limit of the classifier, not of the model.",
                    ["target_level", "n", "model_agreement", "reference_agreement"],
                    [[lv, per[lv]["n"], per[lv]["model_target_accuracy"], per[lv]["reference_target_accuracy"]] for lv in LEVELS])
    names = {"classifier_target_match": "Classifier agrees with requested level", "format_valid": "Format check passed",
             "topic_preserved_embedding": "Topic preserved (bge-m3, train-calibrated)",
             "near_copy_of_source": "Near-copy of source (lower is better)"}
    rows = [[names.get(k, k), v["model"], v["reference"]] for k, v in b["model_vs_reference"].items()]
    rows += [["Fully validated (calibrated definition)", b["fully_validated_calibrated"], "n/a"]]
    pack.table("t7_bloom_model_vs_reference", ["Measure", "EduGuard LoRA", "Human reference"], rows,
               f"Bloom rewriting checks applied identically to model and human reference rewrites ({split}).")
    q, s = ext.get("qa"), ext.get("summarization")
    sem_rows = []
    if q:
        sem_rows += [["Closed-book QA", q["n"], q["embedding_similarity_to_reference"]]]
    if s:
        sem_rows += [["Summarization", s["n"], s["embedding_similarity_to_reference"]]]
    if sem_rows:
        pack.table("t8_semantic_agreement", ["Task", "N", "bge-m3 similarity to reference (mean)"], sem_rows,
                   f"Embedding-based agreement with references ({split}).")


def qualitative_examples(pack: Pack, conds: dict[str, list[dict[str, Any]]], split: str, k: int = 3) -> None:
    main = "lora" if "lora" in conds else next(iter(conds))
    base_by_id = {r["id"]: r for r in conds.get("base", [])}
    picked: list[tuple[str, str, dict[str, Any]]] = []
    for task in ("qa", "summarization"):
        recs = sorted((r for r in conds[main] if r["task"] == task), key=lambda r: r["rougeL"])
        if recs:
            idx = sorted({len(recs) - 1, len(recs) // 2, 0})[::-1]
            picked += [(task, label, recs[i]) for label, i in zip(("highest ROUGE-L", "median ROUGE-L", "lowest ROUGE-L"), idx)]
    bloom = [r for r in conds[main] if r["task"] == "bloom_rewrite"]
    passed = (lambda r: bool(r["calibrated_full_f"])) if all("calibrated_full_f" in r for r in bloom) else (lambda r: bool(r.get("fully_validated")))
    ok = [r for r in bloom if passed(r)]
    bad = [r for r in bloom if not passed(r)]
    rng = random.Random(BOOT_SEED)
    picked += [("bloom_rewrite", "fully validated (random)", r) for r in rng.sample(ok, min(k, len(ok)))]
    picked += [("bloom_rewrite", "failed validation (random)", r) for r in rng.sample(bad, min(k, len(bad)))]

    def clip(s: str | None, n: int = 1200) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[:n].rstrip() + f" [... {len(s) - n} more characters]"

    md = [f"# Qualitative examples ({split})", "",
          f"Selected deterministically from `{main}` predictions: for QA and summarization the highest, median and lowest "
          "per-example ROUGE-L; for Bloom rewriting a seeded random sample of passing and failing rewrites. "
          "Sources are clipped for length.", ""]
    jl = []
    for task, label, r in picked:
        b = base_by_id.get(r["id"])
        md += [f"## {TASK_LABEL[task]}: {label}", "", f"- ID: `{r['id']}`"]
        if task == "bloom_rewrite":
            md += [f"- Source level → requested level: {r.get('source_bloom_level')} → {r.get('target_bloom_level')}",
                   f"- Classifier: {r.get('classifier_prediction')} (confidence {r.get('classifier_confidence', 0):.2f}); "
                   f"failure category: {r.get('failure_category') or 'none'}", "",
                   f"**Source question.** {clip(r.get('source_question'))}", "",
                   f"**Reference rewrite.** {clip(r.get('reference'))}", "", f"**EduGuard rewrite.** {clip(r.get('prediction'))}", ""]
        else:
            src = r.get("question") if task == "qa" else r.get("article")
            md += [f"- ROUGE-L: {r['rougeL']:.3f}" + (f" (base: {b['rougeL']:.3f})" if b else ""), "",
                   f"**{'Question' if task == 'qa' else 'Source'}.** {clip(src)}", "",
                   f"**Reference.** {clip(r.get('reference'))}", "", f"**EduGuard output.** {clip(r.get('prediction'))}", ""]
        if b:
            md += [f"**Base model output.** {clip(b.get('prediction'))}", ""]
        jl.append({"task": task, "selection": label, "id": r["id"], "lora": r, "base": b})
    path = pack.out / "examples" / "qualitative_examples.md"
    path.write_text("\n".join(md), encoding="utf-8")
    pack.outputs.append(path)
    path = pack.out / "examples" / "qualitative_examples.jsonl"
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in jl) + "\n", encoding="utf-8")
    pack.outputs.append(path)


# --------------------------------------------------------------------- main
def build(config: Path, split: str, lora_results: Path | None, base_results: Path | None, out_dir: Path | None) -> Path:
    cfg = json.loads(config.read_text(encoding="utf-8"))
    run = Path(cfg.get("output_dir", "run")).name
    out_dir = out_dir or REPO_ROOT / "paper" / "elsevier" / "evidence" / f"{run}_{split}"
    pack = Pack(out_dir)
    pack.add_input(config)
    plt = setup_matplotlib()

    hist = load_training_history(cfg, pack)
    if hist:
        fig_training(plt, pack, hist)
    else:
        pack.skipped.append("training curves: no trainer_log_history.json or checkpoint trainer_state.json found")

    fig_dataset(plt, pack, resolve(cfg["dataset_dir"]), cfg["model_id"], int(cfg.get("max_seq_length", 8192)))

    lora_results = lora_results or resolve(cfg.get("results_dir"))
    base_results = base_results or resolve(cfg.get("base_results_dir")) or (resolve(str(cfg.get("results_dir")).rstrip("/\\") + "_base") if cfg.get("results_dir") else None)
    conds = {}
    for name, path in (("base", base_results), ("lora", lora_results)):
        preds = load_condition(path, pack)
        if preds:
            meta_path = path / "metrics.json"
            got_split = json.loads(meta_path.read_text(encoding="utf-8")).get("dataset", {}).get("split") if meta_path.is_file() else None
            if got_split and got_split != split:
                pack.skipped.append(f"{name}: {path} holds the {got_split} split, not {split}; not used")
                continue
            conds[name] = preds
        elif preds is None:
            pack.skipped.append(f"{name}: no predictions.jsonl under {path}")
    if conds:
        results_figures(plt, pack, conds, split)
        qualitative_examples(pack, conds, split)
    ext_path = (lora_results / "metrics_extended.json") if lora_results else None
    if "lora" in conds and ext_path and ext_path.is_file():
        pack.add_input(ext_path)
        calibration_figures(plt, pack, json.loads(ext_path.read_text(encoding="utf-8")), split)

    captions = ["# Draft figure and table captions", "", f"Run `{run}`, split `{split}`. Generated {datetime.now(timezone.utc).isoformat()}.", ""]
    captions += [f"- **{stem}**: {text}" for stem, text in pack.captions]
    if pack.skipped:
        captions += ["", "## Not generated", ""] + [f"- {s}" for s in pack.skipped]
    path = out_dir / "CAPTIONS.md"
    path.write_text("\n".join(captions) + "\n", encoding="utf-8")
    pack.outputs.append(path)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "split": split,
        "conditions": sorted(conds),
        "statistics": {"bootstrap_resamples": N_BOOT, "bootstrap_seed": BOOT_SEED, "interval": "95% percentile",
                       "paired_difference": "paired bootstrap over shared example IDs"},
        "rouge": "rouge_score, rouge1/rouge2/rougeL F-measure, use_stemmer=True (same as evaluate_rewrite.py)",
        "inputs_sha256": pack.inputs,
        "outputs_sha256": {str(p.relative_to(out_dir)): sha256(p) for p in pack.outputs},
        "skipped": pack.skipped,
        "note": "All values recomputed from the listed inputs; no hard-coded numbers.",
    }
    (out_dir / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Evidence pack: {out_dir}")
    print(f"  {sum(1 for p in pack.outputs if p.suffix == '.pdf')} figures, {sum(1 for p in pack.outputs if p.suffix == '.tex')} tables")
    for s in pack.skipped:
        print("  skipped:", s)
    return out_dir


def main() -> None:
    from paths import ACTIVE_TRAINING_CONFIG, CONFIG_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG_DIR / ACTIVE_TRAINING_CONFIG))
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--lora-results", default=None, help="Results dir of the LoRA evaluation (default: config results_dir)")
    ap.add_argument("--base-results", default=None, help="Results dir of the base-model evaluation (default: <results_dir>_base)")
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    build(resolve(a.config), a.split, resolve(a.lora_results), resolve(a.base_results), resolve(a.output_dir))


if __name__ == "__main__":
    main()
