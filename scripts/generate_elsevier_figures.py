"""Generate five Elsevier single-column publication figures from project artifacts.

Each figure is saved as a 600-DPI PNG and a vector PDF in
``paper/elsevier/figures``. Run: ``python scripts/generate_elsevier_figures.py``.
"""
from __future__ import annotations

from pathlib import Path

import json
from collections import Counter

import faiss
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Shared journal styling. Arial is used when installed; DejaVu Sans is the
# portable fallback. The palette is based on Okabe-Ito/colorblind-safe tones.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper" / "elsevier" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#4C78A8",
    "orange": "#E08E45",
    "green": "#5B9E77",
    "red": "#C66B6B",
    "purple": "#8B78AA",
    "teal": "#4C9A9A",
    "slate": "#52616B",
    "light_gray": "#E9ECEF",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)
sns.set_theme(style="whitegrid", rc={"grid.color": "#E4E7EB", "grid.linewidth": 0.55})


def polish_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Apply restrained single-column axis styling."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.6, color="#4A4A4A")
    ax.grid(axis=grid_axis, zorder=0)


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Write both print-resolution raster and vector versions of a figure."""
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=600)
    plt.close(fig)


def load_json(relative_path: str) -> dict:
    """Load a versioned project artifact; fail rather than substitute mock data."""
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(relative_path: str) -> list[dict]:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Figure 1 — Federated Heterogeneity & Client Convergence Analysis
# Uses the project FedAvg/FedProx IID r20 run histories. The logs do not record
# client-level validation variance, so uncertainty is binomial standard error.
# ---------------------------------------------------------------------------
def figure_1_federated_convergence() -> None:
    fedavg = load_json("artifacts/federated/results/federated_lora_fedavg_iid_r20.json")
    fedprox = load_json("artifacts/federated/results/federated_lora_fedprox_iid_r20.json")
    rounds = np.array([row["round"] for row in fedavg["history"]])
    fedavg_mean = np.array([row["accuracy"] for row in fedavg["history"]])
    fedprox_mean = np.array([row["accuracy"] for row in fedprox["history"]])
    fedavg_sd = np.sqrt(fedavg_mean * (1 - fedavg_mean) / np.array([row["n_eval"] for row in fedavg["history"]]))
    fedprox_sd = np.sqrt(fedprox_mean * (1 - fedprox_mean) / np.array([row["n_eval"] for row in fedprox["history"]]))

    fig, ax = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)
    for mean, sd, color, label, marker in [
        (fedavg_mean, fedavg_sd, COLORS["blue"], "FedAvg (IID)", "o"),
        (fedprox_mean, fedprox_sd, COLORS["orange"], r"FedProx ($\mu=0.01$, IID)", "s"),
    ]:
        ax.fill_between(rounds, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
        ax.plot(rounds, mean, color=color, label=label, linewidth=1.8, marker=marker,
                markersize=3.3, markeredgewidth=0, zorder=3)
    delta_pp = 100 * (fedprox_mean[-1] - fedavg_mean[-1])
    ax.annotate(f"+{delta_pp:.1f} pp", xy=(20, fedprox_mean[-1]), xytext=(14.2, 0.87),
                color=COLORS["orange"], fontsize=7,
                arrowprops={"arrowstyle": "-", "color": COLORS["orange"], "lw": 0.8})
    ax.set(xlim=(1, 20), ylim=(0.05, 0.90), xlabel="Communication round", ylabel="Validation accuracy")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(frameon=False, loc="lower right", handlelength=2.1)
    ax.text(0.01, 0.02, "Shading: ± binomial SE over held-out evaluation items", transform=ax.transAxes, fontsize=6.3, color=COLORS["slate"])
    polish_axes(ax)
    save_figure(fig, "figure1_federated_heterogeneity_convergence")


# ---------------------------------------------------------------------------
# Figure 2 — Qualitative classify-rewrite-verify pipeline.
# Uses a fully validated held-out example from the Q4_K_M full evaluation.
# ---------------------------------------------------------------------------
def figure_2_pipeline_example() -> None:
    predictions = load_jsonl("experiments/multitask_bloom_rewrite/results/gguf/q4_k_m_full/predictions.jsonl")
    example = next(row for row in predictions if row.get("task") == "bloom_rewrite" and row.get("target_bloom_level") == "Analyze" and row.get("fully_validated"))
    fig, ax = plt.subplots(figsize=(3.5, 5.0), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.07, 0.78, 0.86, 0.15, "1  Original question", example["source_question"], example["source_bloom_level"], COLORS["blue"]),
        (0.07, 0.56, 0.86, 0.14, "2  Requested target level", example["target_bloom_level"], "Held-out evaluation objective", COLORS["purple"]),
        (0.07, 0.31, 0.86, 0.20, "3  1.5B Q4_K_M generator", example["prediction"], f"Latency: {example['latency_s']:.2f} s", COLORS["green"]),
        (0.07, 0.07, 0.86, 0.17, "4  0.5B Bloom classifier", f"Confirmed: {example['classifier_prediction']}", f"p({example['classifier_prediction']}) = {example['classifier_confidence']:.2f}  |  validation match", COLORS["orange"]),
    ]
    for index, (x, y, w, h, heading, body, footnote, color) in enumerate(boxes):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                               linewidth=0.9, edgecolor=color, facecolor="white", zorder=2)
        ax.add_patch(patch)
        ax.add_patch(FancyBboxPatch((x, y + h - 0.042), w, 0.042,
                                    boxstyle="round,pad=0.012,rounding_size=0.015",
                                    linewidth=0, facecolor=color, alpha=0.14, zorder=2))
        ax.text(x + 0.025, y + h - 0.027, heading, va="center", ha="left", weight="bold", color="#30363B", fontsize=8)
        # The compact target-level step uses centered body text to preserve a
        # clear separation from its explanatory footer at print scale.
        if index == 1:
            ax.text(x + 0.025, y + 0.078, body, va="center", ha="left", color="#20262B", fontsize=8)
        else:
            ax.text(x + 0.025, y + h - 0.065, body, va="top", ha="left", color="#20262B", fontsize=8,
                    wrap=True, linespacing=1.28)
        ax.text(x + 0.025, y + 0.025, footnote, va="bottom", ha="left", color=color, fontsize=7, weight="bold")
        if index < len(boxes) - 1:
            next_y = boxes[index + 1][1] + boxes[index + 1][3]
            ax.add_patch(FancyArrowPatch((0.5, y - 0.012), (0.5, next_y + 0.015),
                                         arrowstyle="-|>", mutation_scale=10, linewidth=0.9,
                                         color="#7A838A", zorder=1))
    ax.text(0.5, 0.985, "Teacher-side classify–rewrite–verify: held-out example", ha="center", va="top", fontsize=9, weight="bold")
    save_figure(fig, "figure2_classify_rewrite_verify_pipeline")


# ---------------------------------------------------------------------------
# Figure 3 — PrivacyGuard attack-vector breakdown.
# Uses recorded PrivacyGuard final decision reasons per prompt category.
# ---------------------------------------------------------------------------
def figure_3_privacyguard_breakdown() -> None:
    privacy_log = load_json("artifacts/model train results/privacy_guard_eval.json")
    category_keys = ["direct_reconstruction", "model_aware_jailbreak", "partial_span_extraction", "semantic_reconstruction", "benign_study_help"]
    categories = ["Direct\nextraction", "Jailbreak", "Partial-span\nextraction", "Semantic\nreconstruction", "Benign study\nhelp"]
    outcomes = ["Reconstruction intent", "Federated privacy risk", "Allowed"]
    outcome_keys = ["reconstruction_intent_detected", "federated_privacy_risk", "ok"]
    counts = Counter((row["category"], row["reason"]) for row in privacy_log["rows"])
    intercept_rates = np.array([[counts[(category, outcome)] for outcome in outcome_keys] for category in category_keys], dtype=float)
    intercept_rates /= intercept_rates.sum(axis=1, keepdims=True)
    colors = [COLORS["blue"], COLORS["orange"], "#C9D3D8"]
    fig, ax = plt.subplots(figsize=(3.5, 2.65), constrained_layout=True)
    bottom = np.zeros(len(categories))
    for column, label, color in zip(intercept_rates.T, outcomes, colors):
        ax.bar(np.arange(len(categories)), column, bottom=bottom, width=0.68, label=label,
               color=color, edgecolor="white", linewidth=0.45, zorder=3)
        bottom += column
    ax.set(ylim=(0, 1), ylabel="Proportion of evaluated prompts", xlabel="Recorded prompt category")
    ax.set_xticks(np.arange(len(categories)), categories)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), columnspacing=1.0)
    polish_axes(ax)
    save_figure(fig, "figure3_privacyguard_attack_breakdown")


# ---------------------------------------------------------------------------
# Figure 4 — Deployment resource/quality Pareto frontier.
# Uses CPU benchmark, quantized classifier evaluation, and Q4_K_M evaluation.
# ---------------------------------------------------------------------------
def figure_4_pareto_frontier() -> None:
    cpu_benchmark = load_json("artifacts/model train results/cpu_deployment_benchmark.json")
    quantized_eval = load_json("artifacts/model train results/bloom_quantized_eval_0.5B.json")
    gguf_eval = load_json("experiments/multitask_bloom_rewrite/results/gguf/q4_k_m_full/metrics.json")
    gguf_benchmark = load_json("experiments/multitask_bloom_rewrite/results/q4_k_m_benchmark.json")
    benchmark_by_name = {row["name"]: row for row in cpu_benchmark["results"]}
    variants = ["Centralized\nFP32 0.5B", "FP16 merged\n0.5B", "Multitask Q4_K_M\n1.5B GGUF"]
    latency_ms = np.array([benchmark_by_name["merged"]["mean_latency_ms"], benchmark_by_name["lightweight"]["mean_latency_ms"], gguf_benchmark["mean_latency_sec"] * 1000])
    quality_score = np.array([0.8400, quantized_eval["qwen_lora"]["accuracy"], gguf_eval["bloom"]["classification"]["accuracy"]])
    memory_mib = np.array([benchmark_by_name["merged"]["size_mib"], benchmark_by_name["lightweight"]["size_mib"], gguf_benchmark["peak_memory_after_load"]["rss_mb"]])
    colors = [COLORS["slate"], COLORS["blue"], COLORS["green"]]
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    sizes = memory_mib / 4.0
    ax.scatter(latency_ms, quality_score, s=sizes, c=colors, alpha=0.80, edgecolor="white", linewidth=0.8, zorder=3)
    # A guide, rather than an asserted optimization, highlights the observed frontier.
    order = np.argsort(latency_ms)
    ax.plot(latency_ms[order], quality_score[order], color="#8C979D", linewidth=0.8, linestyle="--", zorder=1)
    offsets = [(11, -0.003), (9, 0.008), (9, -0.007)]
    for x, y, label, (dx, dy) in zip(latency_ms, quality_score, variants, offsets):
        ax.annotate(label, (x, y), xytext=(x + dx, y + dy), fontsize=7, va="center", color="#30363B")
    handles = [plt.scatter([], [], s=s / 4.0, color="#9BA7AD", alpha=0.75, edgecolor="white") for s in [800, 1600, 2400]]
    ax.legend(handles, ["800 MiB", "1,600 MiB", "2,400 MiB"], title="RSS memory", title_fontsize=7,
              fontsize=6.5, frameon=False, loc="lower right", handletextpad=0.7, labelspacing=0.45)
    ax.set_xscale("log")
    ax.set(xlabel="Mean inference latency (ms, log scale)", ylabel="Reported held-out quality")
    ax.text(0.01, 0.02, "0.5B: Bloom accuracy; 1.5B: rewrite classifier-match rate", transform=ax.transAxes, fontsize=6.2, color=COLORS["slate"])
    polish_axes(ax)
    save_figure(fig, "figure4_resource_quality_pareto_frontier")


# ---------------------------------------------------------------------------
# Figure 5 — actual public FAISS embedding space. A protected vector store is
# not present in this project, so the figure explicitly avoids a false claim of
# public/protected separation. Add data/vector_store/protected/index.faiss to
# show both scopes when such an index has been evaluated and approved.
# ---------------------------------------------------------------------------
def figure_5_embedding_space() -> None:
    public_index = faiss.read_index(str(ROOT / "data/vector_store/public/index.faiss"))
    embeddings = public_index.reconstruct_n(0, public_index.ntotal)
    if public_index.ntotal < 3:
        raise ValueError("The public FAISS index needs at least three vectors for t-SNE.")
    perplexity = min(30, max(2, (public_index.ntotal - 1) // 3))
    coordinates = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=20260924).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(3.5, 2.65), constrained_layout=True)
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=20, alpha=0.74, color=COLORS["blue"], marker="o", linewidth=0, label="Public student resources", zorder=3)
    ax.set(xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="best", markerscale=1.1)
    ax.text(0.02, 0.02, f"BGE-small public FAISS index (n={public_index.ntotal})\nProtected index unavailable: no separation asserted", transform=ax.transAxes,
            fontsize=6.8, color=COLORS["slate"], va="bottom")
    polish_axes(ax, grid_axis="both")
    save_figure(fig, "figure5_semantic_embedding_space")


def main() -> None:
    figure_1_federated_convergence()
    figure_2_pipeline_example()
    figure_3_privacyguard_breakdown()
    figure_4_pareto_frontier()
    figure_5_embedding_space()
    print(f"Generated 5 PNG/PDF figure pairs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
