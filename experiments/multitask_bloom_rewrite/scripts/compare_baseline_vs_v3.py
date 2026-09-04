#!/usr/bin/env python
"""Compare locked 1.5B baseline metrics vs improved v3 run (no averaging across tasks)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def g(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-metrics",
        default=str(
            REPO_ROOT
            / "experiments"
            / "multitask_bloom_rewrite"
            / "results"
            / "qwen15b_lora"
            / "metrics.json"
        ),
    )
    parser.add_argument(
        "--improved-metrics",
        default=str(
            REPO_ROOT
            / "experiments"
            / "multitask_bloom_rewrite"
            / "results"
            / "qwen15b_lora_v3"
            / "metrics.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "experiments"
            / "multitask_bloom_rewrite"
            / "results"
            / "qwen15b_lora_v3"
            / "baseline_vs_v3_comparison.json"
        ),
    )
    args = parser.parse_args()
    base_path = Path(args.baseline_metrics)
    imp_path = Path(args.improved_metrics)
    if not base_path.is_absolute():
        base_path = REPO_ROOT / base_path
    if not imp_path.is_absolute():
        imp_path = REPO_ROOT / imp_path
    if not imp_path.is_file():
        raise SystemExit(f"Improved metrics missing: {imp_path}")
    base = load(base_path)
    imp = load(imp_path)

    rows = []

    def add(metric: str, b, i):
        delta = None
        if isinstance(b, (int, float)) and isinstance(i, (int, float)):
            delta = round(float(i) - float(b), 6)
        rows.append({"metric": metric, "baseline_1.5B": b, "improved_1.5B_v3": i, "delta": delta})

    add("accuracy", g(base, "bloom", "classification", "accuracy"), g(imp, "bloom", "classification", "accuracy"))
    add("macro_f1", g(base, "bloom", "classification", "macro_f1"), g(imp, "bloom", "classification", "macro_f1"))
    for lvl in ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"):
        add(
            f"f1_{lvl}",
            g(base, "bloom", "classification", "per_level", lvl, "f1"),
            g(imp, "bloom", "classification", "per_level", lvl, "f1"),
        )
    for rate in (
        "format_valid_rate",
        "cognitive_valid_rate",
        "fully_validated_rewrite_rate",
        "answer_output_rate",
        "semantic_valid_rate",
        "classifier_match_rate",
    ):
        add(rate, g(base, "bloom", "rates", rate), g(imp, "bloom", "rates", rate))
    add("squad_em", g(base, "qa", "squad", "exact_match"), g(imp, "qa", "squad", "exact_match"))
    add("squad_f1", g(base, "qa", "squad", "f1"), g(imp, "qa", "squad", "f1"))
    for r in ("rouge1", "rouge2", "rougeL"):
        add(r, g(base, "summarization", r), g(imp, "summarization", r))

    report = {
        "baseline_path": str(base_path),
        "improved_path": str(imp_path),
        "rows": rows,
        "deployment_recommendation": "INCONCLUSIVE until human review and all protocol checks pass",
        "note": "Task metrics are separate; no overall average is computed.",
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    lines = [
        "# Baseline 1.5B vs Improved v3",
        "",
        "| Metric | Current 1.5B | Improved 1.5B | Delta |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['metric']} | {r['baseline_1.5B']} | {r['improved_1.5B_v3']} | {r['delta']} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", out)
    print("Wrote", md)


if __name__ == "__main__":
    main()
