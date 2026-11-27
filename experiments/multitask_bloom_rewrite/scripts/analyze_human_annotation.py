#!/usr/bin/env python
"""Score the filled human-annotation sheets: per-system results with 95% CIs and inter-rater agreement.

Reads <dir>/0[123]_*_R1.csv and _R2.csv (filled by raters) plus 04_annotation_key_DO_NOT_SHARE.csv,
writes human_eval_results.json and human_eval_results.md next to them.

  python experiments/multitask_bloom_rewrite/scripts/analyze_human_annotation.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_DIR = REPO / "paper" / "elsevier" / "human_annotation"
LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
SEED, BOOT = 42, 2000

# (column, kind, how to turn a value into a number for the per-system mean)
METRICS = {
    "01_bloom_rewrite": [
        ("matches_requested_level", "binary"), ("topic_preserved", "binary"), ("self_contained", "binary"),
        ("factually_sound", "binary"), ("clarity_1to5", "ordinal5"), ("exam_usability_1to5", "ordinal5"),
        ("perceived_bloom_level", "level"),
    ],
    "02_qa": [
        ("correctness", "correctness"), ("contains_false_statement", "binary"),
        ("completeness_1to5", "ordinal5"), ("on_topic_and_concise", "binary"),
    ],
    "03_summarization": [
        ("faithful", "binary"), ("unsupported_claim_count", "count"),
        ("coverage_1to5", "ordinal5"), ("fluency_1to5", "ordinal5"), ("length", "length"),
    ],
}


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def to_value(kind: str, raw: str):
    v = (raw or "").strip().lower()
    if not v:
        return None
    if kind == "binary":
        return {"yes": 1.0, "no": 0.0}.get(v)  # 'n/a' -> None
    if kind == "ordinal5":
        return float(v) if v in {"1", "2", "3", "4", "5"} else None
    if kind == "count":
        return float(v) if v.isdigit() else None
    if kind == "correctness":
        return {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}.get(v)
    if kind == "level":
        return next((float(i) for i, lvl in enumerate(LEVELS) if lvl.lower() == v), None)
    if kind == "length":
        return {"too_short": 0.0, "appropriate": 1.0, "too_long": 2.0}.get(v)
    return None


def mean_ci(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "ci95": None}
    rng = random.Random(SEED)
    boots = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(BOOT))
    return {"n": len(vals), "mean": round(sum(vals) / len(vals), 4),
            "ci95": [round(boots[int(0.025 * BOOT)], 4), round(boots[int(0.975 * BOOT) - 1], 4)]}


def kappa(a: list, b: list, weighted: bool = False) -> float | None:
    """Cohen's kappa; quadratic-weighted when the categories are ordered numbers."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    cats = sorted({x for p in pairs for x in p})
    if len(cats) < 2:
        return 1.0
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in pairs:
        obs[idx[x]][idx[y]] += 1
    n = len(pairs)
    ra = [sum(row) for row in obs]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        return ((i - j) / (k - 1)) ** 2 if weighted else (0.0 if i == j else 1.0)

    o = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    e = sum(w(i, j) * ra[i] * rb[j] for i in range(k) for j in range(k)) / (n * n)
    return None if e == 0 else round(1 - o / e, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    d = Path(args.dir)
    key = {r["item_id"]: r for r in read(d / "04_annotation_key_DO_NOT_SHARE.csv")}
    results: dict = {}
    lines = ["# Human evaluation results", "",
             "Mean over raters and items [95% bootstrap CI]. Binary columns are proportions of 'yes'; "
             "QA correctness counts partially correct as 0.5.", ""]

    for sheet, metrics in METRICS.items():
        raters = sorted(d.glob(f"{sheet}_R*.csv"))
        if not raters:
            continue
        data = {p.stem.rsplit("_", 1)[-1]: {r["item_id"]: r for r in read(p)} for p in raters}
        per_system: dict = defaultdict(lambda: defaultdict(list))
        agreement: dict = {}
        for col, kind in metrics:
            for rater, rows in data.items():
                for item, row in rows.items():
                    val = to_value(kind, row.get(col, ""))
                    if kind == "level":  # human Bloom accuracy: perceived level == requested level
                        requested = key[item]["requested_or_type"]
                        val = None if val is None else float(LEVELS[int(val)] == requested)
                        per_system[key[item]["system"]]["human_level_accuracy"].append(val)
                        continue
                    per_system[key[item]["system"]][col].append(val)
            if len(data) >= 2:
                r1, r2 = list(data.values())[:2]
                items = sorted(set(r1) & set(r2))
                a = [to_value(kind, r1[i].get(col, "")) for i in items]
                b = [to_value(kind, r2[i].get(col, "")) for i in items]
                ordered = kind in {"ordinal5", "level", "correctness", "count", "length"}
                agree = [x == y for x, y in zip(a, b) if x is not None and y is not None]
                agreement[col] = {"kappa": kappa(a, b, weighted=ordered),
                                  "kappa_type": "quadratic-weighted" if ordered else "Cohen",
                                  "percent_agreement": round(sum(agree) / len(agree), 4) if agree else None,
                                  "n_items": len(agree)}
        results[sheet] = {"raters": sorted(data), "per_system": {s: {m: mean_ci(v) for m, v in cols.items()}
                                                                   for s, cols in per_system.items()},
                          "agreement": agreement}

        lines += [f"## {sheet}", "", f"Raters: {', '.join(sorted(data))}", ""]
        systems = sorted(per_system)
        cols = sorted({m for s in systems for m in per_system[s]})
        lines += ["| Measure | " + " | ".join(systems) + " |", "|---|" + "---:|" * len(systems)]
        for m in cols:
            cells = []
            for s in systems:
                r = results[sheet]["per_system"][s].get(m, {"mean": None})
                cells.append("—" if r["mean"] is None else f"{r['mean']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        if agreement:
            lines += ["", "| Inter-rater agreement | κ | type | % agreement | n |", "|---|---:|---|---:|---:|"]
            for col, a in agreement.items():
                lines.append(f"| {col} | {a['kappa']} | {a['kappa_type']} | {a['percent_agreement']} | {a['n_items']} |")
        lines.append("")

    (d / "human_eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (d / "human_eval_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
