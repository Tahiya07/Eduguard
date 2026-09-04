#!/usr/bin/env python
"""Compare structural diversity of bloom_rewrite_synth_v2 vs v3 (code-only report)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stats_for(rows: list[dict]) -> dict:
    rewrites = [r.get("target_rewrite") or r.get("prediction") or "" for r in rows]
    prefixes = Counter((t.split()[0].lower() if t.split() else "") for t in rewrites)
    lengths = [len(t) for t in rewrites]
    words = [len(t.split()) for t in rewrites]
    # structural signature: first 3 tokens pattern
    sigs = Counter(" ".join(t.lower().split()[:3]) for t in rewrites if t)
    # repeated long phrases (5-grams)
    grams: Counter[str] = Counter()
    for t in rewrites:
        toks = re.findall(r"[a-z0-9]+", t.lower())
        for i in range(max(0, len(toks) - 4)):
            grams[" ".join(toks[i : i + 5])] += 1
    repeated = sum(1 for g, c in grams.items() if c >= 5)
    return {
        "n": len(rows),
        "unique_rewrites": len(set(rewrites)),
        "prefix_top20": dict(prefixes.most_common(20)),
        "mean_chars": round(sum(lengths) / max(1, len(lengths)), 2),
        "mean_words": round(sum(words) / max(1, len(words)), 2),
        "signature_top20": dict(sigs.most_common(20)),
        "n_structural_signatures": len(sigs),
        "repeated_5gram_types_ge5": repeated,
        "by_target": dict(Counter(r.get("target_bloom_level") for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2",
        default=str(REPO_ROOT / "data" / "bloom_rewrite" / "train.jsonl"),
    )
    parser.add_argument(
        "--v3",
        default=str(
            REPO_ROOT
            / "data"
            / "bloom_rewrite_versions"
            / "bloom_rewrite_synth_v3"
            / "train.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "experiments"
            / "multitask_bloom_rewrite"
            / "reports"
            / "diversity_v2_vs_v3.json"
        ),
    )
    args = parser.parse_args()
    v2 = Path(args.v2)
    v3 = Path(args.v3)
    if not v2.is_absolute():
        v2 = REPO_ROOT / v2
    if not v3.is_absolute():
        v3 = REPO_ROOT / v3
    if not v3.is_file():
        raise SystemExit(f"v3 train missing (generate first): {v3}")
    report = {
        "v2": stats_for(read_jsonl(v2)),
        "v3": stats_for(read_jsonl(v3)),
        "claim_rule": (
            "Do not claim v3 is more diverse unless n_structural_signatures increases "
            "and repeated_5gram_types_ge5 decreases relative to v2."
        ),
    }
    v2s, v3s = report["v2"], report["v3"]
    report["comparison"] = {
        "unique_ratio_v2": round(v2s["unique_rewrites"] / max(1, v2s["n"]), 4),
        "unique_ratio_v3": round(v3s["unique_rewrites"] / max(1, v3s["n"]), 4),
        "signatures_v2": v2s["n_structural_signatures"],
        "signatures_v3": v3s["n_structural_signatures"],
        "repeated_5grams_v2": v2s["repeated_5gram_types_ge5"],
        "repeated_5grams_v3": v3s["repeated_5gram_types_ge5"],
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
