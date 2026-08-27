#!/usr/bin/env python
"""Paired comparison of 0.5B vs 1.5B Bloom rewrite generators.

Both models must be scored on the identical held-out test examples.
This script does not select a winner unless both evaluation files exist.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from bloom_target_policy import BLOOM_LEVELS  # noqa: E402
from paths import HUMAN_EVAL_DIR, RESULTS_DIR  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mcnemar(b: int, c: int) -> dict:
    """McNemar test on discordant paired binary outcomes.

    b = A correct, B wrong; c = A wrong, B correct.
    """
    n_disc = b + c
    if n_disc == 0:
        return {
            "b": b,
            "c": c,
            "chi2": 0.0,
            "p_value": 1.0,
            "note": "No discordant pairs; models produced identical correctness.",
        }
    chi2 = (abs(b - c) - 1) ** 2 / n_disc  # continuity correction
    # chi-square survival with 1 df: erfc(sqrt(x/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": chi2, "p_value": p, "n_discordant": n_disc}


def bootstrap_diff(a: list[int], b: list[int], seed: int = 42, n_boot: int = 2000) -> dict:
    rng = __import__("random").Random(seed)
    n = len(a)
    if n == 0:
        return {"mean_diff": None, "ci95": [None, None], "n": 0, "note": "empty"}
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        acc_a = sum(a[i] for i in idx) / n
        acc_b = sum(b[i] for i in idx) / n
        diffs.append(acc_b - acc_a)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[min(n_boot - 1, int(0.975 * n_boot))]
    return {
        "mean_diff_b_minus_a": sum(diffs) / n_boot,
        "ci95": [lo, hi],
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
    }


def index_preds(rows: list[dict]) -> dict[str, dict]:
    return {row["example_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-a", required=True, help="0.5B predictions.jsonl")
    parser.add_argument("--pred-b", required=True, help="1.5B predictions.jsonl")
    parser.add_argument("--metrics-a", default=None)
    parser.add_argument("--metrics-b", default=None)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "comparison"))
    parser.add_argument("--label-a", default="qwen05b")
    parser.add_argument("--label-b", default="qwen15b")
    args = parser.parse_args()

    a_rows = read_jsonl(Path(args.pred_a))
    b_rows = read_jsonl(Path(args.pred_b))
    a_map = index_preds(a_rows)
    b_map = index_preds(b_rows)
    shared = sorted(set(a_map) & set(b_map))
    if not shared:
        raise SystemExit("No shared example_ids. Refusing to compare unequal test sets.")
    only_a = sorted(set(a_map) - set(b_map))
    only_b = sorted(set(b_map) - set(a_map))

    a_corr = [int(bool(a_map[i]["target_match"])) for i in shared]
    b_corr = [int(bool(b_map[i]["target_match"])) for i in shared]
    b_disc = sum(1 for x, y in zip(a_corr, b_corr) if x == 1 and y == 0)
    c_disc = sum(1 for x, y in zip(a_corr, b_corr) if x == 0 and y == 1)
    acc_a = sum(a_corr) / len(shared)
    acc_b = sum(b_corr) / len(shared)

    failures = {args.label_a: Counter(), args.label_b: Counter()}
    per_level = {
        args.label_a: defaultdict(lambda: {"n": 0, "correct": 0}),
        args.label_b: defaultdict(lambda: {"n": 0, "correct": 0}),
    }
    for eid in shared:
        for label, row in ((args.label_a, a_map[eid]), (args.label_b, b_map[eid])):
            tgt = row["target_level"]
            per_level[label][tgt]["n"] += 1
            per_level[label][tgt]["correct"] += int(bool(row["target_match"]))
            if row.get("failure_category"):
                failures[label][row["failure_category"]] += 1

    comparison = {
        "n_shared": len(shared),
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
        "accuracy": {args.label_a: acc_a, args.label_b: acc_b, "difference_b_minus_a": acc_b - acc_a},
        "mcnemar": mcnemar(b_disc, c_disc),
        "bootstrap": bootstrap_diff(a_corr, b_corr),
        "per_target_counts": {
            label: {level: dict(per_level[label][level]) for level in BLOOM_LEVELS}
            for label in (args.label_a, args.label_b)
        },
        "failures": {k: dict(v) for k, v in failures.items()},
        "small_sample_note": (
            None
            if len(shared) >= 200
            else "Test set is small for significance testing; interpret p-values cautiously."
        ),
        "compared_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.metrics_a and Path(args.metrics_a).is_file():
        comparison["metrics_a"] = json.loads(Path(args.metrics_a).read_text(encoding="utf-8"))
    if args.metrics_b and Path(args.metrics_b).is_file():
        comparison["metrics_b"] = json.loads(Path(args.metrics_b).read_text(encoding="utf-8"))

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    blinded = []
    key = []
    for eid in shared[:60]:
        pair = [(args.label_a, a_map[eid]), (args.label_b, b_map[eid])]
        pair.sort(key=lambda item: eid + item[0])
        for system_idx, (label, row) in enumerate(pair, start=1):
            item_id = f"{eid}_{system_idx}"
            blinded.append(
                {
                    "item_id": item_id,
                    "example_id": eid,
                    "source_question": row["source_question"],
                    "target_level": row["target_level"],
                    "system_output": row["generated_rewrite"],
                    "rater_target_bloom_correct": "",
                    "rater_topic_preservation": "",
                    "rater_question_quality": "",
                    "rater_academic_appropriateness": "",
                    "notes": "",
                }
            )
            key.append({"item_id": item_id, "example_id": eid, "hidden_model": label})
    if blinded:
        with (HUMAN_EVAL_DIR / "blinded_eval.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(blinded[0].keys()))
            writer.writeheader()
            writer.writerows(blinded)
        (HUMAN_EVAL_DIR / "blind_key.json").write_text(
            json.dumps(
                {
                    "warning": "Withhold this file from raters. It unblinds 0.5B vs 1.5B.",
                    "key": key,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(comparison, indent=2))
    print("Human scoring remains pending. Blinded CSV written for later rating.")


if __name__ == "__main__":
    main()
