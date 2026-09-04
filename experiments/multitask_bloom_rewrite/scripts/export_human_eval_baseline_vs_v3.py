#!/usr/bin/env python
"""Export blinded human-eval pairs: baseline vs improved (no model names/scores)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-predictions",
        default=str(
            REPO_ROOT
            / "experiments/multitask_bloom_rewrite/results/qwen15b_lora/predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--improved-predictions",
        default=str(
            REPO_ROOT
            / "experiments/multitask_bloom_rewrite/results/qwen15b_lora_v3/predictions.jsonl"
        ),
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "experiments/multitask_bloom_rewrite/human_eval/blinded_baseline_vs_v3.jsonl"
        ),
    )
    args = parser.parse_args()
    base_path = Path(args.baseline_predictions)
    imp_path = Path(args.improved_predictions)
    if not base_path.is_absolute():
        base_path = REPO_ROOT / base_path
    if not imp_path.is_absolute():
        imp_path = REPO_ROOT / imp_path
    if not imp_path.is_file():
        raise SystemExit(f"Improved predictions missing: {imp_path}")

    base = {r["id"]: r for r in read_jsonl(base_path) if r.get("task") == "bloom_rewrite"}
    imp = {r["id"]: r for r in read_jsonl(imp_path) if r.get("task") == "bloom_rewrite"}
    common = sorted(set(base) & set(imp))
    rng = random.Random(args.seed)
    rng.shuffle(common)
    sample = common[: args.sample_size]

    out_rows = []
    for i, eid in enumerate(sample):
        b, a = base[eid], imp[eid]
        # Randomize order of A/B without revealing which is baseline
        if rng.random() < 0.5:
            left, right, left_tag, right_tag = b, a, "X", "Y"
        else:
            left, right, left_tag, right_tag = a, b, "X", "Y"
        out_rows.append(
            {
                "item_id": f"he_{i:04d}",
                "example_id": eid,
                "original_question": b.get("source_question"),
                "target_bloom_level": b.get("target_bloom_level"),
                "output_X": left.get("prediction"),
                "output_Y": right.get("prediction"),
                # hidden mapping for later unblinding only (not for raters)
                "_hidden_mapping": {
                    "X": "baseline" if left is b else "improved",
                    "Y": "baseline" if right is b else "improved",
                },
                "ratings": {
                    "bloom_alignment_X": None,
                    "bloom_alignment_Y": None,
                    "cognitive_demand_X": None,
                    "cognitive_demand_Y": None,
                    "topic_preservation_X": None,
                    "topic_preservation_Y": None,
                    "question_validity_X": None,
                    "question_validity_Y": None,
                    "grammar_X": None,
                    "grammar_Y": None,
                    "overall_X": None,
                    "overall_Y": None,
                    "preference": None,
                },
            }
        )

    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Rater file without hidden mapping
    rater_path = out.with_name(out.stem + "_rater.jsonl")
    with rater_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            public = {k: v for k, v in row.items() if not k.startswith("_")}
            handle.write(json.dumps(public, ensure_ascii=False) + "\n")
    with out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Wrote rater file", rater_path)
    print("Wrote keyed file", out)


if __name__ == "__main__":
    main()
