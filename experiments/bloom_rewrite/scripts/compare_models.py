#!/usr/bin/env python
"""Paired comparison of rewrite generators. Does not invent a winner without metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_bloom_rewrite_models import main as _paired_main  # noqa: E402


DECISION_RULE = {
    "defined_before_results": True,
    "primary": [
        "classifier_based_target_agreement on held-out test (not treated as human GT)",
        "independent cognitive-task / topic / triviality validators",
        "human 1-5 ratings when collected",
    ],
    "secondary": ["latency", "peak RSS", "GGUF size", "startup cost"],
    "rule": [
        "Exclude models that fail the quality threshold on target cognitive transformation.",
        "Among remaining models, prefer the cheaper model if quality is practically comparable.",
        "Select 1.5B only if it has a substantial quality advantage AND fits CPU/offline memory constraints.",
        "Do not select by size alone.",
    ],
    "quality_threshold_note": (
        "Numeric threshold is locked after a pilot on validation only; it must not be "
        "tuned on the test set. Until training exists, no threshold is applied."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-a", default=None)
    parser.add_argument("--pred-b", default=None)
    parser.add_argument("--metrics-a", default=None)
    parser.add_argument("--metrics-b", default=None)
    parser.add_argument("--label-a", default="qwen05b")
    parser.add_argument("--label-b", default="qwen15b")
    parser.add_argument("--output-dir", default="experiments/bloom_rewrite/results/comparison")
    args, unknown = parser.parse_known_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[3] / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision_rule.json").write_text(json.dumps(DECISION_RULE, indent=2), encoding="utf-8")

    if not args.pred_a or not args.pred_b:
        payload = {
            "status": "NO MODEL WINNER — TRAINING/EVALUATION NOT COMPLETED.",
            "decision_rule": DECISION_RULE,
            "required": "predictions.jsonl from both 0.5B and 1.5B on the identical test split",
        }
        (out / "model_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        reports = Path(__file__).resolve().parents[1] / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "model_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    sys.argv = [
        sys.argv[0],
        "--pred-a", args.pred_a,
        "--pred-b", args.pred_b,
        "--label-a", args.label_a,
        "--label-b", args.label_b,
        "--output-dir", str(out),
    ]
    if args.metrics_a:
        sys.argv += ["--metrics-a", args.metrics_a]
    if args.metrics_b:
        sys.argv += ["--metrics-b", args.metrics_b]
    _paired_main()


if __name__ == "__main__":
    main()
