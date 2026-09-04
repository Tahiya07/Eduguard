#!/usr/bin/env python
"""Static + offline summarization length/truncation audit (no model inference).

Inspects multitask summarization rows and optional prediction files.
Reports length distributions, truncation risk under max_seq_length / max_new_tokens.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from paths import TASK_SUMMARIZATION  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def summarize_lengths(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {
        "n": len(s),
        "mean": round(sum(s) / len(s), 2),
        "median": round(statistics.median(s), 2),
        "p95": round(pct(s, 95), 2),
        "min": s[0],
        "max": s[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarization length/truncation audit")
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "data" / "multitask_bloom_rewrite" / "train.jsonl"),
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional predictions.jsonl to score empty/short/long/repetition rates",
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--approx-chars-per-token",
        type=float,
        default=4.0,
        help="Rough char→token estimate when tokenizer unavailable",
    )
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "experiments"
            / "multitask_bloom_rewrite"
            / "reports"
            / "summarization_length_audit.json"
        ),
    )
    args = parser.parse_args()
    data_path = Path(args.dataset)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    rows = [r for r in read_jsonl(data_path) if r.get("task") == TASK_SUMMARIZATION]
    arts = [str(r.get("article") or "") for r in rows]
    abs_ = [str(r.get("abstract") or "") for r in rows]
    art_chars = [len(a) for a in arts]
    abs_chars = [len(a) for a in abs_]
    art_tok_est = [int(c / args.approx_chars_per_token) for c in art_chars]
    abs_tok_est = [int(c / args.approx_chars_per_token) for c in abs_chars]
    # Prompt overhead estimate (~80 tokens for ChatML + system)
    prompt_overhead = 80
    trunc_src = sum(
        1 for t in art_tok_est if t + prompt_overhead > args.max_seq_length
    )
    trunc_tgt = sum(1 for t in abs_tok_est if t > args.max_new_tokens)
    ratios = [
        (abs_chars[i] / art_chars[i]) if art_chars[i] else 0.0 for i in range(len(rows))
    ]
    report: dict = {
        "dataset": str(data_path),
        "n_summarization": len(rows),
        "source_chars": summarize_lengths(art_chars),
        "target_chars": summarize_lengths(abs_chars),
        "source_tokens_est": summarize_lengths(art_tok_est),
        "target_tokens_est": summarize_lengths(abs_tok_est),
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "approx_chars_per_token": args.approx_chars_per_token,
        "source_truncation_rate_est": round(trunc_src / max(1, len(rows)), 6),
        "target_exceeds_max_new_tokens_rate_est": round(trunc_tgt / max(1, len(rows)), 6),
        "mean_compression_ratio_chars": round(sum(ratios) / max(1, len(ratios)), 6),
        "findings": [],
        "recommended_followup_config": None,
    }
    if report["source_truncation_rate_est"] > 0.25:
        report["findings"].append(
            "High estimated source truncation under max_seq_length=512; "
            "consider a controlled summarization-only config with larger max_seq_length."
        )
        report["recommended_followup_config"] = (
            "experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_sumfix.json"
        )
    if report["target_exceeds_max_new_tokens_rate_est"] > 0.5:
        report["findings"].append(
            "Most gold abstracts appear longer than max_new_tokens=128 (est.); "
            "ROUGE may be depressed by generation length cap — create controlled "
            "sumfix config with max_new_tokens=256 without changing Bloom/QA protocol defaults."
        )
        report["recommended_followup_config"] = (
            "experiments/multitask_bloom_rewrite/configs/qwen15b_multitask_sumfix.json"
        )

    if args.predictions:
        pred_path = Path(args.predictions)
        if not pred_path.is_absolute():
            pred_path = REPO_ROOT / pred_path
        preds = [r for r in read_jsonl(pred_path) if r.get("task") == TASK_SUMMARIZATION]
        outs = [str(r.get("prediction") or "") for r in preds]
        empty = sum(1 for o in outs if not o.strip())
        short = sum(1 for o in outs if 0 < len(o.split()) < 10)
        long = sum(1 for o in outs if len(o.split()) > args.max_new_tokens)
        reps = 0
        for o in outs:
            words = o.lower().split()
            if len(words) >= 8:
                # crude repetition: most common bigram share
                from collections import Counter

                bg = Counter(zip(words, words[1:]))
                if bg and bg.most_common(1)[0][1] >= 4:
                    reps += 1
        report["prediction_audit"] = {
            "n": len(preds),
            "empty_output_rate": round(empty / max(1, len(preds)), 6),
            "excessively_short_rate": round(short / max(1, len(preds)), 6),
            "excessively_long_rate": round(long / max(1, len(preds)), 6),
            "repetition_rate_heuristic": round(reps / max(1, len(preds)), 6),
        }

    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_summarization", "findings", "source_truncation_rate_est", "target_exceeds_max_new_tokens_rate_est")}, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
