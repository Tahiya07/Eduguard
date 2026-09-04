#!/usr/bin/env python
"""Optional deterministic second-pass rewrite pipeline (separate from primary benchmark).

Flow:
  generate → format/semantic/cognitive validators → optional Bloom classifier
  if fail → constrained regenerate once → revalidate

Keeps first-pass and second-pass stats separate. Does not alter primary metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bloom_validation import validate_bloom_example  # noqa: E402
from eval_classifier import load_classifier  # noqa: E402
from eval_dataset import load_test_split  # noqa: E402
from eval_model import HFGenerator, resolve_checkpoint, set_seed, validate_checkpoint  # noqa: E402
from paths import CONFIG_DIR, SEED, TASK_BLOOM  # noqa: E402
from prompts import build_generation_prompt  # noqa: E402

CONSTRAINT = (
    "\n\nConstraint: OUTPUT ONLY ONE exam question. "
    "Do not answer. Do not explain. Prefer a clear interrogative or exam imperative."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional second-pass Bloom rewrite eval")
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "qwen15b_multitask_v3.json"),
    )
    parser.add_argument("--condition", choices=["base", "lora"], default="lora")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: results/<model>_second_pass/",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in ("output_dir", "results_dir", "dataset_dir"):
        if key in cfg and cfg[key] and not Path(cfg[key]).is_absolute():
            cfg[key] = str((REPO_ROOT / cfg[key]).resolve())

    gen_cfg = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": int(cfg.get("generation", {}).get("max_new_tokens", 128)),
    }
    set_seed(SEED)
    data_dir = Path(cfg.get("dataset_dir", REPO_ROOT / "data" / "multitask_bloom_rewrite_v3"))
    rows, meta = load_test_split(data_dir)
    bloom_rows = [r for r in rows if r["task"] == TASK_BLOOM]
    if args.limit:
        bloom_rows = bloom_rows[: args.limit]

    ckpt = resolve_checkpoint(cfg, args.condition)
    validate_checkpoint(ckpt, int(cfg.get("max_seq_length", 512)), gen_cfg)
    classify_fn, clf_meta = load_classifier(None, repo_root=REPO_ROOT, require_smoke=True)
    generator = HFGenerator(ckpt, int(cfg.get("max_seq_length", 512)))

    first_pass_ok = 0
    second_pass_ok = 0
    latencies = []
    records = []
    for row in bloom_rows:
        prompt = build_generation_prompt(TASK_BLOOM, row)
        t0 = time.perf_counter()
        pred1 = generator.generate(prompt, gen_cfg).strip()
        lat1 = time.perf_counter() - t0
        v1 = validate_bloom_example(row["source_question"], row["target_bloom_level"], pred1)
        clf1 = None
        conf1 = 0.0
        try:
            c = classify_fn(pred1)
            clf1, conf1 = c["predicted_level"], c["confidence"]
        except Exception:
            pass
        passed1 = v1.accepted and clf1 == row["target_bloom_level"] and conf1 >= 0.5
        pred2 = None
        v2 = None
        lat2 = 0.0
        passed2 = False
        if passed1:
            first_pass_ok += 1
        else:
            prompt2 = prompt + CONSTRAINT
            t1 = time.perf_counter()
            pred2 = generator.generate(prompt2, gen_cfg).strip()
            lat2 = time.perf_counter() - t1
            v2 = validate_bloom_example(row["source_question"], row["target_bloom_level"], pred2)
            clf2 = None
            conf2 = 0.0
            try:
                c = classify_fn(pred2)
                clf2, conf2 = c["predicted_level"], c["confidence"]
            except Exception:
                pass
            passed2 = bool(v2 and v2.accepted and clf2 == row["target_bloom_level"] and conf2 >= 0.5)
            if passed2:
                second_pass_ok += 1
        latencies.append(lat1 + lat2)
        records.append(
            {
                "id": row.get("example_id"),
                "first_pass_prediction": pred1,
                "first_pass_accepted": passed1,
                "second_pass_prediction": pred2,
                "second_pass_accepted": passed2,
                "first_pass_latency_s": round(lat1, 6),
                "second_pass_latency_s": round(lat2, 6),
            }
        )

    out_dir = Path(args.output_dir) if args.output_dir else (
        EXPERIMENT_DIR / "results" / f"qwen{cfg.get('model_key', '15b')}_second_pass"
    )
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(bloom_rows) or 1
    summary = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(bloom_rows),
        "first_pass_accept_rate": round(first_pass_ok / n, 6),
        "second_pass_rescue_rate": round(second_pass_ok / n, 6),
        "combined_accept_rate": round((first_pass_ok + second_pass_ok) / n, 6),
        "mean_latency_s": round(sum(latencies) / n, 6),
        "classifier_meta": {k: v for k, v in clf_meta.items() if k != "smoke_traceback"},
        "note": "Second-pass stats are SEPARATE from primary benchmark metrics.json",
        "dataset": meta,
    }
    (out_dir / "second_pass_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "second_pass_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
