#!/usr/bin/env python
"""Build multitask Mix-A corpus v3 WITHOUT overwriting data/multitask_bloom_rewrite.

Critical: FREEZES the exact locked TEST split from the current baseline
(data/multitask_bloom_rewrite/test.jsonl) for fair 1.5B comparison.

Train Bloom rows come from bloom_rewrite_synth_v3 (leakage-checked).
QA / summarization rows are reused from the locked multitask corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from bloom_validation import validate_bloom_example  # noqa: E402
from paths import (  # noqa: E402
    DEFAULT_TRAIN_MIX,
    REPORTS_DIR,
    SEED,
    TASK_BLOOM,
    TASK_QA,
    TASK_SUMMARIZATION,
)
from prompts import build_generation_prompt, build_prompt_only_text, build_sft_text  # noqa: E402

LOCKED_MULTITASK = REPO_ROOT / "data" / "multitask_bloom_rewrite"
V3_BLOOM_DIR = REPO_ROOT / "data" / "bloom_rewrite_versions" / "bloom_rewrite_synth_v3"
OUT_DIR = REPO_ROOT / "data" / "multitask_bloom_rewrite_v3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def bloom_key(row: dict) -> str:
    return str(row.get("group_id") or row.get("source_id") or row.get("source_question", "")).lower()


def enrich_bloom(row: dict) -> dict:
    out = dict(row)
    out["task"] = TASK_BLOOM
    out["text"] = build_sft_text(TASK_BLOOM, out)
    # Training script requires sft_text; keep text for compatibility/inspection.
    out["sft_text"] = out["text"]
    out["prompt_text"] = build_prompt_only_text(TASK_BLOOM, out)
    out["generation_prompt"] = build_generation_prompt(TASK_BLOOM, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multitask Mix-A corpus v3 (frozen test)")
    parser.add_argument("--locked-multitask-dir", default=str(LOCKED_MULTITASK))
    parser.add_argument("--bloom-v3-dir", default=str(V3_BLOOM_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mix-bloom", type=float, default=DEFAULT_TRAIN_MIX[TASK_BLOOM])
    parser.add_argument("--mix-qa", type=float, default=DEFAULT_TRAIN_MIX[TASK_QA])
    parser.add_argument("--mix-sum", type=float, default=DEFAULT_TRAIN_MIX[TASK_SUMMARIZATION])
    args = parser.parse_args()

    locked = Path(args.locked_multitask_dir)
    if not locked.is_absolute():
        locked = REPO_ROOT / locked
    bloom_v3 = Path(args.bloom_v3_dir)
    if not bloom_v3.is_absolute():
        bloom_v3 = REPO_ROOT / bloom_v3
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    locked_test = read_jsonl(locked / "test.jsonl")
    locked_train = read_jsonl(locked / "train.jsonl")
    locked_val = read_jsonl(locked / "validation.jsonl")
    locked_manifest = json.loads((locked / "dataset_manifest.json").read_text(encoding="utf-8"))

    if len(locked_test) != 8321:
        raise SystemExit(f"Frozen test must have 8321 rows; found {len(locked_test)}")

    test_bloom_keys = {bloom_key(r) for r in locked_test if r["task"] == TASK_BLOOM}
    train_bloom_keys = {bloom_key(r) for r in locked_train if r["task"] == TASK_BLOOM}
    val_bloom_keys = {bloom_key(r) for r in locked_val if r["task"] == TASK_BLOOM}

    v3_train = read_jsonl(bloom_v3 / "train.jsonl")
    v3_val = read_jsonl(bloom_v3 / "validation.jsonl")
    if not v3_train:
        raise SystemExit(f"Missing v3 Bloom train at {bloom_v3 / 'train.jsonl'}")

    # Leakage: no v3 train/val source group may appear in frozen test.
    def filter_leak(rows: list[dict], name: str) -> list[dict]:
        kept = []
        leaked = 0
        for r in rows:
            k = bloom_key(r)
            if k in test_bloom_keys:
                leaked += 1
                continue
            kept.append(enrich_bloom(r))
        if leaked:
            print(f"Dropped {leaked} {name} rows overlapping frozen test groups")
        return kept

    bloom_train = filter_leak(v3_train, "v3-train")
    bloom_val = filter_leak(v3_val, "v3-val")

    # Reuse locked QA/sum splits exactly (same IDs / texts).
    qa_train = [r for r in locked_train if r["task"] == TASK_QA]
    sum_train = [r for r in locked_train if r["task"] == TASK_SUMMARIZATION]
    qa_val = [r for r in locked_val if r["task"] == TASK_QA]
    sum_val = [r for r in locked_val if r["task"] == TASK_SUMMARIZATION]

    # Mix A on train: Bloom 40 / QA 30 / Sum 30 of TRAIN rows via controlled caps.
    rng = random.Random(args.seed)
    # Keep Bloom as large as available; size QA/sum to match mix relative to Bloom count.
    n_bloom = len(bloom_train)
    if n_bloom == 0:
        raise SystemExit("No Bloom train rows after leakage filter")
    # desired: bloom/total=0.4 => total = bloom/0.4; qa = 0.3*total
    desired_total = int(round(n_bloom / args.mix_bloom))
    n_qa = min(len(qa_train), int(round(desired_total * args.mix_qa)))
    n_sum = min(len(sum_train), int(round(desired_total * args.mix_sum)))
    rng.shuffle(qa_train)
    rng.shuffle(sum_train)
    train_rows = bloom_train + qa_train[:n_qa] + sum_train[:n_sum]
    rng.shuffle(train_rows)

    # Validation: Bloom v3 val + locked QA/sum val
    val_rows = bloom_val + qa_val + sum_val
    rng.shuffle(val_rows)

    # TEST: exact locked copy
    test_rows = locked_test

    # Assert no train/val Bloom key in test
    for split_name, rows in (("train", train_rows), ("validation", val_rows)):
        overlap = {bloom_key(r) for r in rows if r["task"] == TASK_BLOOM} & test_bloom_keys
        if overlap:
            raise SystemExit(f"Leakage into frozen test from {split_name}: {len(overlap)} keys")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "validation.jsonl", val_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    # Hash test file — must match locked file bytes for freeze guarantee.
    locked_test_hash = sha256_file(locked / "test.jsonl")
    new_test_hash = sha256_file(out_dir / "test.jsonl")
    if locked_test_hash != new_test_hash:
        raise SystemExit(
            "FATAL: v3 multitask test.jsonl hash differs from locked baseline test. "
            "Freeze requirement violated."
        )

    corpus_hash = sha256_text(
        locked_test_hash
        + sha256_file(out_dir / "train.jsonl")
        + sha256_file(out_dir / "validation.jsonl")
    )
    counts = {
        "train": {
            "total": len(train_rows),
            "by_task": dict(Counter(r["task"] for r in train_rows)),
        },
        "validation": {
            "total": len(val_rows),
            "by_task": dict(Counter(r["task"] for r in val_rows)),
        },
        "test": {
            "total": len(test_rows),
            "by_task": dict(Counter(r["task"] for r in test_rows)),
        },
    }
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "mix_preset": "A",
        "mix": {
            TASK_BLOOM: args.mix_bloom,
            TASK_QA: args.mix_qa,
            TASK_SUMMARIZATION: args.mix_sum,
        },
        "bloom_dataset_version": "bloom_rewrite_synth_v3",
        "locked_baseline_multitask_dir": str(locked.relative_to(REPO_ROOT)).replace("\\", "/"),
        "locked_test_sha256": locked_test_hash,
        "test_frozen": True,
        "test_identical_to_baseline": True,
        "corpus_hash": corpus_hash,
        "counts": counts,
        "notes": [
            "TEST split is byte-identical to data/multitask_bloom_rewrite/test.jsonl.",
            "Bloom train/val from synth_v3; QA/sum reused from locked multitask.",
            "Do not tune on test.",
        ],
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "multitask_v3_prepare_report.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Wrote", out_dir)
    print(json.dumps(counts, indent=2))
    print("locked_test_sha256", locked_test_hash)


if __name__ == "__main__":
    main()
