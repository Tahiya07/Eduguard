#!/usr/bin/env python3
"""Build the corrected multitask SFT corpus.

Bloom rewrite rows come ONLY from data/rewrite_dataset/bloom_rewrite_*.jsonl.
QA and summarization rows are reused from the locked baseline multitask corpus.
The Bloom train portion is sampled at source level to produce an exact
40/30/30 Mix-A-style corpus as closely as integer counts allow.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BLOOM_DIR = ROOT / "data" / "rewrite_dataset"
LOCKED = ROOT / "data" / "multitask_bloom_rewrite"
OUT = ROOT / "data" / "multitask_corrected"
SEED = 42

LEVELS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )

def main() -> None:
    bloom_train = read_jsonl(BLOOM_DIR / "bloom_rewrite_train.jsonl")
    bloom_val = read_jsonl(BLOOM_DIR / "bloom_rewrite_validation.jsonl")
    bloom_test = read_jsonl(BLOOM_DIR / "bloom_rewrite_test.jsonl")

    base_train = read_jsonl(LOCKED / "train.jsonl")
    base_val = read_jsonl(LOCKED / "validation.jsonl")
    base_test = read_jsonl(LOCKED / "test.jsonl")

    def only_task(rows, task):
        return [r for r in rows if r.get("task") == task]

    qa_train = only_task(base_train, "qa")
    sum_train = only_task(base_train, "summarization")
    qa_val = only_task(base_val, "qa")
    sum_val = only_task(base_val, "summarization")
    qa_test = only_task(base_test, "qa")
    sum_test = only_task(base_test, "summarization")

    # Select complete Bloom source groups so all six target levels for a source
    # remain together.  Desired Bloom count for 40/30/30 with fixed QA/Sum:
    # total = (qa+sum) / .6; bloom = .4*total = (qa+sum)*2/3.
    fixed_other = len(qa_train) + len(sum_train)
    desired_bloom = round(fixed_other * (0.4 / 0.6))
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in bloom_train:
        by_source[r["source_id"]].append(r)

    # Prefer stratified selection by the original source Bloom label.
    groups = defaultdict(list)
    for sid, rows in by_source.items():
        if len(rows) != 6 or {r["target_bloom_level"] for r in rows} != set(LEVELS):
            continue
        groups[rows[0]["source_bloom_level"]].append((sid, rows))

    labels = [k for k in sorted(groups)]
    rng = random.Random(SEED)
    for label in labels:
        groups[label].sort(key=lambda x: stable_key(x[0]))

    n_sources_target = desired_bloom // 6
    # Allocate source slots proportionally to available label groups.
    total_available_groups = sum(len(groups[k]) for k in labels)
    allocation = {}
    for label in labels:
        allocation[label] = int(round(n_sources_target * len(groups[label]) / total_available_groups))
    # Correct rounding deterministically.
    while sum(allocation.values()) < n_sources_target:
        label = max(labels, key=lambda k: len(groups[k]) - allocation[k])
        allocation[label] += 1
    while sum(allocation.values()) > n_sources_target:
        label = max((k for k in labels if allocation[k] > 0), key=lambda k: allocation[k])
        allocation[label] -= 1

    selected = []
    for label in labels:
        selected.extend([rows for _, rows in groups[label][:allocation[label]]])
    bloom_train_selected = [r for group in selected for r in group]
    rng.shuffle(bloom_train_selected)

    train = bloom_train_selected + qa_train + sum_train
    rng.shuffle(train)

    validation = bloom_val + qa_val + sum_val
    rng.shuffle(validation)

    test = bloom_test + qa_test + sum_test
    rng.shuffle(test)

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "validation.jsonl", validation)
    write_jsonl(OUT / "test.jsonl", test)

    def counts(rows):
        return {
            "total": len(rows),
            "by_task": dict(Counter(r.get("task") for r in rows)),
        }

    manifest = {
        "dataset_version": "multitask_corrected_v1",
        "seed": SEED,
        "base_model_target": "Qwen2.5-1.5B-Instruct",
        "mix_preset": "A",
        "requested_mix": {
            "bloom_rewrite": 0.4,
            "qa": 0.3,
            "summarization": 0.3,
        },
        "sources": {
            "bloom_rewrite": "data/rewrite_dataset/bloom_rewrite_*.jsonl",
            "qa_and_summarization": "data/multitask_bloom_rewrite/{train,validation,test}.jsonl",
        },
        "train": counts(train),
        "validation": counts(validation),
        "test": counts(test),
        "bloom_train_sources_selected": len(selected),
        "bloom_train_rows_selected": len(bloom_train_selected),
        "bloom_train_source_all_six_targets": True,
        "notes": [
            "Old Bloom rewrite rows from data/multitask_bloom_rewrite are not reused.",
            "QA and summarization rows are unchanged from the locked baseline.",
            "Bloom train selection is source-grouped to preserve six target levels together.",
            "Test is never sampled or tuned against; Bloom test comes from Figshare test.",
        ],
    }
    (OUT / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
