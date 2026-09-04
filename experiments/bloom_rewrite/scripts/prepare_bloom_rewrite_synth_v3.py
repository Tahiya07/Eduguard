#!/usr/bin/env python
"""Build bloom_rewrite_synth_v3 (does not overwrite v1/v2).

Uses Figshare classification questions as a SOURCE POOL only.
Target rewrites are deterministic synthetic exam questions (policy v3).

Writes under:
  data/bloom_rewrite_versions/bloom_rewrite_synth_v3/

Does NOT modify data/bloom_rewrite/ (v2 active corpus).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from bloom_target_policy_v3 import (  # noqa: E402
    BLOOM_LEVELS,
    POLICY_VERSION,
    all_source_target_pairs,
    canonical_level,
    source_is_usable,
    synthesize_rewrite_v3,
    template_inventory,
    validate_rewrite_v3,
)
from grouping import group_questions  # noqa: E402
from paths import FIGSHARE_V1, REWRITE_ARCHIVE_DIR, SEED  # noqa: E402
from prompt_format import build_messages, build_sft_text  # noqa: E402

DATASET_VERSION = "bloom_rewrite_synth_v3"
OUT_DIR = REWRITE_ARCHIVE_DIR / DATASET_VERSION


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_sources(figshare_path: Path) -> pd.DataFrame:
    df = pd.read_csv(figshare_path)
    if "question" not in df.columns or "bloom_level" not in df.columns:
        raise ValueError(f"{figshare_path} must contain question and bloom_level")
    df = df.copy()
    df["source_question"] = df["question"].astype(str).str.strip()
    df["source_bloom_level"] = df["bloom_level"].map(canonical_level)
    df["origin_row"] = df.index.astype(int)
    df["source_file"] = str(figshare_path.relative_to(REPO_ROOT)).replace("\\", "/")
    return df


def assign_splits(group_ids: list[int], seed: int) -> dict[int, str]:
    """70/15/15 by source group (same proportions as v2 pipeline intent)."""
    rng = random.Random(seed)
    unique = sorted(set(group_ids))
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    mapping: dict[int, str] = {}
    for i, gid in enumerate(unique):
        if i < n_train:
            mapping[gid] = "train"
        elif i < n_train + n_val:
            mapping[gid] = "validation"
        else:
            mapping[gid] = "test"
    return mapping


def near_duplicate_rate(texts: list[str], threshold: float = 0.92) -> float:
    """Approximate near-duplicate rate via normalized Jaccard on word sets (O(n^2) capped)."""
    if len(texts) < 2:
        return 0.0
    sample = texts if len(texts) <= 800 else texts[:800]
    toks = [set(re_tokenize(t)) for t in sample]
    near = 0
    pairs = 0
    for i in range(len(toks)):
        for j in range(i + 1, min(len(toks), i + 40)):
            pairs += 1
            a, b = toks[i], toks[j]
            if not a or not b:
                continue
            sim = len(a & b) / len(a | b)
            if sim >= threshold:
                near += 1
    return round(near / pairs, 6) if pairs else 0.0


def re_tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bloom_rewrite_synth_v3")
    parser.add_argument("--figshare", default=str(FIGSHARE_V1))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--freeze-test-from",
        default=str(REPO_ROOT / "data" / "bloom_rewrite" / "test.jsonl"),
        help="Optional: reuse exact source groups from existing v2 Bloom test (preferred for multitask freeze)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    figshare = Path(args.figshare)
    if not figshare.is_absolute():
        figshare = REPO_ROOT / figshare

    df = load_sources(figshare)
    keep = []
    reject = Counter()
    for _, row in df.iterrows():
        ok, reason = source_is_usable(row["source_question"], row["source_bloom_level"])
        if ok:
            keep.append(row)
        else:
            reject[reason] += 1
    sources = pd.DataFrame(keep).reset_index(drop=True)
    sources["source_id"] = [
        f"src_{i:05d}_{sha256_text(q)[:10]}"
        for i, q in enumerate(sources["source_question"].tolist())
    ]

    groups = group_questions(sources["source_question"].tolist())
    sources["group_id"] = groups
    split_map = assign_splits(groups, args.seed)

    frozen_questions: set[str] = set()
    freeze_path = Path(args.freeze_test_from)
    if not freeze_path.is_absolute():
        freeze_path = REPO_ROOT / freeze_path
    if freeze_path.is_file():
        from grouping import normalize_question

        for line in freeze_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("source_question")
            if q:
                frozen_questions.add(normalize_question(q))
        for i, q in enumerate(sources["source_question"].tolist()):
            if normalize_question(q) in frozen_questions:
                split_map[int(sources.loc[i, "group_id"])] = "test"

    sources["split"] = sources["group_id"].map(split_map)

    rows: list[dict] = []
    fail_counts = Counter()
    template_counts: dict[str, Counter] = defaultdict(Counter)
    for _, src in sources.iterrows():
        src_level = src["source_bloom_level"]
        for tgt in BLOOM_LEVELS:
            if tgt == src_level:
                continue
            try:
                rewrite, meta = synthesize_rewrite_v3(
                    src["source_question"], src_level, tgt, src["source_id"]
                )
                validation = validate_rewrite_v3(src["source_question"], tgt, rewrite)
                if not validation.ok:
                    fail_counts[validation.failure_category or "fail"] += 1
                    continue
                example_id = sha256_text(
                    f"{src['source_id']}|{src_level}|{tgt}|{rewrite}|v3"
                )[:16]
                messages = build_messages(src["source_question"], tgt, rewrite)
                # Strengthen assistant-side instruction consistency via SFT text from prompt_format
                text = build_sft_text(src["source_question"], tgt, rewrite)
                rows.append(
                    {
                        "example_id": example_id,
                        "source_id": src["source_id"],
                        "group_id": int(src["group_id"]),
                        "split": src["split"],
                        "source_question": src["source_question"],
                        "source_bloom_level": src_level,
                        "target_bloom_level": tgt,
                        "target_rewrite": rewrite,
                        "transformation_type": f"{src_level}->{tgt}",
                        "synthetic_or_original": "synthetic",
                        "synthetic": True,
                        "dataset_version": DATASET_VERSION,
                        "policy_version": POLICY_VERSION,
                        "quality_status": "pass",
                        "topic_overlap": validation.topic_overlap,
                        "source_file": src["source_file"],
                        "generator_inputs": ["source_question", "target_bloom_level"],
                        "template_index": meta["template_index"],
                        "messages": messages,
                        "text": text,
                    }
                )
                template_counts[tgt][meta["template_index"]] += 1
            except Exception as exc:  # noqa: BLE001
                fail_counts[type(exc).__name__] += 1

    # Balance target levels approximately within each split by capped sampling.
    balanced: list[dict] = []
    by_split_level: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_split_level[(r["split"], r["target_bloom_level"])].append(r)
    rng = random.Random(args.seed)
    for split in ("train", "validation", "test"):
        level_lists = [by_split_level[(split, lvl)] for lvl in BLOOM_LEVELS]
        if not any(level_lists):
            continue
        target_n = min(len(x) for x in level_lists if x) if any(level_lists) else 0
        for lvl, items in zip(BLOOM_LEVELS, level_lists):
            rng.shuffle(items)
            balanced.extend(items[:target_n])

    # Prefer balanced set; if empty fall back to all accepted rows.
    final_rows = balanced if balanced else rows
    rng.shuffle(final_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        subset = [r for r in final_rows if r["split"] == split]
        write_jsonl(out_dir / f"{split}.jsonl", subset)

    # Corpus hash over sorted example payloads (stable).
    payload = "\n".join(
        json.dumps(
            {
                "example_id": r["example_id"],
                "source_id": r["source_id"],
                "target_bloom_level": r["target_bloom_level"],
                "target_rewrite": r["target_rewrite"],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        for r in sorted(final_rows, key=lambda x: x["example_id"])
    )
    corpus_hash = sha256_text(payload)

    counts = {
        split: {
            "total": sum(1 for r in final_rows if r["split"] == split),
            "by_target": dict(
                Counter(r["target_bloom_level"] for r in final_rows if r["split"] == split)
            ),
            "by_source_target": dict(
                Counter(r["transformation_type"] for r in final_rows if r["split"] == split)
            ),
        }
        for split in ("train", "validation", "test")
    }
    rewrites = [r["target_rewrite"] for r in final_rows]
    stats = {
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "seed": args.seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_figshare": str(figshare.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_figshare_sha256": sha256_file(figshare) if figshare.is_file() else None,
        "corpus_hash": corpus_hash,
        "counts": counts,
        "rejected_sources": dict(reject),
        "synthesis_failures": dict(fail_counts),
        "template_inventory": template_inventory(),
        "template_usage": {k: dict(v) for k, v in template_counts.items()},
        "prefix_distribution": dict(
            Counter(
                (r["target_rewrite"].split()[0].lower() if r["target_rewrite"].split() else "")
                for r in final_rows
            ).most_common(40)
        ),
        "mean_rewrite_chars": round(sum(len(t) for t in rewrites) / max(1, len(rewrites)), 2),
        "near_duplicate_rate_approx": near_duplicate_rate(rewrites),
        "frozen_test_questions": len(frozen_questions),
        "n_frozen_test_questions": len(frozen_questions),
        "notes": [
            "Synthetic supervision only; not human gold.",
            "Does not overwrite bloom_rewrite_synth_v1/v2 or data/bloom_rewrite.",
            "Source Bloom level is metadata only in generator prompts.",
            "For multitask comparison, prepare_multitask_dataset_v3.py freezes the exact multitask test.jsonl.",
        ],
    }
    (out_dir / "dataset_statistics.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    manifest = {
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "corpus_hash": corpus_hash,
        "seed": args.seed,
        "paths": {
            "train": str((out_dir / "train.jsonl").relative_to(REPO_ROOT)).replace("\\", "/"),
            "validation": str((out_dir / "validation.jsonl").relative_to(REPO_ROOT)).replace("\\", "/"),
            "test": str((out_dir / "test.jsonl").relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "counts": {k: v["total"] for k, v in counts.items()},
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Wrote", out_dir)
    print("corpus_hash", corpus_hash)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
