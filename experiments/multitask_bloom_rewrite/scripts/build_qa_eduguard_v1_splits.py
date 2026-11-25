"""Split data/train.jsonl (1,600 authored academic QA rows) into exact 1,172 / 146 / 147 train/validation/test files.

Input (read only, never modified):  data/train.jsonl
Output (refuses to overwrite):      data/qa_eduguard_v1/{train,validation,test,reserve}.jsonl + dataset_manifest.json

Rules
  1. Exact duplicate questions (normalised case/punctuation/whitespace) keep the lowest ID; later copies go to reserve
     with reason "duplicate_question".
  2. Leakage groups: rows sharing a normalised topic or a normalised question are one group (union-find), so a group
     never spans two splits.
  3. Groups are shuffled with a fixed seed and assigned greedily: test is filled to 147, then validation to 146, then
     train to 1,172. A group that would overflow a split is skipped for that split. Rows left over go to reserve with
     reason "not_needed_for_target_counts" (kept, not deleted).
  4. Every row keeps all original fields; only "split" is set to the new assignment (the source has "train" for all).
Run from the repository root:  python -I experiments/multitask_bloom_rewrite/scripts/build_qa_eduguard_v1_splits.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "data" / "train.jsonl"
OUT = REPO_ROOT / "data" / "qa_eduguard_v1"
TARGETS = {"test": 147, "validation": 146, "train": 1172}  # fill order: smallest held-out splits first
SPLIT_ORDER = ("train", "validation", "test")
SEED = 20261125
EXPECTED_SOURCE_ROWS = 1600


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split())


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT} already exists; refusing to overwrite.")
    src_sha = sha256(SRC)
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_SOURCE_ROWS:
        sys.exit(f"expected {EXPECTED_SOURCE_ROWS} source rows, found {len(rows)}")
    if len({r["id"] for r in rows}) != len(rows):
        sys.exit("source IDs are not unique")

    # 1. exact duplicate questions -> reserve (keep lowest ID)
    reserve, kept, seen = [], [], {}
    for r in sorted(rows, key=lambda r: r["id"]):
        q = norm(r["question"])
        if q in seen:
            reserve.append({**r, "reserve_reason": "duplicate_question", "duplicate_of": seen[q]})
        else:
            seen[q] = r["id"]
            kept.append(r)

    # 2. leakage groups over shared topic or question
    parent = list(range(len(kept)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, r in enumerate(kept):
        for key in (("topic", norm(r["topic"])), ("question", norm(r["question"]))):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    groups = defaultdict(list)
    for i, r in enumerate(kept):
        groups[find(i)].append(r)
    group_list = sorted(groups.values(), key=lambda g: min(r["id"] for r in g))
    random.Random(SEED).shuffle(group_list)

    # 3. greedy exact fill
    assigned = {s: [] for s in TARGETS}
    remaining = group_list
    for split, target in TARGETS.items():
        leftover = []
        for g in remaining:
            if len(assigned[split]) + len(g) <= target:
                assigned[split].extend(g)
            else:
                leftover.append(g)
        remaining = leftover
        if len(assigned[split]) != target:
            sys.exit(f"could not fill {split} exactly: {len(assigned[split])}/{target}")
    for g in remaining:
        reserve.extend({**r, "reserve_reason": "not_needed_for_target_counts"} for r in g)

    # 4. write
    OUT.mkdir(parents=True)
    files, counts, stats = {}, {}, {}
    for split in (*SPLIT_ORDER, "reserve"):
        out_rows = sorted((assigned[split] if split != "reserve" else reserve), key=lambda r: r["id"])
        out_rows = [{**r, "split": split} for r in out_rows]
        path = OUT / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for r in out_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[f"{split}.jsonl"] = sha256(path)
        counts[split] = len(out_rows)
        stats[split] = {"disciplines": dict(Counter(r["discipline"] for r in out_rows).most_common()),
                        "qa_type": dict(Counter(r["qa_type"] for r in out_rows).most_common()),
                        "difficulty": dict(Counter(r["difficulty"] for r in out_rows).most_common())}
    if sha256(SRC) != src_sha:
        sys.exit("source file changed during the build")

    used = sum(counts[s] for s in SPLIT_ORDER)
    manifest = {
        "version": "qa_eduguard_v1",
        "task": "academic_qa",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"path": "data/train.jsonl", "sha256": src_sha, "rows": len(rows), "modified": False},
        "build_script": "experiments/multitask_bloom_rewrite/scripts/build_qa_eduguard_v1_splits.py",
        "seed": SEED,
        "target_counts": {s: TARGETS[s] for s in SPLIT_ORDER},
        "counts": counts,
        "ratio_of_used_rows": {s: round(counts[s] / used, 4) for s in SPLIT_ORDER},
        "reserve_reasons": dict(Counter(r["reserve_reason"] for r in reserve)),
        "leakage_grouping": "union of rows sharing a normalised topic or normalised question; groups never span splits",
        "group_count": len(group_list),
        "files_sha256": files,
        "split_stats": stats,
        "notes": [
            "All original fields are preserved; only 'split' is reassigned (source rows all say 'train').",
            "reserve.jsonl holds rows not used by train/validation/test; nothing was deleted.",
            "verification is 'pending' for every source row: answers are not independently verified.",
            "This dataset is not wired into dataset_io.py or any training config.",
        ],
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"counts": counts, "reserve_reasons": manifest["reserve_reasons"],
                      "ratio_of_used_rows": manifest["ratio_of_used_rows"]}, indent=1))


if __name__ == "__main__":
    main()
