"""Build summarization v3.1 from v3: rebalance thin train departments with OpenStax rows, keeping every split size.

Usage (repository root, project venv; needs scikit-learn):
  python -I experiments/multitask_bloom_rewrite/scripts/topup_summarization_v3_1.py CANDIDATES.jsonl \
      [--src data/summarization_v3] [--out data/summarization_v3_1]

Why: the v3 evaluation showed degenerate repetition loops on Biology sections, a department with 6 train rows but
9 validation rows. Departments whose train count is below TRAIN_PER_HELDOUT x their validation count are topped up.
Validation and test are copied unchanged. For every row added to train, one OpenStax train row of the same length
category is removed from the largest departments (DONOR_DEPARTMENTS), so split sizes and length categories stay at
the plan (1,172 / 146 / 147). Owner-written (v2) rows are never removed.

An added candidate must pass the v3 candidate filter and
  * not come from a chapter that has a validation/test row (chapter quarantine),
  * have TF-IDF source similarity < UNIT_LINK (0.45) to every validation/test row, and
  * have similarity < NEAR_DUP (0.50) to every train row and to every other added row.
The final re-fitted cross-split check requires every cross-split pair < NEAR_DUP, as in v3.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_summarization_v3 import (  # noqa: E402
    DEPARTMENT, NEAR_DUP, RANGES, REPO_ROOT, SPLITS, UNIT_LINK, category, norm, similarity, words,
)

TRAIN_PER_HELDOUT = 5   # target train rows per validation row of a department
MAX_TARGET = 60
DONOR_DEPARTMENTS = ("Business Administration", "Psychology")
PER_CHAPTER = 8  # these books have few chapter labels (e.g. 3-16 per department)
SEED = 20261127


def eligible(r: dict) -> str | None:
    n, m = words(r["source_text"]), words(r["summary"])
    cat = category(n)
    if cat in (None, "short") or m < 40 or not 0.025 * n <= m <= 0.40 * n:
        return None
    if norm(r["source_text"]).startswith(norm(r["summary"])[:300]):
        return None
    if re.search(r"\*\*|(^|\n)#|```", r["source_text"] + r["summary"]) or r["book"] not in DEPARTMENT:
        return None
    return cat


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", type=Path)
    ap.add_argument("--src", type=Path, default=Path("data/summarization_v3"))
    ap.add_argument("--out", type=Path, default=Path("data/summarization_v3_1"))
    args = ap.parse_args()
    src, out = REPO_ROOT / args.src, REPO_ROOT / args.out
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    data = {s: [json.loads(l) for l in (src / f"{s}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            for s in SPLITS}
    cand_all = [json.loads(l) for l in args.candidates.read_text(encoding="utf-8").splitlines() if l.strip()]

    train_dep = Counter(r["department"] for r in data["train"])
    val_dep = Counter(r["department"] for r in data["validation"])
    osx_deps = set(DEPARTMENT.values())
    targets = {d: min(MAX_TARGET, TRAIN_PER_HELDOUT * v) for d, v in val_dep.items()
               if d in osx_deps and d not in DONOR_DEPARTMENTS and train_dep[d] < TRAIN_PER_HELDOUT * v}
    need = {d: t - train_dep[d] for d, t in targets.items()}
    print("department top-up targets (train now -> target):", {d: (train_dep[d], t) for d, t in targets.items()})

    used_refs = {r.get("source_ref", "").split(":modules/")[-1] + "|" + r.get("source_ref", "").split("@")[0]
                 for s in SPLITS for r in data[s]}
    used_text = {norm(r["source_text"]) for s in SPLITS for r in data[s]}
    held_chapters = {(r["course"], r.get("chapter") or "") for s in ("validation", "test") for r in data[s]
                     if r["source_type"] == "openstax_textbook_section"}
    pool = []
    for c in cand_all:
        dep = DEPARTMENT.get(c["book"])
        if dep not in need:
            continue
        cat = eligible(c)
        if not cat or (c["book"], c["chapter"] or "") in held_chapters:
            continue
        if f"{c['module']}|{c['repo']}" in used_refs or norm(c["source_text"]) in used_text:
            continue
        pool.append({**c, "category": cat, "department": dep})
    print("eligible candidates in thin departments:", Counter(p["department"] for p in pool))

    # similarity of candidates to the existing splits and to each other
    existing = [(s, r) for s in SPLITS for r in data[s]]
    sim = similarity([r["source_text"] for _, r in existing] + [p["source_text"] for p in pool])
    E = len(existing)
    split_arr = np.array([s for s, _ in existing])
    to_held = sim[E:, :E][:, split_arr != "train"].max(axis=1)
    to_train = sim[E:, :E][:, split_arr == "train"].max(axis=1)
    ok = [i for i in range(len(pool)) if to_held[i] < UNIT_LINK and to_train[i] < NEAR_DUP]
    print("pass leakage/near-duplicate checks:", len(ok), "of", len(pool))

    rng = random.Random(SEED)
    by_dep = defaultdict(list)
    for i in ok:
        by_dep[pool[i]["department"]].append(i)
    added: list[int] = []
    for d in sorted(by_dep):
        items = sorted(by_dep[d], key=lambda i: (pool[i]["book"], pool[i]["order"]))
        rng.shuffle(items)
        # spread over books and chapters: at most PER_CHAPTER rows per chapter
        per_chapter = Counter()
        for i in items:
            if need[d] <= 0:
                break
            key = (pool[i]["book"], pool[i]["chapter"])
            if per_chapter[key] >= PER_CHAPTER or any(sim[E + i, E + j] >= NEAR_DUP for j in added):
                continue
            added.append(i)
            per_chapter[key] += 1
            need[d] -= 1
    print("added per department:", Counter(pool[i]["department"] for i in added), "unfilled:", {d: n for d, n in need.items() if n > 0})

    # remove the same number of OpenStax train rows from donor departments, same length category
    add_cats = Counter(pool[i]["category"] for i in added)
    donors = defaultdict(list)
    for r in data["train"]:
        if r["department"] in DONOR_DEPARTMENTS and r["source_type"] == "openstax_textbook_section":
            donors[category(words(r["source_text"]))].append(r)
    removed = []
    for cat, n in sorted(add_cats.items()):
        cands = sorted(donors[cat], key=lambda r: r["id"])
        rng.shuffle(cands)
        by_d = {d: [r for r in cands if r["department"] == d] for d in DONOR_DEPARTMENTS}
        # always take from the donor that currently has the most train rows
        while n > 0 and any(by_d.values()):
            d = max(by_d, key=lambda d: (train_dep[d] - sum(x["department"] == d for x in removed)) if by_d[d] else -1)
            removed.append(by_d[d].pop())
            n -= 1
        if n:
            sys.exit(f"not enough donor rows in category {cat}")
    removed_ids = {r["id"] for r in removed}

    last = max(int(r["id"].split("-")[-1]) for s in SPLITS for r in data[s] if r["id"].startswith("OSX-SUM-"))
    new_rows = []
    for k, i in enumerate(sorted(added, key=lambda i: (pool[i]["book"], pool[i]["order"])), 1):
        c = pool[i]
        new_rows.append({
            "id": f"OSX-SUM-{last + k:04d}", "department": c["department"], "course": c["book"], "topic": c["section"],
            "source_text": c["source_text"], "summary": c["summary"], "source_type": "openstax_textbook_section",
            "review_status": "public_author_written_summary", "chapter": c["chapter"], "license": c["license"],
            "source_ref": f"{c['repo']}@{c['commit'][:12]}:modules/{c['module']}",
        })
    final = {"train": [r for r in data["train"] if r["id"] not in removed_ids] + new_rows,
             "validation": data["validation"], "test": data["test"]}

    # verification (re-fitted, as in v3)
    flat = [(s, r) for s in SPLITS for r in final[s]]
    assert len({r["id"] for _, r in flat}) == len(flat)
    assert len({norm(r["source_text"]) for _, r in flat}) == len(flat)
    assert len({norm(r["summary"]) for _, r in flat}) == len(flat)
    fsim = similarity([r["source_text"] for _, r in flat])
    arr = np.array([s for s, _ in flat])
    cross = {f"{a}/{b}": round(float(fsim[np.ix_(arr == a, arr == b)].max()), 4)
             for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    if max(cross.values()) >= NEAR_DUP:
        sys.exit(f"near duplicate crosses splits: {cross}")
    counts = {s: len(final[s]) for s in SPLITS}
    old_counts = {s: len(data[s]) for s in SPLITS}
    assert counts == old_counts, (counts, old_counts)
    cats = {s: {c: sum(category(words(r["source_text"])) == c for r in final[s]) for c in RANGES} for s in SPLITS}
    old_cats = {s: {c: sum(category(words(r["source_text"])) == c for r in data[s]) for c in RANGES} for s in SPLITS}
    assert cats == old_cats

    out.mkdir(parents=True)
    hashes = {}
    for s in SPLITS:
        p = out / f"{s}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            for r in final[s]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((src / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "dataset": "EduGuard academic summarization v3.1 (v3 with thin train departments rebalanced)",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tools": manifest.get("tools", []) + ["experiments/multitask_bloom_rewrite/scripts/topup_summarization_v3_1.py"],
        "derived_from": {"path": args.src.as_posix(), "files_sha256": manifest.get("files_sha256")},
        "rebalance": {
            "reason": "repetition loops on Biology validation sections; Biology had 6 train rows vs 9 validation rows",
            "rule": f"departments with train < {TRAIN_PER_HELDOUT} x validation rows topped up to that (max {MAX_TARGET}); "
                    f"equal number of same-category OpenStax train rows removed from {list(DONOR_DEPARTMENTS)}",
            "added_ids": [r["id"] for r in new_rows], "removed_ids": sorted(removed_ids),
            "added_by_department": dict(Counter(r["department"] for r in new_rows)),
            "removed_by_department": dict(Counter(r["department"] for r in removed)),
            "validation_and_test": "unchanged (byte-identical rows)", "seed": SEED,
        },
        "split_counts": counts, "length_category_by_split": cats,
        "source_type_by_split": {s: dict(Counter(r["source_type"] for r in final[s])) for s in SPLITS},
        "department_by_split": {s: dict(Counter(r["department"] for r in final[s]).most_common()) for s in SPLITS},
        "files_sha256": hashes,
    })
    manifest["checks"]["max_cross_split_similarity"] = cross
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split_counts": counts, "cross": cross, "rebalance": {k: v for k, v in manifest["rebalance"].items()
                                                                          if k.endswith("department")}}, indent=1))
    print("train departments:", dict(Counter(r["department"] for r in final["train"]).most_common()))


if __name__ == "__main__":
    main()
