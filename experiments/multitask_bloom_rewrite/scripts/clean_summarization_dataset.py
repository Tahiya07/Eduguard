"""Clean the owner-edited data/summarization splits into a training-ready copy (input files are never modified).

Usage (repository root, project venv; needs scikit-learn):
  .venv/Scripts/python.exe -I experiments/multitask_bloom_rewrite/scripts/clean_summarization_dataset.py \
      [--source data/summarization] [--out data/summarization_v2]

Rules, applied in order; every dropped row is logged with its reason in <out>/cleaning_log.json:
  1. blank lines are skipped; rows already rendered for SFT (have sft_text) are copies of raw rows and are dropped
     (24 of them sat in validation/test while the same ID was in train: train/held-out leakage)
  2. arXiv rows are dropped (source_type arxiv_scientific_article or arxiv_* IDs)
  3. a summary must be shorter than MAX_SUMMARY_RATIO of its source and must not be an extractive lead
     (summary text equal to the beginning of the source)
  4. template padding: a source sentence is boilerplate when, with the topic and capitalised words masked, it occurs
     in >= BOILER_MIN_ROWS distinct sources; rows whose boilerplate share is > MAX_BOILER are dropped
  5. exact duplicate source/summary and repeated IDs: the first clean occurrence is kept
  6. ids are normalised to UAP-SUM-#### (zero padded); a normalised id clash is resolved by keeping the first row
  7. a new row is dropped if its topic repeats an existing topic or its source is a near duplicate (>= NEAR_DUP) of a row in another split
     (same-split near duplicates on related topics are allowed, as in v1); no near duplicate may cross splits
Kept rows are written with the fixed schema; rows the owner added without metadata get
source_type=original_academic_example, review_status=owner_added_unverified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "validation", "test")
SCHEMA = ["id", "department", "course", "topic", "source_text", "summary", "source_type", "review_status"]
NEW_META = {"source_type": "original_academic_example", "review_status": "owner_added_unverified"}
MAX_SUMMARY_RATIO = 0.40
BOILER_MIN_ROWS = 5
MAX_BOILER = 0.05
NEAR_DUP = 0.50
DEPT_ALIASES = {"Civil": "Civil Engineering", "ME": "Mechanical Engineering", "EEE": "EEE", "CSE": "CSE"}
RANGES = {"short": (0, 600), "medium": (601, 1200), "long": (1201, 2500), "very_long": (2501, 10**6)}
PLAN = {"train": 1172, "validation": 146, "test": 147}


def words(t: str) -> int:
    return len(str(t).split())


def norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(t).lower()).split())


def sentences(t: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", t) if len(x.split()) >= 4]


def masked(sentence: str, topic: str) -> str:
    s = sentence.replace(topic, "T").replace(topic.lower(), "T")
    return re.sub(r"\b[A-Z][\w-]*\b", "X", s)


def category(n: int) -> str:
    return next(k for k, (lo, hi) in RANGES.items() if lo <= n <= hi)


def similarity(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(texts)
    word = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    sim = np.maximum(cosine_similarity(char), cosine_similarity(word))
    np.fill_diagonal(sim, 0.0)
    return sim


def fix_id(rid: str) -> str:
    m = re.fullmatch(r"UAP-SUM-(\d{1,4})", rid)
    return f"UAP-SUM-{int(m.group(1)):04d}" if m else rid


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/summarization"))
    ap.add_argument("--out", type=Path, default=Path("data/summarization_v2"))
    args = ap.parse_args()
    src, out = REPO_ROOT / args.source, REPO_ROOT / args.out
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")

    rows = []  # (split, line, row)
    for s in SPLITS:
        for i, line in enumerate((src / f"{s}.jsonl").read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                rows.append((s, i, json.loads(line)))
    freq = Counter()
    seen_src = set()
    for _, _, r in rows:
        if r["source_text"] not in seen_src:
            seen_src.add(r["source_text"])
            freq.update({masked(x, r["topic"]) for x in sentences(r["source_text"])})

    log, kept = [], []
    drop = lambda s, i, r, reason: log.append({"split": s, "line": i, "id": r.get("id"), "topic": r.get("topic"),
                                                "reason": reason})
    for s, i, r in rows:
        if "sft_text" in r:
            drop(s, i, r, "rendered SFT copy of a raw row (duplicate)"); continue
        if str(r.get("source_type", "")).startswith("arxiv") or str(r.get("id", "")).startswith("arxiv"):
            drop(s, i, r, "arXiv row"); continue
        n, m = words(r["source_text"]), words(r["summary"])
        if m > MAX_SUMMARY_RATIO * n:
            drop(s, i, r, f"summary {m} words > {MAX_SUMMARY_RATIO:.0%} of {n}-word source"); continue
        if norm(r["source_text"]).startswith(norm(r["summary"])[:300]):
            drop(s, i, r, "summary is a verbatim lead of the source (extractive copy)"); continue
        ss = sentences(r["source_text"])
        boiler = sum(freq[masked(x, r["topic"])] >= BOILER_MIN_ROWS for x in ss) / max(1, len(ss))
        if boiler > MAX_BOILER:
            drop(s, i, r, f"template padding: {boiler:.0%} of source sentences are boilerplate repeated across "
                          f">= {BOILER_MIN_ROWS} rows"); continue
        kept.append((s, i, r))

    final: dict[str, list[dict]] = {k: [] for k in SPLITS}
    ids, srcs, sums, topics = {}, set(), set(), {}
    added = []
    for s, i, r in kept:
        rid = fix_id(r["id"])
        if rid in ids:
            drop(s, i, r, f"id {rid} already used ({ids[rid]})"); continue
        if norm(r["source_text"]) in srcs or norm(r["summary"]) in sums:
            drop(s, i, r, "exact duplicate source or summary"); continue
        is_new = "source_type" not in r
        if is_new and norm(r["topic"]) in topics:
            drop(s, i, r, f"topic already used by {topics[norm(r['topic'])]}"); continue
        rec = {k: r.get(k) for k in SCHEMA}
        rec["id"] = rid
        if is_new:
            rec.update(NEW_META)
            rec["department"] = DEPT_ALIASES.get(rec["department"], rec["department"])
            added.append(rid)
        ids[rid] = s
        srcs.add(norm(r["source_text"])); sums.add(norm(r["summary"]))
        topics.setdefault(norm(r["topic"]), rid)
        final[s].append(rec)

    # near-duplicate check: new rows against everything kept; nothing may cross splits
    flat = [(s, r) for s in SPLITS for r in final[s]]
    sim = similarity([r["source_text"] for _, r in flat])
    remove = set()
    for a, (sa, ra) in enumerate(flat):
        if ra["id"] not in added:
            continue
        for b, (sb, rb) in enumerate(flat):
            if b != a and sb != sa and sim[a, b] >= NEAR_DUP and (rb["id"] not in added or b < a) and rb["id"] not in remove:
                remove.add(ra["id"])
                log.append({"split": sa, "id": ra["id"], "topic": ra["topic"],
                            "reason": f"near duplicate of {rb['id']} (similarity {sim[a, b]:.2f})"})
                break
    for s in SPLITS:
        final[s] = [r for r in final[s] if r["id"] not in remove]
    flat = [(s, r) for s in SPLITS for r in final[s]]
    sim = similarity([r["source_text"] for _, r in flat])
    arr = np.array([s for s, _ in flat])
    cross = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        cross[f"{a}/{b}"] = round(float(sim[np.ix_(arr == a, arr == b)].max()), 4)
    if max(cross.values()) >= NEAR_DUP:
        sys.exit(f"near duplicate crosses splits: {cross}")

    out.mkdir(parents=True)
    hashes = {}
    for s in SPLITS:
        p = out / f"{s}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            for r in sorted(final[s], key=lambda r: r["id"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    cats = {s: dict(Counter(category(words(r["source_text"])) for r in final[s])) for s in SPLITS}
    reasons = Counter(re.sub(r"\d+(\.\d+)?%?|UAP-\S+", "#", d["reason"]) for d in log)
    manifest = {
        "dataset": "EduGuard academic summarization v2 (cleaned from the owner-edited data/summarization)",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tool": "experiments/multitask_bloom_rewrite/scripts/clean_summarization_dataset.py",
        "input_sha256": {f"{s}.jsonl": hashlib.sha256((src / f"{s}.jsonl").read_bytes()).hexdigest() for s in SPLITS},
        "input_rows": dict(Counter(s for s, _, _ in rows)),
        "split_counts": {s: len(final[s]) for s in SPLITS},
        "plan": PLAN, "shortfall_vs_plan": {s: PLAN[s] - len(final[s]) for s in SPLITS},
        "length_category_by_split": cats,
        "source_type_by_split": {s: dict(Counter(r["source_type"] for r in final[s])) for s in SPLITS},
        "review_status_counts": dict(Counter(r["review_status"] for _, r in flat)),
        "new_rows_kept": sorted(set(added) - remove),
        "dropped_rows": len(log), "dropped_by_reason": dict(reasons.most_common()),
        "checks": {"unique_ids": len({r["id"] for _, r in flat}) == len(flat),
                   "exact_duplicate_source": len(flat) - len({norm(r["source_text"]) for _, r in flat}),
                   "exact_duplicate_summary": len(flat) - len({norm(r["summary"]) for _, r in flat}),
                   "max_cross_split_similarity": cross, "near_duplicate_threshold": NEAR_DUP,
                   "max_summary_ratio": round(max(words(r["summary"]) / words(r["source_text"]) for _, r in flat), 3),
                   "boilerplate_rule": {"min_rows": BOILER_MIN_ROWS, "max_share": MAX_BOILER}},
        "source_words": {s: {"median": statistics.median(words(r["source_text"]) for r in final[s]),
                             "max": max(words(r["source_text"]) for r in final[s])} for s in SPLITS},
        "files_sha256": hashes,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    (out / "cleaning_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("input_rows", "split_counts", "shortfall_vs_plan",
                                               "length_category_by_split", "dropped_by_reason", "checks")}, indent=1))


if __name__ == "__main__":
    main()
