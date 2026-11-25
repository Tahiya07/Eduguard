"""Maintain data/summarization (EduGuard academic summarization dataset, generated_unverified).

Commands (repository root, project venv; needs scikit-learn):
  rebalance-short   one-off: reassign the 293 Short records (UAP-SUM-0001..0293) to exactly 235/29/29 so that no
                    near-duplicate or same-topic pair crosses splits; validation/test take only 300-600-word passages.
                    Record text is never changed; blank lines are dropped.
  add-batch FILE --category {medium,long,very_long} --split {train,validation,test}
                    validate a staged JSONL batch and append it. Every record must: have exactly the schema fields in
                    order and the fixed metadata; continue the global ID sequence; fall inside the category's word
                    range; have a summary <= 40% of the source length; not repeat an existing topic name; and stay
                    below the near-duplicate threshold against every existing and batch record. Nothing is appended
                    unless the whole batch passes, and no split x category target may be exceeded.
  status            print progress against the plan.
  audit             full audit; writes data/summarization/dataset_manifest.json.

Category of a record: Short = IDs 0001-0293 (the pre-existing block, kept as authored even where under 300 words);
otherwise by source word count: medium 601-1200, long 1201-2500, very_long 2501-4000.
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
DIR = REPO_ROOT / "data" / "summarization"
SPLITS = ("train", "validation", "test")
SCHEMA = ["id", "department", "course", "topic", "source_text", "summary", "source_type", "review_status"]
META = {"source_type": "original_academic_example", "review_status": "generated_unverified"}
SHORT_LAST_ID = 293
RANGES = {"short": (300, 600), "medium": (601, 1200), "long": (1201, 2500), "very_long": (2501, 4000)}
TARGETS = {
    "short": {"train": 235, "validation": 29, "test": 29},
    "medium": {"train": 352, "validation": 44, "test": 44},
    "long": {"train": 410, "validation": 51, "test": 52},
    "very_long": {"train": 175, "validation": 22, "test": 22},
}
ORDER = [(c, s) for c in ("medium", "long", "very_long") for s in SPLITS]
NEAR_DUP = 0.50          # max(char 3-5-gram, word) TF-IDF cosine on source_text
MAX_SUMMARY_RATIO = 0.40
SEED = 20261126


def words(text: str) -> int:
    return len(str(text).split())


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split())


def id_num(rid: str) -> int:
    m = re.fullmatch(r"UAP-SUM-(\d{4})", rid)
    if not m:
        raise ValueError(f"bad id {rid!r}")
    return int(m.group(1))


def is_generated(rid: str) -> bool:
    """UAP-SUM-#### records belong to this dataset's own ID sequence; other IDs are merged external records."""
    return re.fullmatch(r"UAP-SUM-\d{4}", rid) is not None


def sort_key(rid: str) -> tuple:
    return (0, id_num(rid), "") if is_generated(rid) else (1, 0, rid)


def next_generated_id(items) -> int:
    return max(id_num(r["id"]) for _, r in items if is_generated(r["id"])) + 1


def category(row: dict) -> str:
    if is_generated(row["id"]) and id_num(row["id"]) <= SHORT_LAST_ID:
        return "short"
    n = words(row["source_text"])
    for name, (lo, hi) in RANGES.items():
        if name != "short" and lo <= n <= hi:
            return name
    return f"out_of_range({n})"


def load() -> dict[str, list[dict]]:
    out = {}
    for s in SPLITS:
        path = DIR / f"{s}.jsonl"
        out[s] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


def write(rows: dict[str, list[dict]]) -> None:
    for s in SPLITS:
        with (DIR / f"{s}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in sorted(rows[s], key=lambda r: sort_key(r["id"])):
                fh.write(json.dumps({k: r[k] for k in SCHEMA}, ensure_ascii=False) + "\n")


def similarity(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(texts)
    word = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    sim = np.maximum(cosine_similarity(char), cosine_similarity(word))
    np.fill_diagonal(sim, 0.0)
    return sim


def flat(rows: dict[str, list[dict]]) -> list[tuple[str, dict]]:
    return [(s, r) for s in SPLITS for r in rows[s]]


# ---------------------------------------------------------------- rebalance-short
def rebalance_short() -> None:
    import random
    rows = load()
    items = flat(rows)
    if any(id_num(r["id"]) > SHORT_LAST_ID for _, r in items):
        sys.exit("rebalance-short only runs before any non-Short record is added.")
    recs = [r for _, r in items]
    sim = similarity([r["source_text"] for r in recs])
    parent = list(range(len(recs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[str, int] = {}
    for i, r in enumerate(recs):
        key = norm(r["topic"])
        if key in first:
            parent[find(i)] = find(first[key])
        else:
            first[key] = i
    for a, b in zip(*np.where(np.triu(sim >= NEAR_DUP))):
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(recs)):
        groups[find(i)].append(i)
    in_range = lambda i: RANGES["short"][0] <= words(recs[i]["source_text"]) <= RANGES["short"][1]
    eligible = [g for g in groups.values() if len(g) == 1 and in_range(g[0])]
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    # spread departments: round-robin over departments in seeded order
    by_dept: dict[str, list[list[int]]] = defaultdict(list)
    for g in eligible:
        by_dept[recs[g[0]]["department"]].append(g)
    picked: list[int] = []
    need = TARGETS["short"]["validation"] + TARGETS["short"]["test"]
    while len(picked) < need and any(by_dept.values()):
        for d in sorted(by_dept, key=lambda d: (-len(by_dept[d]), d)):
            if by_dept[d] and len(picked) < need:
                picked.append(by_dept[d].pop()[0])
    if len(picked) < need:
        sys.exit(f"only {len(picked)} isolated in-range Short records; cannot fill held-out splits")
    held = sorted(picked, key=lambda i: id_num(recs[i]["id"]))
    rng.shuffle(held)
    val = set(held[: TARGETS["short"]["validation"]])
    test = set(held[TARGETS["short"]["validation"]:])
    new = {s: [] for s in SPLITS}
    for i, r in enumerate(recs):
        new["validation" if i in val else "test" if i in test else "train"].append(r)
    split_of = {i: ("validation" if i in val else "test" if i in test else "train") for i in range(len(recs))}
    worst = max((sim[a, b] for a in range(len(recs)) for b in range(a + 1, len(recs))
                 if split_of[a] != split_of[b]), default=0.0)
    if worst >= NEAR_DUP:
        sys.exit(f"rebalance left a cross-split pair at {worst:.3f}")
    before = {r["id"]: s for s, r in items}
    moved = sorted((r["id"], before[r["id"]], split_of[i]) for i, r in enumerate(recs) if before[r["id"]] != split_of[i])
    write(new)
    (DIR / "short_rebalance_log.json").write_text(json.dumps({
        "date": "2026-11-26", "reason": "Short block was 235/50/8 with near-duplicates across splits; plan is 235/29/29",
        "rule": f"topic-equal or source similarity >= {NEAR_DUP} records stay in one split; validation/test drawn only "
                "from isolated 300-600-word records, round-robin by department; record text unchanged",
        "max_cross_split_similarity": round(float(worst), 4),
        "moved": [{"id": a, "from": b, "to": c} for a, b, c in moved],
    }, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"counts": {s: len(v) for s, v in new.items()}, "moved": len(moved),
                      "max_cross_split_similarity": round(float(worst), 4)}, indent=1))


# ---------------------------------------------------------------- add-batch
def parse_text_batch(text: str) -> list[str]:
    """Plain-text staging format, one block per record:

    === UAP-SUM-0294 | Department | Course | Topic
    SOURCE:
    paragraphs...
    SUMMARY:
    paragraphs...
    """
    lines = []
    for block in re.split(r"^=== ", text, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        parts = [p.strip() for p in header.split("|")]
        if len(parts) != 4 or "SOURCE:" not in body or "SUMMARY:" not in body:
            raise SystemExit(f"malformed block header/body: {header[:80]!r}")
        source, _, summary = body.partition("SUMMARY:")
        source = source.split("SOURCE:", 1)[1]
        clean = lambda t: "\n\n".join(" ".join(p.split()) for p in re.split(r"\n\s*\n", t.strip()) if p.strip())
        rec = dict(zip(SCHEMA, [parts[0], parts[1], parts[2], parts[3], clean(source), clean(summary),
                                META["source_type"], META["review_status"]]))
        lines.append(json.dumps(rec, ensure_ascii=False))
    return lines


def add_batch(path: Path, cat: str, split: str) -> None:
    rows = load()
    items = flat(rows)
    text = path.read_text(encoding="utf-8")
    batch_lines = parse_text_batch(text) if path.suffix == ".txt" else [l for l in text.splitlines() if l.strip()]
    errors: list[str] = []
    batch = []
    for i, line in enumerate(batch_lines, 1):
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {i}: invalid JSON ({exc})")
            continue
        if list(r) != SCHEMA:
            errors.append(f"line {i}: keys {list(r)} != schema")
        batch.append(r)
    if errors:
        sys.exit("batch rejected:\n  " + "\n  ".join(errors))
    next_id = next_generated_id(items)
    have = Counter((category(r), s) for s, r in items)
    remaining = TARGETS[cat][split] - have[(cat, split)]
    if len(batch) > remaining:
        errors.append(f"{len(batch)} records but only {remaining} remain for {cat}/{split}")
    lo, hi = RANGES[cat]
    topics = {norm(r["topic"]): r["id"] for _, r in items}
    for k, r in enumerate(batch):
        rid = r.get("id", "")
        if rid != f"UAP-SUM-{next_id + k:04d}":
            errors.append(f"{rid}: expected UAP-SUM-{next_id + k:04d}")
        for f in SCHEMA:
            if not isinstance(r.get(f), str) or not r[f].strip():
                errors.append(f"{rid}: empty {f}")
        for f, v in META.items():
            if r.get(f) != v:
                errors.append(f"{rid}: {f} must be {v!r}")
        n, m = words(r.get("source_text", "")), words(r.get("summary", ""))
        if not lo <= n <= hi:
            errors.append(f"{rid}: source_text has {n} words, outside {cat} range {lo}-{hi}")
        if m > MAX_SUMMARY_RATIO * n:
            errors.append(f"{rid}: summary {m} words is over {int(MAX_SUMMARY_RATIO * 100)}% of the {n}-word source")
        t = norm(r.get("topic", ""))
        if t in topics:
            errors.append(f"{rid}: topic {r['topic']!r} already used by {topics[t]}")
        topics[t] = rid
        if re.search(r"\*\*|^#|\n#|```", r.get("source_text", "") + r.get("summary", "")):
            errors.append(f"{rid}: Markdown found in text")
    texts = [r["source_text"] for _, r in items] + [r["source_text"] for r in batch]
    sim = similarity(texts)
    base = len(items)
    for k, r in enumerate(batch):
        row = sim[base + k, : base + k]
        j = int(row.argmax()) if len(row) else -1
        if j >= 0 and row[j] >= NEAR_DUP:
            other = items[j][1]["id"] if j < base else batch[j - base]["id"]
            errors.append(f"{r['id']}: near-duplicate of {other} (similarity {row[j]:.2f})")
    if errors:
        sys.exit("batch rejected (nothing appended):\n  " + "\n  ".join(errors))
    rows[split].extend(batch)
    write(rows)
    print(f"appended {len(batch)} {cat} records to {split}: {batch[0]['id']}..{batch[-1]['id']}; "
          f"{remaining - len(batch)} remain for {cat}/{split}")
    print_status(rows)


# ---------------------------------------------------------------- status / audit
def print_status(rows: dict[str, list[dict]] | None = None) -> None:
    rows = rows or load()
    have = Counter((category(r), s) for s, r in flat(rows))
    nxt = next_generated_id(flat(rows))
    print(f"next id: UAP-SUM-{nxt:04d}")
    for c in TARGETS:
        print(f"  {c:10s} " + "  ".join(f"{s}: {have[(c, s)]}/{TARGETS[c][s]}" for s in SPLITS))
    odd = {k: v for k, v in have.items() if k[0] not in TARGETS}
    if odd:
        print("  out-of-range records:", odd)
    for c, s in ORDER:
        if have[(c, s)] < TARGETS[c][s]:
            print(f"next batch: {c}/{s} ({TARGETS[c][s] - have[(c, s)]} remaining)")
            break


def audit() -> None:
    rows = load()
    items = flat(rows)
    ids = [r["id"] for _, r in items]
    nums = sorted(id_num(i) for i in ids if is_generated(i))
    cats = Counter((category(r), s) for s, r in items)
    src_w = [words(r["source_text"]) for _, r in items]
    sum_w = [words(r["summary"]) for _, r in items]
    sim = similarity([r["source_text"] for _, r in items])
    split_arr = np.array([s for s, _ in items])
    cross = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        ia, ib = np.where(split_arr == a)[0], np.where(split_arr == b)[0]
        cross[f"{a}/{b}"] = round(float(sim[np.ix_(ia, ib)].max()), 4) if len(ia) and len(ib) else None
    within = int((np.triu(sim) >= NEAR_DUP).sum())
    topic_dups = sum(v - 1 for v in Counter(norm(r["topic"]) for _, r in items).values() if v > 1)
    short_under = [r["id"] for _, r in items if category(r) == "short" and words(r["source_text"]) < 300]
    complete = all(cats[(c, s)] == TARGETS[c][s] for c in TARGETS for s in SPLITS)
    stats = lambda xs: {"min": min(xs), "max": max(xs), "mean": round(sum(xs) / len(xs), 1),
                        "median": statistics.median(xs)}
    manifest = {
        "dataset": "EduGuard academic summarization (original generated academic examples)",
        "provenance": "generated_unverified: model-generated (Claude) source passages and summaries; not authentic "
                      "course materials and not human-verified",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tool": "experiments/multitask_bloom_rewrite/scripts/summarization_dataset_tool.py",
        "total_records": len(items),
        "unique_ids": len(set(ids)),
        "id_range": [f"UAP-SUM-{nums[0]:04d}", f"UAP-SUM-{nums[-1]:04d}"],
        "id_gaps": sorted(set(range(nums[0], nums[-1] + 1)) - set(nums)),
        "split_counts": {s: len(rows[s]) for s in SPLITS},
        "category_counts": {c: sum(v for (cc, _), v in cats.items() if cc == c) for c in sorted({c for c, _ in cats})},
        "split_by_category": {c: {s: cats[(c, s)] for s in SPLITS} for c in sorted({c for c, _ in cats})},
        "targets": TARGETS,
        "complete": complete,
        "source_word_ranges": RANGES,
        "source_words": stats(src_w),
        "summary_words": stats(sum_w),
        "known_deviations": {
            "short_records_under_300_words": {"count": len(short_under), "ids": short_under,
                                              "decision": "kept as authored by the dataset owner; all are in train"},
        },
        "duplicate_checks": {
            "performed": True,
            "duplicate_ids": len(ids) - len(set(ids)),
            "exact_duplicate_source_text": sum(v - 1 for v in Counter(norm(r["source_text"]) for _, r in items).values() if v > 1),
            "exact_duplicate_summary": sum(v - 1 for v in Counter(norm(r["summary"]) for _, r in items).values() if v > 1),
            "repeated_topic_names": topic_dups,
            "near_duplicate_threshold": NEAR_DUP,
            "near_duplicate_pairs_any_split": within,
            "max_cross_split_source_similarity": cross,
            "method": "max(char 3-5-gram TF-IDF cosine, word TF-IDF cosine) over source_text",
        },
        "schema_check": {"performed": True,
                         "records_with_wrong_keys": sum(list(r) != SCHEMA for _, r in items),
                         "records_with_wrong_metadata": sum(any(r.get(k) != v for k, v in META.items()) for _, r in items)},
        "summary_faithfulness": "model-assisted at generation time only; not independently verified",
        "files_sha256": {f"{s}.jsonl": hashlib.sha256((DIR / f"{s}.jsonl").read_bytes()).hexdigest() for s in SPLITS},
    }
    (DIR / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("total_records", "split_counts", "split_by_category", "complete",
                                               "duplicate_checks")}, indent=1))


def merge_external(suffix: str, exclusions: dict[str, str], log_name: str) -> None:
    """Merge externally supplied split files ({split}{suffix}.jsonl) without duplication.

    Within the external files, records with the same topic or source similarity >= 0.80 are one group and only the
    lowest ID is considered. A candidate is accepted only if its topic is new, its source similarity to every
    existing and already-accepted record is below NEAR_DUP, it is not in the manual concept-overlap exclusions, it
    keeps its own split, and no split x category target is exceeded. Original IDs and provenance metadata are kept;
    fields outside the schema are dropped from the dataset row and preserved in the merge log.
    """
    rows = load()
    items = flat(rows)
    ext = [(s, json.loads(l)) for s in SPLITS
           for l in (DIR / f"{s}{suffix}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    existing_ids = {r["id"] for _, r in items}
    sim = similarity([r["source_text"] for _, r in items] + [r["source_text"] for _, r in ext])
    base = len(items)
    # group near-identical external records
    group_of: dict[int, int] = {}
    reps: list[int] = []
    for i in sorted(range(len(ext)), key=lambda i: ext[i][1]["id"]):
        g = next((j for j in reps if ext[j][0] == ext[i][0] and (
            sim[base + i, base + j] >= 0.80 or norm(ext[i][1]["topic"]) == norm(ext[j][1]["topic"]))), None)
        if g is None:
            reps.append(i)
            group_of[i] = i
        else:
            group_of[i] = g
    have = Counter((category(r), s) for s, r in items)
    topics = {norm(r["topic"]): r["id"] for _, r in items}
    accepted: list[int] = []
    decisions: list[dict] = []
    for i, (s, r) in enumerate(ext):
        if group_of[i] != i:
            decisions.append({"id": r["id"], "split": s, "decision": "rejected",
                              "reason": f"near-identical copy of {ext[group_of[i]][1]['id']} "
                                        f"(similarity {sim[base + i, base + group_of[i]]:.2f})"})
    for i in reps:
        s, r = ext[i]
        reason = None
        if r["id"] in existing_ids:
            reason = "id already present"
        elif r["id"] in exclusions:
            reason = f"concept overlap: {exclusions[r['id']]}"
        elif norm(r["topic"]) in topics:
            reason = f"topic already used by {topics[norm(r['topic'])]}"
        else:
            row = sim[base + i, :base]
            j = int(row.argmax())
            acc = max(((sim[base + i, base + k], ext[k][1]["id"]) for k in accepted), default=(0.0, None))
            if row[j] >= NEAR_DUP:
                reason = f"near-duplicate of existing {items[j][1]['id']} (similarity {row[j]:.2f})"
            elif acc[0] >= NEAR_DUP:
                reason = f"near-duplicate of merged {acc[1]} (similarity {acc[0]:.2f})"
        cat = category(r)
        if reason is None and cat not in TARGETS:
            reason = f"source length {words(r['source_text'])} words outside every category"
        if reason is None and have[(cat, s)] >= TARGETS[cat][s]:
            reason = f"{cat}/{s} target already full"
        if reason:
            decisions.append({"id": r["id"], "split": s, "decision": "rejected", "reason": reason})
            continue
        accepted.append(i)
        have[(cat, s)] += 1
        topics[norm(r["topic"])] = r["id"]
        extra = {k: v for k, v in r.items() if k not in SCHEMA}
        decisions.append({"id": r["id"], "split": s, "decision": "merged", "category": cat,
                          **({"dropped_fields": extra} if extra else {})})
        rows[s].append({k: r[k] for k in SCHEMA})
    write(rows)
    split_arr = [s for s, _ in flat(rows)]
    log = {
        "date": "2026-11-26", "source_files": [f"{s}{suffix}.jsonl" for s in SPLITS],
        "rule": "one record per near-identical group (lowest id); merged only if topic is new and source similarity "
                f"< {NEAR_DUP} to every existing and merged record, excluding manual concept overlaps; splits kept",
        "counts": {"external_records": len(ext), "distinct_groups": len(reps), "merged": len(accepted),
                   "merged_by_split": dict(Counter(ext[i][0] for i in accepted))},
        "decisions": sorted(decisions, key=lambda d: d["id"]),
    }
    (DIR / log_name).write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(log["counts"], indent=1))
    print_status(rows)


def next_topics(n: int, split: str, plan_file: Path) -> None:
    """Next n planned topics for a split not yet in the dataset, round-robin across departments."""
    plan = json.loads(plan_file.read_text(encoding="utf-8"))[split]
    done = {norm(r["topic"]) for _, r in flat(load())}
    nxt = next_generated_id(flat(load()))
    queue = []
    depth = max(len(v) for v in plan.values())
    for k in range(depth):
        for dept in plan:
            if k < len(plan[dept]) and norm(plan[dept][k]) not in done:
                queue.append((dept, plan[dept][k]))
    print(f"{len(queue)} planned {split} topics remain; next ids from UAP-SUM-{nxt:04d}")
    for i, (dept, topic) in enumerate(queue[:n]):
        print(f"UAP-SUM-{nxt + i:04d} | {dept} | {topic}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rebalance-short")
    b = sub.add_parser("add-batch")
    b.add_argument("file", type=Path)
    b.add_argument("--category", required=True, choices=["medium", "long", "very_long"])
    b.add_argument("--split", required=True, choices=list(SPLITS))
    sub.add_parser("status")
    sub.add_parser("audit")
    me = sub.add_parser("merge-external")
    me.add_argument("--suffix", default="_2")
    me.add_argument("--exclude", action="append", default=[], help="ID=reason (manual concept overlap)")
    me.add_argument("--log", default="merge_log_2.json")
    nt = sub.add_parser("next-topics")
    nt.add_argument("n", type=int)
    nt.add_argument("--split", default="train", choices=list(SPLITS))
    nt.add_argument("--plan", type=Path, default=DIR / "topic_plan_medium.json")
    args = ap.parse_args()
    if args.cmd == "rebalance-short":
        rebalance_short()
    elif args.cmd == "add-batch":
        add_batch(args.file, args.category, args.split)
    elif args.cmd == "status":
        print_status()
    elif args.cmd == "merge-external":
        merge_external(args.suffix, dict(e.split("=", 1) for e in args.exclude), args.log)
    elif args.cmd == "next-topics":
        next_topics(args.n, args.split, args.plan)
    else:
        audit()


if __name__ == "__main__":
    main()
