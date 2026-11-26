"""Build EduGuard summarization v3: the clean v2 set topped up with OpenStax textbook sections to the planned sizes.

Usage (repository root, project venv; needs scikit-learn):
  1. shallow-clone the OpenStax repositories (collections/ + modules/ only) into RAW_DIR; the repositories and
     commits used are recorded in <out>/dataset_manifest.json ("openstax.repositories")
  2. python -I experiments/multitask_bloom_rewrite/scripts/import_openstax_summarization.py RAW_DIR CANDIDATES.jsonl
  3. python -I experiments/multitask_bloom_rewrite/scripts/build_summarization_v3.py CANDIDATES.jsonl \
         [--v2 data/summarization_v2] [--out data/summarization_v3]

Plan (rows per split x length category; length = whitespace words of source_text):
  short <= 600 | medium 601-1200 | long 1201-2500 | very_long 2501-4000
  train 1,172 / validation 146 / test 147 = 1,465. All v2 rows are kept in their split; OpenStax fills the rest.

OpenStax candidate filter: medium..very_long source; summary >= 40 words, between 2.5% and 40% of the source and not
a verbatim lead of it; no Markdown. Similarity = max(char 3-5-gram, word) TF-IDF cosine of source_text (the v1/v2
measure). Leakage control:
  * candidates with similarity >= UNIT_LINK (0.45) to any v2 row are excluded
  * units = connected components of candidates linked at >= UNIT_LINK; a unit lies in one split and contributes one
    row per near-duplicate (>= NEAR_DUP) cluster
  * validation/test are drawn chapter by chapter, round-robin over books (<= PER_CHAPTER rows per chapter and
    PER_BOOK per book and split); every chapter touching a held-out unit is quarantined from all other splits;
    unused held-out rows are discarded, never moved to train
  * the final re-fitted check requires every cross-split pair < NEAR_DUP (0.50)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "validation", "test")
SCHEMA = ["id", "department", "course", "topic", "source_text", "summary", "source_type", "review_status"]
RANGES = {"short": (0, 600), "medium": (601, 1200), "long": (1201, 2500), "very_long": (2501, 4000)}
PLAN = {"train": 1172, "validation": 146, "test": 147}
# per split x category; train short = all clean short rows (240), so train very_long absorbs the difference
TARGETS = {
    "validation": {"short": 29, "medium": 44, "long": 51, "very_long": 22},
    "test": {"short": 29, "medium": 44, "long": 52, "very_long": 22},
    "train": {"medium": 352, "long": 410},  # short: whatever v2 has; very_long: remainder to reach PLAN
}
NEAR_DUP = 0.50
PER_CHAPTER = 4
PER_BOOK = 8
UNIT_LINK = 0.45  # units are linked below NEAR_DUP so the final re-fitted check keeps a margin
SEED = 20261126
DEPARTMENT = {
    "Additive Manufacturing Essentials": "Mechanical Engineering",
    "American Government 4e": "Political Science", "Introduction to Political Science": "Political Science",
    "Anatomy and Physiology 2e": "Biomedical Science", "Introduction to Behavioral Neuroscience": "Psychology",
    "Astronomy 2e": "Physics", "Biology 2e": "Biology", "Biology for AP® Courses 2e": "Biology",
    "Concepts of Biology": "Biology", "Microbiology": "Microbiology",
    "Business Ethics": "Business Administration", "Business Law I Essentials 2e": "Law",
    "Introduction to Intellectual Property": "Law",
    "Chemistry 2e": "Chemistry", "Chemistry: Atoms First 2e": "Chemistry", "Organic Chemistry": "Chemistry",
    "Entrepreneurship": "Business Administration", "Introduction to Business 2e": "Business Administration",
    "Organizational Behavior": "Business Administration", "Principles of Management": "Business Administration",
    "Principles of Marketing": "Business Administration",
    "Foundations of Information Systems": "CSE", "Principles of Data Science": "CSE",
    "Introduction to Python Programming": "CSE",
    "Introduction to Anthropology": "Sociology", "Introduction to Sociology 3e": "Sociology",
    "Introduction to Philosophy 2e": "Philosophy", "Lifespan Development": "Psychology", "Psychology 2e": "Psychology",
    "Population Health for Nurses": "Public Health", "Nutrition for Nurses": "Public Health",
    "Pharmacology for Nurses": "Pharmacy",
    "Principles of Accounting, Volume 1: Financial Accounting": "Accounting and Finance",
    "Principles of Accounting, Volume 2: Managerial Accounting": "Accounting and Finance",
    "Principles of Finance 2e": "Accounting and Finance",
    "Principles of Economics 3e": "Economics", "Principles of Microeconomics for AP® Courses 2e": "Economics",
    "Principles of Macroeconomics 3e": "Economics", "Principles of Microeconomics 3e": "Economics",
    "Principles of Macroeconomics for AP® Courses 2e": "Economics",
    "U.S. History": "History", "World History Volume 1, to 1500": "History", "World History Volume 2, from 1400": "History",
}


def words(t: str) -> int:
    return len(str(t).split())


def norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(t).lower()).split())


def category(n: int) -> str | None:
    return next((k for k, (lo, hi) in RANGES.items() if lo <= n <= hi), None)


def similarity(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32).fit_transform(texts)
    word = TfidfVectorizer(stop_words="english", sublinear_tf=True, dtype=np.float32).fit_transform(texts)
    sim = np.maximum(cosine_similarity(char), cosine_similarity(word))
    np.fill_diagonal(sim, 0.0)
    return sim


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", type=Path)
    ap.add_argument("--v2", type=Path, default=Path("data/summarization_v2"))
    ap.add_argument("--out", type=Path, default=Path("data/summarization_v3"))
    args = ap.parse_args()
    v2dir, out = REPO_ROOT / args.v2, REPO_ROOT / args.out
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    v2 = {s: [json.loads(l) for l in (v2dir / f"{s}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
          for s in SPLITS}
    cand_all = [json.loads(l) for l in args.candidates.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 1. candidate filter
    rejected = Counter()
    cand = []
    for r in cand_all:
        n, m = words(r["source_text"]), words(r["summary"])
        cat = category(n)
        if cat in (None, "short"):
            rejected["source length outside medium..very_long"] += 1
        elif m < 40:
            rejected["summary < 40 words"] += 1
        elif not 0.025 * n <= m <= 0.40 * n:
            rejected["summary/source ratio outside 2.5-40%"] += 1
        elif norm(r["source_text"]).startswith(norm(r["summary"])[:300]):
            rejected["summary is a lead of the source"] += 1
        elif re.search(r"\*\*|(^|\n)#|```", r["source_text"] + r["summary"]):
            rejected["markdown"] += 1
        elif r["book"] not in DEPARTMENT:
            rejected["book without department mapping"] += 1
        else:
            cand.append({**r, "category": cat})
    print("candidates", len(cand_all), "eligible", len(cand), dict(rejected))

    # 2. similarity over v2 + candidates; units
    v2flat = [(s, r) for s in SPLITS for r in v2[s]]
    texts = [r["source_text"] for _, r in v2flat] + [c["source_text"] for c in cand]
    sim = similarity(texts)
    base = len(v2flat)
    csim = sim[base:, base:]
    near_v2 = (sim[base:, :base] >= UNIT_LINK).any(axis=1)
    rejected_v2 = int(near_v2.sum())
    dsu = DSU(len(cand))
    for a, b in zip(*np.where(np.triu(csim >= UNIT_LINK))):
        dsu.union(int(a), int(b))
    # near-duplicate clusters (for one-row-per-cluster)
    clus = DSU(len(cand))
    for a, b in zip(*np.where(np.triu(csim >= NEAR_DUP))):
        clus.union(int(a), int(b))
    units: dict[int, list[int]] = defaultdict(list)
    for i in range(len(cand)):
        if not near_v2[i]:
            units[dsu.find(i)].append(i)
    # one row per near-duplicate cluster inside a unit: keep the earliest in book order (stable)
    for u, members in units.items():
        seen, keep = set(), []
        for i in sorted(members, key=lambda i: (cand[i]["book"], cand[i]["order"])):
            if clus.find(i) not in seen:
                seen.add(clus.find(i))
                keep.append(i)
        units[u] = keep
    unit_book = {u: Counter(cand[i]["book"] for i in m).most_common(1)[0][0] for u, m in units.items()}
    print("units", len(units), "excluded as near-duplicates of v2 rows", rejected_v2,
          "rows after cluster dedupe", sum(len(m) for m in units.values()))

    # 3. quotas
    have = {s: Counter(category(words(r["source_text"])) for r in v2[s]) for s in SPLITS}
    need = {}
    for s in ("validation", "test"):
        need[s] = {c: TARGETS[s][c] - have[s][c] for c in RANGES}
        if need[s]["short"] != 0:
            print(f"warning: {s} short rows {have[s]['short']} != target {TARGETS[s]['short']}")
        need[s]["short"] = 0
    need["train"] = {"short": 0, "medium": TARGETS["train"]["medium"] - have["train"]["medium"],
                     "long": TARGETS["train"]["long"] - have["train"]["long"]}
    need["train"]["very_long"] = PLAN["train"] - len(v2["train"]) - need["train"]["medium"] - need["train"]["long"]
    print("need", need)

    # 4. held-out, chapter by chapter, round-robin over books. Picking a chapter claims every unit that touches it;
    #    rows are taken from those units only while a quota is open (at most PER_CHAPTER per chapter, PER_BOOK per
    #    book and split), and every chapter that has a row in a claimed unit is quarantined: none of its rows can
    #    enter another split. Unused rows of claimed units are discarded.
    rng = random.Random(SEED)
    unit_of = {i: u for u, m in units.items() for i in m}
    chap = lambda i: (cand[i]["book"], cand[i]["chapter"] or "")
    chapters: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in sorted(unit_of, key=lambda i: (cand[i]["book"], cand[i]["order"])):
        chapters[chap(i)].append(i)
    by_book: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for k in sorted(chapters):
        by_book[k[0]].append(k)
    for b in by_book:
        rng.shuffle(by_book[b])
    books = sorted(by_book)
    rng.shuffle(books)
    chosen = {s: [] for s in SPLITS}
    claimed: dict[int, str] = {}
    quarantine: dict[tuple[str, str], str] = {}
    for s in ("validation", "test"):
        rem = dict(need[s])
        per_book = Counter()
        progress = True
        while sum(rem.values()) > 0 and progress:
            progress = False
            for b in books:
                if sum(rem.values()) == 0:
                    break
                if per_book[b] >= PER_BOOK:
                    continue
                for k in by_book[b]:
                    if k in quarantine:
                        continue
                    us = {unit_of[i] for i in chapters[k]}
                    if any(u in claimed and claimed[u] != s for u in us):
                        continue
                    rows_k = [i for u in us for i in units[u]]
                    touched = {chap(i) for i in rows_k}
                    if any(quarantine.get(t, s) != s for t in touched):
                        continue
                    if not any(rem[cand[i]["category"]] > 0 for i in rows_k):
                        continue
                    for u in us:
                        claimed[u] = s
                    for t in touched:
                        quarantine[t] = s
                    taken = 0
                    for i in sorted(rows_k, key=lambda i: (chap(i) != k, cand[i]["order"])):
                        c = cand[i]["category"]
                        if rem[c] > 0 and taken < PER_CHAPTER and per_book[b] < PER_BOOK:
                            rem[c] -= 1
                            taken += 1
                            per_book[b] += 1
                            chosen[s].append(i)
                    progress = True
                    break
        if sum(rem.values()) > 0:
            sys.exit(f"could not fill {s}: remaining {rem}")

    # 5. train: rows of unclaimed units in non-quarantined chapters, round-robin over books within each category
    rem = dict(need["train"])
    pool_rows = defaultdict(lambda: defaultdict(list))
    for i, u in unit_of.items():
        if u not in claimed and chap(i) not in quarantine:
            pool_rows[cand[i]["category"]][cand[i]["book"]].append(i)
    # the split totals are the plan; if a category runs out, its shortfall is filled from the next shorter one
    substituted = {}
    for c, nxt in (("very_long", "long"), ("long", "medium"), ("medium", None)):
        bk = sorted(pool_rows[c])
        for b in bk:
            pool_rows[c][b].sort()
            rng.shuffle(pool_rows[c][b])
        rng.shuffle(bk)
        while rem[c] > 0 and any(pool_rows[c][b] for b in bk):
            for b in bk:
                if rem[c] > 0 and pool_rows[c][b]:
                    chosen["train"].append(pool_rows[c][b].pop())
                    rem[c] -= 1
        if rem[c] > 0:
            if nxt is None:
                sys.exit(f"could not fill train/{c}: {rem[c]} missing")
            substituted[f"{c}->{nxt}"] = rem[c]
            rem[nxt] += rem[c]
            rem[c] = 0
    if substituted:
        print("train category substitutions (split total kept):", substituted)
    print("held-out chapters quarantined", len(quarantine), "claimed units", len(claimed))

    # 6. records
    final = {s: list(v2[s]) for s in SPLITS}
    counter = 0
    for s in SPLITS:
        for i in sorted(chosen[s], key=lambda i: (cand[i]["book"], cand[i]["order"])):
            c = cand[i]
            counter += 1
            final[s].append({
                "id": f"OSX-SUM-{counter:04d}", "department": DEPARTMENT[c["book"]], "course": c["book"],
                "topic": c["section"], "source_text": c["source_text"], "summary": c["summary"],
                "source_type": "openstax_textbook_section", "review_status": "public_author_written_summary",
                "chapter": c["chapter"], "license": c["license"],
                "source_ref": f"{c['repo']}@{c['commit'][:12]}:modules/{c['module']}",
            })

    # 7. verification
    flat = [(s, r) for s in SPLITS for r in final[s]]
    ids = [r["id"] for _, r in flat]
    assert len(ids) == len(set(ids))
    assert len({norm(r["source_text"]) for _, r in flat}) == len(flat), "exact duplicate source"
    assert len({norm(r["summary"]) for _, r in flat}) == len(flat), "exact duplicate summary"
    fsim = similarity([r["source_text"] for _, r in flat])
    arr = np.array([s for s, _ in flat])
    cross = {f"{a}/{b}": round(float(fsim[np.ix_(arr == a, arr == b)].max()), 4)
             for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    if max(cross.values()) >= NEAR_DUP:
        sys.exit(f"near duplicate crosses splits: {cross}")
    counts = {s: len(final[s]) for s in SPLITS}
    assert counts == PLAN, counts
    cats = {s: {c: sum(category(words(r["source_text"])) == c for r in final[s]) for c in RANGES} for s in SPLITS}

    out.mkdir(parents=True)
    hashes = {}
    for s in SPLITS:
        p = out / f"{s}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            for r in final[s]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    osx = [r for _, r in flat if r["source_type"] == "openstax_textbook_section"]
    repos = sorted({(r["source_ref"].split("@")[0], r["source_ref"].split("@")[1].split(":")[0]) for r in osx})
    manifest = {
        "dataset": "EduGuard academic summarization v3 (clean v2 + OpenStax textbook sections)",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tools": ["experiments/multitask_bloom_rewrite/scripts/clean_summarization_dataset.py",
                  "experiments/multitask_bloom_rewrite/scripts/import_openstax_summarization.py",
                  "experiments/multitask_bloom_rewrite/scripts/build_summarization_v3.py"],
        "seed": SEED, "plan": PLAN, "split_counts": counts, "length_category_by_split": cats,
        "train_category_substitutions": substituted,
        "source_type_by_split": {s: dict(Counter(r["source_type"] for r in final[s])) for s in SPLITS},
        "department_by_split": {s: dict(Counter(r["department"] for r in final[s]).most_common()) for s in SPLITS},
        "openstax": {
            "rows": len(osx), "books_by_split": {s: dict(Counter(r["course"] for r in final[s]
                                                                 if r["source_type"] == "openstax_textbook_section"))
                                                 for s in SPLITS},
            "licenses": dict(Counter(r["license"] for r in osx)),
            "repositories": [{"repo": a, "commit": b} for a, b in repos],
            "candidates": len(cand_all), "eligible": len(cand), "filter_rejections": dict(rejected),
            "excluded_near_duplicate_of_v2": rejected_v2,
            "attribution": "Content from OpenStax textbooks (Rice University), https://openstax.org, used under the "
                           "licence stated per row (CC BY-NC-SA 4.0 or CC BY 4.0). Text was converted from CNXML: "
                           "figures, tables, boxed features, exercises and citations were removed; the summary is "
                           "the author-written section summary.",
        },
        "checks": {"unique_ids": True, "exact_duplicate_source": 0, "exact_duplicate_summary": 0,
                   "max_cross_split_similarity": cross, "near_duplicate_threshold": NEAR_DUP,
                   "summary_ratio_max": round(max(words(r["summary"]) / words(r["source_text"]) for _, r in flat), 3),
                   "source_words_max": max(words(r["source_text"]) for _, r in flat),
                   "summary_words_median": {s: statistics.median(words(r["summary"]) for r in final[s]) for s in SPLITS}},
        "files_sha256": hashes,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("split_counts", "length_category_by_split", "checks")}, indent=1))
    print(json.dumps(manifest["department_by_split"], indent=0)[:1500])


if __name__ == "__main__":
    main()
