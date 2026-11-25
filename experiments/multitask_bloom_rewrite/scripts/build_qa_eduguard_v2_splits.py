"""Build data/qa_eduguard_v2: leakage-safe, discipline-stratified 1,172 / 146 / 147 academic QA splits.

v1 (data/qa_eduguard_v1) is kept unchanged as the audit baseline. Its audit found (a) reworded duplicate questions
across train/validation/test, (b) one discipline family (computing) at ~45% under three aliases, and (c) answers that
are unverified. v2 addresses these with documented, reproducible rules:

  Inputs (read only): data/train.jsonl (1,600 rows)
                      experiments/multitask_bloom_rewrite/qa_review/answer_review_v2.json (model-assisted review)
  1. Labels: original fields are kept verbatim; added fields discipline_canonical (aliases merged, e.g. CSE ->
     Computer Science and Engineering), discipline_group (11 coarse groups), qa_type_canonical, difficulty_canonical.
  2. Similarity: question-text TF-IDF cosine (max of char 3-5-gram and word unigram) on CPU.
  3. Exclusion of exactly 135 rows (1,600 - 1,465), in priority order:
       a. answers flagged by the review as incorrect or unsound;
       b. exact duplicate questions (later ID);
       c. reworded near-twins (similarity >= 0.80 to a row still kept), most similar first;
       d. if still short, the most redundant remaining rows. Discipline mix stays as authored (CSE-leaning).
     Excluded rows go to reserve.jsonl with a reason; nothing is deleted.
  4. Leakage groups: rows sharing a normalised topic or question, or with question similarity >= 0.55, form one
     group (union-find). A group never spans two splits.
  5. Split: per discipline_group quotas for test (147) and validation (146) by largest remainder, filled with whole
     groups (singletons first, seeded order); everything else is train (1,172).
Run from the repository root with the project venv (needs scikit-learn):
  .venv/Scripts/python.exe -I experiments/multitask_bloom_rewrite/scripts/build_qa_eduguard_v2_splits.py
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "data" / "train.jsonl"
V1 = REPO_ROOT / "data" / "qa_eduguard_v1"
REVIEW = REPO_ROOT / "experiments" / "multitask_bloom_rewrite" / "qa_review" / "answer_review_v2.json"
DEFAULT_OUT = REPO_ROOT / "data" / "qa_eduguard_v2"
SPLITS = ("train", "validation", "test")
HELDOUT_TARGETS = {"test": 147, "validation": 146}
TRAIN_TARGET = 1172
EXPECTED_SOURCE_ROWS = 1600
SEED = 20261126
GROUP_SIM = 0.55   # leakage grouping threshold
TWIN_SIM = 0.80    # redundancy threshold for exclusion
EXCLUDE_VERDICTS = {"incorrect", "unsound"}

DISCIPLINE_CANONICAL = {
    "CSE": "Computer Science and Engineering",
    "Computer Science": "Computer Science and Engineering",
    "Electrical Engineering": "Electrical and Electronic Engineering",
    "Social Science": "Social Sciences",
    "Business": "Business Administration",
}
DISCIPLINE_GROUP = {
    "Computer Science and Engineering": "Computing", "Information Technology": "Computing",
    "Electrical and Electronic Engineering": "Electrical and Electronic Engineering",
    "Civil Engineering": "Other Engineering and Architecture", "Mechanical Engineering": "Other Engineering and Architecture",
    "Industrial and Production Engineering": "Other Engineering and Architecture",
    "Textile Engineering": "Other Engineering and Architecture", "Environmental Engineering": "Other Engineering and Architecture",
    "Engineering": "Other Engineering and Architecture", "Architecture": "Other Engineering and Architecture",
    "Biomedical Engineering": "Other Engineering and Architecture",
    "Mathematics": "Mathematics and Statistics", "Statistics": "Mathematics and Statistics",
    "Physics": "Physics", "Chemistry": "Chemistry",
    "Earth Science": "Earth and Environmental Sciences", "Environmental Science": "Earth and Environmental Sciences",
    "Biology": "Life and Agricultural Sciences", "Microbiology": "Life and Agricultural Sciences",
    "Agriculture": "Life and Agricultural Sciences", "Fisheries": "Life and Agricultural Sciences",
    "Veterinary Science": "Life and Agricultural Sciences", "Nutrition and Food Science": "Life and Agricultural Sciences",
    "Medicine": "Health and Medicine", "Dentistry": "Health and Medicine", "Nursing": "Health and Medicine",
    "Pharmacy": "Health and Medicine", "Public Health": "Health and Medicine", "Health Sciences": "Health and Medicine",
    "Business Administration": "Business, Accounting and Economics", "Accounting": "Business, Accounting and Economics",
    "Economics": "Business, Accounting and Economics",
    "Law": "Law", "Law and Technology": "Law",
    "Psychology": "Social Sciences, Humanities and Education", "Education": "Social Sciences, Humanities and Education",
    "Humanities": "Social Sciences, Humanities and Education", "Social Sciences": "Social Sciences, Humanities and Education",
    "Sociology": "Social Sciences, Humanities and Education", "Political Science": "Social Sciences, Humanities and Education",
    "International Relations": "Social Sciences, Humanities and Education",
    "Communication": "Social Sciences, Humanities and Education",
    "Journalism and Mass Communication": "Social Sciences, Humanities and Education",
    "English": "Social Sciences, Humanities and Education", "Bangladesh Studies": "Social Sciences, Humanities and Education",
}
QA_TYPE_CANONICAL = {
    "comparison": "comparative", "analysis": "analytical", "critical_analysis": "analytical",
    "case_analysis": "analytical", "diagnostic": "diagnosis", "process_explanation": "process",
    "numerical": "calculation", "problem_solving": "calculation",
}
DIFFICULTY_CANONICAL = {"easy": "easy", "beginner": "easy", "medium": "medium", "intermediate": "medium",
                        "hard": "hard", "advanced": "hard"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split())


def similarity(questions: list[str]) -> np.ndarray:
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(questions)
    word = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(questions)
    sim = np.maximum(cosine_similarity(char), cosine_similarity(word))
    np.fill_diagonal(sim, 0.0)
    return sim


def largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    whole = sum(weights.values())
    raw = {k: v * total / whole for k, v in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--print-heldout", action="store_true", help="print held-out rows lacking an answer review")
    args = ap.parse_args()
    out_dir = args.out.resolve()
    if out_dir.exists():
        sys.exit(f"{out_dir} already exists; refusing to overwrite.")

    src_sha = sha256(SRC)
    v1_hashes = {p.name: sha256(p) for p in sorted(V1.glob("*"))} if V1.is_dir() else {}
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_SOURCE_ROWS or len({r["id"] for r in rows}) != len(rows):
        sys.exit("unexpected source row count or non-unique IDs")
    rows.sort(key=lambda r: r["id"])
    review = json.loads(REVIEW.read_text(encoding="utf-8")) if REVIEW.is_file() else {"reviews": {}}
    verdicts = {rid: v for rid, v in review["reviews"].items()}
    unknown = set(verdicts) - {r["id"] for r in rows}
    if unknown:
        sys.exit(f"review references unknown IDs: {sorted(unknown)[:5]}")

    for r in rows:
        canon = DISCIPLINE_CANONICAL.get(r["discipline"], r["discipline"])
        if canon not in DISCIPLINE_GROUP:
            sys.exit(f"{r['id']}: no discipline group for {r['discipline']!r}")
        r["discipline_canonical"] = canon
        r["discipline_group"] = DISCIPLINE_GROUP[canon]
        r["qa_type_canonical"] = QA_TYPE_CANONICAL.get(r["qa_type"], r["qa_type"])
        r["difficulty_canonical"] = DIFFICULTY_CANONICAL[r["difficulty"]]
        v = verdicts.get(r["id"])
        r["answer_review"] = v["verdict"] if v else "not_reviewed"
        if v and v.get("note"):
            r["answer_review_note"] = v["note"]

    sim = similarity([r["question"] for r in rows])
    n = len(rows)

    # 3. exclusion
    target_kept = TRAIN_TARGET + sum(HELDOUT_TARGETS.values())
    n_exclude = n - target_kept
    kept = np.ones(n, dtype=bool)
    reasons: dict[int, str] = {}
    first_q: dict[str, int] = {}
    for i, r in enumerate(rows):
        if r["answer_review"] in EXCLUDE_VERDICTS:
            reasons[i] = f"answer_review_{r['answer_review']}"
    for i, r in enumerate(rows):
        q = norm(r["question"])
        if q in first_q and i not in reasons:
            reasons[i] = "duplicate_question"
        first_q.setdefault(q, i)
    for i in reasons:
        kept[i] = False
    if len(reasons) > n_exclude:
        sys.exit(f"{len(reasons)} mandatory exclusions exceed the {n_exclude} available")

    # Discipline mix is kept as authored (CSE-leaning by the owner's choice): exclusion is by redundancy only.
    while len(reasons) < n_exclude:
        maxsim = np.where(kept[None, :], sim, 0.0).max(axis=1)
        live = np.where(kept)[0]
        best = int(live[np.lexsort((live, -maxsim[live]))[0]])  # most redundant; ties -> lowest index
        reasons[best] = "redundant_near_twin" if maxsim[best] >= TWIN_SIM else "most_redundant_remaining"
        kept[best] = False

    # 4. leakage groups over kept rows
    idx = [i for i in range(n) if kept[i]]
    parent = {i: i for i in idx}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i in idx:
        for key in (("t", norm(rows[i]["topic"])), ("q", norm(rows[i]["question"]))):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    kept_arr = np.array(idx)
    sub = sim[np.ix_(kept_arr, kept_arr)]
    for a, b in zip(*np.where(np.triu(sub >= GROUP_SIM))):
        parent[find(kept_arr[a])] = find(kept_arr[b])
    groups: dict[int, list[int]] = defaultdict(list)
    for i in idx:
        groups[find(i)].append(i)

    def group_of(members: list[int]) -> str:
        return Counter(rows[i]["discipline_group"] for i in members).most_common(1)[0][0]

    by_dgroup: dict[str, list[list[int]]] = defaultdict(list)
    for members in sorted(groups.values(), key=lambda m: min(m)):
        by_dgroup[group_of(members)].append(members)
    rng = random.Random(SEED)
    for g in by_dgroup.values():
        rng.shuffle(g)
        g.sort(key=len)  # stable: singletons first, seeded order within size
    weights = {g: sum(len(m) for m in ms) for g, ms in by_dgroup.items()}

    # 5. per-group quotas filled with whole groups
    assignment: dict[int, str] = {}
    for split, total in HELDOUT_TARGETS.items():
        quotas = largest_remainder(weights, total)
        filled = 0
        for g, ms in sorted(by_dgroup.items()):
            got, rest = 0, []
            for m in ms:
                if any(i in assignment for i in m):
                    continue
                if got + len(m) <= quotas[g]:
                    for i in m:
                        assignment[i] = split
                    got += len(m)
                else:
                    rest.append(m)
            if got != quotas[g]:
                sys.exit(f"{split}: could not fill quota for {g} ({got}/{quotas[g]})")
            filled += got
        assert filled == total
    for i in idx:
        assignment.setdefault(i, "train")
    counts = Counter(assignment.values())
    if counts["train"] != TRAIN_TARGET:
        sys.exit(f"train has {counts['train']} rows, expected {TRAIN_TARGET}")

    # cross-split leakage re-check on the final assignment
    split_arr = np.array([assignment.get(i, "reserve") for i in range(n)])
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        ia, ib = np.where(split_arr == a)[0], np.where(split_arr == b)[0]
        worst = float(sim[np.ix_(ia, ib)].max())
        if worst >= GROUP_SIM:
            sys.exit(f"leakage: {a}/{b} max question similarity {worst:.3f}")

    if args.print_heldout:
        for i in range(n):
            if split_arr[i] in ("validation", "test") and rows[i]["answer_review"] == "not_reviewed":
                r = rows[i]
                print(json.dumps({"id": r["id"], "split": split_arr[i], "discipline": r["discipline"],
                                  "q": r["question"], "a": r["reference_answer"]}, ensure_ascii=False))
        return

    # write
    out_dir.mkdir(parents=True)
    files, out_counts, stats, max_cross = {}, {}, {}, {}
    for split in (*SPLITS, "reserve"):
        chosen = [i for i in range(n) if split_arr[i] == split]
        out_rows = []
        for i in chosen:
            r = {**rows[i], "split": split}
            if split == "reserve":
                r["reserve_reason"] = reasons[i]
            out_rows.append(r)
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for r in out_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[f"{split}.jsonl"] = sha256(path)
        out_counts[split] = len(out_rows)
        stats[split] = {
            "discipline_group": dict(Counter(r["discipline_group"] for r in out_rows).most_common()),
            "discipline_canonical": dict(Counter(r["discipline_canonical"] for r in out_rows).most_common()),
            "qa_type_canonical": dict(Counter(r["qa_type_canonical"] for r in out_rows).most_common()),
            "difficulty_canonical": dict(Counter(r["difficulty_canonical"] for r in out_rows).most_common()),
            "answer_review": dict(Counter(r["answer_review"] for r in out_rows).most_common()),
        }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        ia, ib = np.where(split_arr == a)[0], np.where(split_arr == b)[0]
        max_cross[f"{a}/{b}"] = round(float(sim[np.ix_(ia, ib)].max()), 4)
    if sha256(SRC) != src_sha or (V1.is_dir() and {p.name: sha256(p) for p in sorted(V1.glob("*"))} != v1_hashes):
        sys.exit("source or v1 files changed during the build")

    src_groups = Counter(r["discipline_group"] for r in rows)
    used_groups = Counter(rows[i]["discipline_group"] for i in idx)
    manifest = {
        "version": "qa_eduguard_v2",
        "task": "academic_qa (closed-book: question -> reference_answer, no context passage)",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"path": "data/train.jsonl", "sha256": src_sha, "rows": n, "modified": False},
        "audit_baseline": {"path": "data/qa_eduguard_v1", "files_sha256": v1_hashes, "modified": False},
        "answer_review": {"path": REVIEW.relative_to(REPO_ROOT).as_posix(),
                          "sha256": sha256(REVIEW) if REVIEW.is_file() else None,
                          "method": review.get("method"),
                          "scope": review.get("scope")},
        "build_script": "experiments/multitask_bloom_rewrite/scripts/build_qa_eduguard_v2_splits.py",
        "seed": SEED,
        "thresholds": {"leakage_group_question_similarity": GROUP_SIM, "redundant_twin_similarity": TWIN_SIM,
                       "similarity": "max(cosine of char_wb 3-5-gram TF-IDF, cosine of word TF-IDF) on question text"},
        "counts": out_counts,
        "ratio": {s: round(out_counts[s] / target_kept, 4) for s in SPLITS},
        "reserve_reasons": dict(Counter(reasons.values()).most_common()),
        "discipline_group_share": {
            "source_1600": {g: round(c / n, 4) for g, c in src_groups.most_common()},
            "used_1465": {g: round(c / target_kept, 4) for g, c in used_groups.most_common()},
        },
        "leakage_groups": len(groups),
        "max_cross_split_question_similarity": max_cross,
        "files_sha256": files,
        "split_stats": stats,
        "limitations": [
            "Answers are not human-verified. answer_review is a model-assisted (Claude) review, not an expert check.",
            "Discipline mix is kept as authored (computing-leaning, by the dataset owner's choice); splits are stratified by discipline_group.",
            "Many disciplines have fewer than 10 rows; their held-out quota can be 0 or 1.",
            "Near-duplicate detection is lexical (TF-IDF); paraphrases with little word overlap can still be missed.",
            "Not wired into dataset_io.py or any training config; the existing qa task expects context/answer fields.",
        ],
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "counts": out_counts, "reserve_reasons": manifest["reserve_reasons"],
                      "share": manifest["discipline_group_share"], "max_cross": max_cross,
                      "heldout_review": {s: stats[s]["answer_review"] for s in ("validation", "test")}}, indent=1))


if __name__ == "__main__":
    main()
