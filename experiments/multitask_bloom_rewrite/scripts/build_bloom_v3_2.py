"""Build EduGuard Bloom rewrite v3.2 from v3.1: reviewed corrections, Analyze verb disambiguation, pivot pairs.

Usage (repository root, project venv; bge-m3 runs on CPU unless --gpu):
  python -I experiments/multitask_bloom_rewrite/scripts/build_bloom_v3_2.py \
      [--src data/bloom_rewrite_eduguard_v3_1] [--out data/bloom_rewrite_eduguard_v3_2]

1. Corrections (experiments/multitask_bloom_rewrite/data_corrections/bloom_v3_2_corrections.json), reviewed row by
   row: rewrites that referred to "supplied / given / stated / specified" material the question does not contain are
   made self-contained (prompt rule: never imply data or requirements were supplied when they were not); four
   Remember rewrites that asked for interpretation, two Create rewrites that were Apply tasks, and two duplicate
   rewrites are replaced. The v3.1 text is kept in each revised row (v3_1_target_rewrite) with the reason.
2. Analyze rewrites that open with "Determine why/how/which/whether ..." (an Apply verb in prompts.py) open with an
   analysis verb instead (Analyze / Examine / Investigate), chosen deterministically per row. Wording is otherwise
   unchanged.
3. Pivot pairs (pair_type = "pivot"). Each source question and its five rewrites are six questions on one topic with
   known levels, so a rewrite can serve as the source for another member. Pivots fill source levels that are rare
   in v3.1 (7 Remember-level source questions in total) up to FLOOR rows per source level in train (held-out:
   floor scaled by the number of source questions). A pivot is used only if its target
     * is not contested, not a deliberately false claim and not an authored scenario/evaluation object,
     * adds no number that is absent from the pivot source (no invented data),
     * does not refer to supplied / given / stated / listed material unless the pivot source does too, and
     * has bge-m3 similarity to the pivot source >= the 10th percentile of the direct pairs (topic kept).
   Pivots stay inside their group's split, so no text crosses splits. Direct rows keep their v3.1 ids.
Every row carries pair_type ("direct" | "pivot"); revised rows carry v3_2_revision.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "validation", "test")
LEVELS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")
CORRECTIONS = REPO_ROOT / "experiments/multitask_bloom_rewrite/data_corrections/bloom_v3_2_corrections.json"
FLOOR_TRAIN = 300
MAX_PIVOTS_PER_SOURCE = 2
SIM_PERCENTILE = 10
SEED = 20261126
PHANTOM = re.compile(r"\b(?:the|a|an)\s+(?:\w+\s+)?(?:supplied|given|stated|specified)\b|\b(?:supplied|given|stated|specified)\s+"
                     r"(?:data|values?|parameters|information|requirements|constraints|criteria|specifications|case|scenario)\b",
                     re.I)
REFERS = re.compile(r"(supplied|given|stated|specified|provided|listed|following|identified|described|reported|proposed)")
VERBS = {"why": ("Analyze why", "Examine why", "Investigate why"),
         "how": ("Examine how", "Analyze how", "Investigate how"),
         "which": ("Examine which", "Investigate which"),
         "whether": ("Investigate whether", "Examine whether"),
         "what": ("Examine what", "Analyze what"),
         "under": ("Examine under",), "where": ("Investigate where",), "the": ("Analyze the",)}


def stable(key: str, n: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n


def disambiguate_analyze(text: str, rid: str) -> str:
    m = re.match(r"Determine(,)?\s+(\w+)", text)
    if not m:
        return text
    word = m.group(2).lower()
    if m.group(1):  # "Determine, from ..."
        return "Examine" + text[len("Determine"):]
    if word not in VERBS:
        return text
    opts = VERBS[word]
    head = opts[stable(rid, len(opts))]
    return head + text[m.end():]


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/bloom_rewrite_eduguard_v3_1"))
    ap.add_argument("--out", type=Path, default=Path("data/bloom_rewrite_eduguard_v3_2"))
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    src, out = REPO_ROOT / args.src, REPO_ROOT / args.out
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rows = {s: [json.loads(l) for l in (src / f"{s}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            for s in SPLITS}
    corr = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    ids = {r["id"] for s in SPLITS for r in rows[s]}
    missing = sorted(set(corr) - ids)
    if missing:
        sys.exit(f"corrections for unknown ids: {missing[:5]}")

    # 1-2. corrections and Analyze verb disambiguation
    stats = Counter()
    for s in SPLITS:
        for r in rows[s]:
            r["pair_type"] = "direct"
            old = r["target_rewrite"]
            reasons = []
            if r["id"] in corr:
                r["target_rewrite"] = corr[r["id"]]["target_rewrite"]
                reasons.append(corr[r["id"]]["reason"])
                stats["corrected:" + corr[r["id"]]["reason"].split(":")[0]] += 1
            if r["target_bloom_level"] == "Analyze":
                new = disambiguate_analyze(r["target_rewrite"], r["id"])
                if new != r["target_rewrite"]:
                    r["target_rewrite"] = new
                    reasons.append("analyze_verb: 'Determine' (an Apply verb) replaced by an analysis verb")
                    stats["analyze_verb"] += 1
            if reasons:
                r["v3_2_revision"] = reasons
                r["v3_1_target_rewrite"] = old
                r["review_status"] = "model_assisted_not_human_verified"
    flat = [r for s in SPLITS for r in rows[s]]
    tgt_counts = Counter(r["target_rewrite"].strip().lower() for r in flat)
    dups = [t for t, c in tgt_counts.items() if c > 1]
    if dups:
        sys.exit(f"duplicate target rewrites remain: {dups[:3]}")
    phantom_left = [r["id"] for r in flat if PHANTOM.search(r["target_rewrite"]) and not PHANTOM.search(r["source_question"])
                    and not re.search(r"\b(supplied|given|stated|specified|provided)\b", r["source_question"], re.I)]
    print("corrections", dict(stats), "| phantom references left (strict pattern):", len(phantom_left), phantom_left[:8])

    # 3. pivots
    sys.path.insert(0, str(REPO_ROOT / "experiments/multitask_bloom_rewrite/scripts"))
    from augment_evaluation import Embedder
    emb = Embedder()
    direct_sims = emb.cos([r["source_question"] for r in flat], [r["target_rewrite"] for r in flat])
    for r, x in zip(flat, direct_sims):
        r["_sim"] = x
    srt = sorted(direct_sims)
    sim_floor = srt[int(len(srt) * SIM_PERCENTILE / 100)]
    print(f"direct-pair bge-m3 similarity p{SIM_PERCENTILE} = {sim_floor:.3f}, p1 = {srt[len(srt) // 100]:.3f}")

    rng = random.Random(SEED)
    pivots = {s: [] for s in SPLITS}
    pivot_log = {}
    n_src = {s: len({r["source_id"] for r in rows[s]}) for s in SPLITS}
    for s in SPLITS:
        groups = defaultdict(list)
        for r in rows[s]:
            groups[r["source_id"]].append(r)
        contested_groups = {g for g, m in groups.items() if any(x.get("gold_label_contested") is True for x in m)}
        floor = round(FLOOR_TRAIN * n_src[s] / n_src["train"])
        have = Counter(r["source_bloom_level"] for r in rows[s] if r.get("gold_label_contested") is not True)
        # members per group: (text, level, row-or-None, provenance ok?)
        members = {}
        for g, m in groups.items():
            if g in contested_groups:
                continue
            orig = m[0]
            mem = [{"text": orig["source_question"], "level": orig["source_bloom_level"], "row": None, "eligible_target": True}]
            for x in m:
                ok = (x.get("deliberately_false_claim") is not True and x.get("content_provenance") == "source_derived")
                mem.append({"text": x["target_rewrite"], "level": x["target_bloom_level"], "row": x, "eligible_target": ok})
            members[g] = mem
        cand_pairs = defaultdict(list)  # source level -> [(g, src member, tgt member)]
        for g, mem in members.items():
            for a in mem:
                if a["row"] is None:
                    continue  # the original question is already the source of the direct pairs
                for b in mem:
                    if b is a or b["level"] == a["level"] or not b["eligible_target"]:
                        continue
                    if numbers(b["text"]) - numbers(a["text"]):
                        continue
                    if REFERS.findall(b["text"].lower()) and not set(REFERS.findall(b["text"].lower())) <= set(REFERS.findall(a["text"].lower())):
                        continue  # target points at supplied/given material the pivot source does not mention
                    cand_pairs[a["level"]].append((g, a, b))
        texts_a = [a["text"] for lv in cand_pairs for _, a, _ in cand_pairs[lv]]
        texts_b = [b["text"] for lv in cand_pairs for _, _, b in cand_pairs[lv]]
        sims = iter(emb.cos(texts_a, texts_b))
        scored = {lv: [(g, a, b, next(sims)) for g, a, b in cand_pairs[lv]] for lv in cand_pairs}
        for lv in LEVELS:
            need = max(0, floor - have[lv])
            pool = [p for p in scored.get(lv, []) if p[3] >= sim_floor]
            rng.shuffle(pool)
            by_target = defaultdict(list)
            for p in pool:
                by_target[p[2]["level"]].append(p)
            used_src = Counter()
            used_pair = set()
            picked = []
            order = [t for t in LEVELS if t != lv]
            while len(picked) < need and any(by_target[t] for t in order):
                for t in order:
                    while by_target[t]:
                        g, a, b, x = by_target[t].pop()
                        key = (a["text"], b["text"])
                        if used_src[a["text"]] >= MAX_PIVOTS_PER_SOURCE or key in used_pair:
                            continue
                        used_src[a["text"]] += 1
                        used_pair.add(key)
                        picked.append((g, a, b, x))
                        break
                    if len(picked) >= need:
                        break
            for g, a, b, x in picked:
                base = a["row"]
                tgt_row = b["row"]
                num = base["source_id"]
                rid = f"eduguard_v3_2_{num}-{lv}-to-{b['level']}"
                k = 2
                while rid in pivot_log:
                    rid = f"eduguard_v3_2_{num}-{lv}-to-{b['level']}-{k}"
                    k += 1
                row = {
                    "task": "bloom_rewrite", "split": s, "id": rid, "source_id": base["source_id"],
                    "group_id": base["group_id"], "discipline": base.get("discipline"), "topic": base.get("topic"),
                    "source_question": a["text"], "source_bloom_level": lv, "target_bloom_level": b["level"],
                    "target_rewrite": b["text"], "transformation_type": f"{lv}->{b['level']}",
                    "eligibility": (tgt_row or base).get("eligibility"),
                    "answer_source_available": (tgt_row or base).get("answer_source_available"),
                    "content_provenance": "source_derived", "deliberately_false_claim": False,
                    "pass3_rewritten": False, "gold_label_contested": False,
                    "review_status": "model_assisted_not_human_verified", "independent_review_verified": False,
                    "pair_type": "pivot",
                    "pivot_of": {"source_row": base["id"], "target_row": tgt_row["id"] if tgt_row else f"{base['source_id']}:original"},
                    "pivot_similarity": round(x, 4),
                }
                pivots[s].append(row)
                pivot_log[rid] = row["pivot_of"]
        print(s, "floor", floor, "pivots", len(pivots[s]), dict(Counter(p["source_bloom_level"] for p in pivots[s])))

    # 4. write
    out.mkdir(parents=True)
    hashes, counts = {}, {}
    for s in SPLITS:
        allrows = rows[s] + pivots[s]
        for r in allrows:
            r.pop("_sim", None)
        with (out / f"{s}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in allrows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        hashes[f"{s}.jsonl"] = hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest()
        counts[s] = {
            "rows": len(allrows), "direct_rows": len(rows[s]), "pivot_rows": len(pivots[s]),
            "source_questions": n_src[s], "revised_direct_rows": sum("v3_2_revision" in r for r in rows[s]),
            "source_level_counts": dict(Counter(r["source_bloom_level"] for r in allrows)),
            "target_level_counts": dict(Counter(r["target_bloom_level"] for r in allrows)),
            "transformation_pairs": len({r["transformation_type"] for r in allrows}),
            "min_rows_per_transformation_pair": min(Counter(r["transformation_type"] for r in allrows).values()),
            "contested_rows": sum(r.get("gold_label_contested") is True for r in allrows),
        }
    shutil.copy2(CORRECTIONS, out / "corrections_v3_2.json")
    manifest = {
        "version": "bloom_rewrite_eduguard_v3_2", "task": "bloom_rewrite", "date": dt.date.today().isoformat(),
        "derived_from": {"path": args.src.as_posix(), "files_sha256": {f"{s}.jsonl": hashlib.sha256(
            (src / f"{s}.jsonl").read_bytes()).hexdigest() for s in SPLITS}},
        "tool": "experiments/multitask_bloom_rewrite/scripts/build_bloom_v3_2.py", "seed": SEED,
        "changes": {"corrections": dict(stats), "corrections_file": "corrections_v3_2.json",
                    "pivot_rule": {"floor_train_rows_per_source_level": FLOOR_TRAIN, "max_pivots_per_source": MAX_PIVOTS_PER_SOURCE,
                                   "similarity_model": "BAAI/bge-m3", "similarity_floor": round(sim_floor, 4),
                                   "similarity_floor_percentile_of_direct_pairs": SIM_PERCENTILE,
                                   "target_filters": ["not contested", "not deliberately false", "source_derived provenance",
                                                      "no number absent from the pivot source", "no supplied/given/stated reference absent from the pivot source"]}},
        "counts": counts, "files_sha256": hashes,
        "leakage": {"pivots_stay_in_group_split": True},
        "review": {"corrections": "model-assisted (Claude), reviewed row by row; not human-verified",
                   "contested_gold_label_rows": sum(c["contested_rows"] for c in counts.values())},
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
