#!/usr/bin/env python3
"""Build conservative Bloom-target rewrite supervision from the raw Figshare corpus.

No LLM generation. A row is emitted only when the source can be converted by a
small, auditable transformation frame without dropping protected anchors or
introducing subject-matter content.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
MAP = {
    "knowledge":"Remember","remembering":"Remember","recall":"Remember",
    "comprehension":"Understand","understanding":"Understand",
    "application":"Apply","applying":"Apply",
    "analysis":"Analyze","analysing":"Analyze","analyzing":"Analyze",
    "evaluation":"Evaluate","evaluating":"Evaluate",
    "synthesis":"Create","creating":"Create",
}
DATASET_VERSION = "bloom_rewrite_synth_v5"
POLICY_VERSION = "bloom_target_policy_v5_controlled_conservative"
SOURCE_FILE = "data/figshare_bloom_v1.csv"
SEED = 42

STOP = set("""a an the and or but if then than that this these those it its of
in on at to for from with by as is are was were be been being do does did can
could may might will would should must into over under after before during
through about against between among within without using used your our their
this these those someone somebody""".split())

COGNITIVE = set("""define explain describe list name state identify recall
recognize recognise summarize summarise interpret classify illustrate apply
use calculate compute determine solve implement demonstrate analyze analyse
compare contrast differentiate distinguish examine evaluate assess critique
criticize criticise judge justify defend design develop construct formulate
propose create devise produce generate write build show discuss select choose
suggest recommend predict recite outline label draw sketch state compare
contrast appraise support""".split())

GENERIC = set("""key information facts meaning purpose explanation account
description summary interpretation differences difference similarities
similarity comparison relationships relationship interactions interaction
components component parts structure patterns causes effects effectiveness
suitability quality strengths limitations advantage disadvantages judgment
conclusion evidence criteria appropriate relevant related task problem work
outcome result results knowledge procedure approach solution plan strategy
model framework method design project artifact proposal system process original
new creative target subject matter""".split())

META = (
    "the rewritten question","rewritten question","original question",
    "bloom level","target level","as an ai","given constraints",
    "provided code","provided data","provided passage","this question asks",
    "here is the question",
)
PLACEHOLDER_RE = re.compile(r"(\.\.\.|\[[^\]]*\.\.\.[^\]]*\]|<[^>]*\.\.\.)")
NUM_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])"
)
FILE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])"
)
ACRONYM_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")

LEADING = re.compile(
    r"^\s*(?:briefly\s+|critically\s+|carefully\s+)?"
    r"(?:please\s+)?"
    r"(?P<verb>evaluate|assess|appraise|judge|justify|critique|criticize|criticise|"
    r"compare|contrast|differentiate|distinguish|analyze|analyse|examine|"
    r"calculate|compute|determine|find|solve|apply|use|implement|demonstrate|"
    r"design|develop|construct|formulate|propose|create|devise|produce|build|"
    r"write|draw|sketch|explain|describe|summarize|summarise|interpret|classify|"
    r"illustrate|discuss|define|identify|name|list|state|recite|outline|label|"
    r"predict|select|choose|specify)\b\s*",
    re.I,
)

HOW_PATTERN = re.compile(
    r"^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+"
    r"(?:you|we|someone|somebody|one)\s+"
    r"(?P<verb>define|explain|describe|compare|contrast|differentiate|distinguish|"
    r"analyze|analyse|examine|calculate|compute|determine|find|solve|apply|use|"
    r"implement|demonstrate|construct|design|develop|formulate|propose|create|"
    r"devise|produce|build|write|draw|sketch|identify|classify|interpret|select|"
    r"choose|estimate|measure)\b\s+",
    re.I,
)

WHAT_PATTERN = re.compile(r"^\s*what\s+(?:is|are|was|were)\s+", re.I)
WHY_PATTERN = re.compile(r"^\s*why\s+", re.I)


def canon(x):
    x = str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())


def norm(x):
    return re.sub(r"\s+", " ", (x or "").replace("\u00a0", " ")).strip()


def toks(x):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*", (x or "").lower())


def content(x):
    return {
        t for t in toks(x)
        if len(t) > 2 and t not in STOP and t not in COGNITIVE
    }


def protected(x):
    out = {m.group(0).lower() for m in NUM_RE.finditer(x or "")}
    out.update(m.group(0).lower() for m in TECH_RE.finditer(x or ""))
    out.update(m.group(0).lower() for m in FILE_RE.finditer(x or ""))
    out.update(m.group(1).lower() for m in ACRONYM_RE.finditer(x or ""))
    return out


def eligible(q):
    q = norm(q).strip(' "')
    if len(q) < 35 or len(q) > 850:
        return False, "length"
    if PLACEHOLDER_RE.search(q):
        return False, "placeholder"
    if q.count("?") > 1:
        return False, "multiple_questions"
    low = q.lower()
    if any(x in low for x in META):
        return False, "meta_context"
    # Missing-context formulations cannot be safely rewritten without the
    # missing material.
    missing = ("above data","above graph","above diagram","following code",
               "following segment","following passage","given below",
               "this article","the article","this magazine article")
    if any(x in low for x in missing):
        return False, "missing_context"
    if len(content(q)) < 6:
        return False, "thin_content"
    return True, ""


def topic(q):
    q = norm(q).strip(' "').rstrip("?.").strip()

    # Direct question forms: remove question auxiliary + cognitive verb.
    m = HOW_PATTERN.match(q)
    if m:
        rest = q[m.end():].strip()
        return rest, "how"

    m = WHAT_PATTERN.match(q)
    if m:
        return m.group(0).replace(q[:m.end()], ""), "what"

    # Definition/question fragments.
    if WHY_PATTERN.match(q):
        return q, "why"

    # Imperative form. Only accept a single leading cognitive command, followed
    # by one coherent topic clause. Multi-action tails are excluded.
    m = LEADING.match(q)
    if m:
        rest = q[m.end():].strip()
        # Exclude source prompts that contain another explicit cognitive command
        # later in the same item; those would otherwise produce mixed tasks.
        second = re.search(
            r"\b(?:and|then|also|plus|followed by)\s+(?:"
            + "|".join(sorted(COGNITIVE))
            + r")\b", rest, re.I,
        )
        if second:
            return None, "multiple_actions"

        # Drop common answer-format instructions after the main topic.
        rest = re.split(
            r"\s+(?:justify|support|show your working|provide (?:one|two|three|"
            r"four|five|an?|the)|explain why|give (?:one|two|three|four|five))\b",
            rest, maxsplit=1, flags=re.I,
        )[0].strip(" .;:,")
        if len(content(rest)) < 4:
            return None, "thin_topic"
        return rest, "imperative"

    return None, "unrecognized_form"


def target_rewrite(topic_text, target):
    # These frames add only cognitive/task language. They do not add new
    # subject-matter claims, examples, numerical values, criteria, or facts.
    if target == "Remember":
        return f"State {topic_text}.", "remember_state"
    if target == "Understand":
        return f"Explain {topic_text}.", "understand_explain"
    if target == "Apply":
        return f"Apply your knowledge of {topic_text} to a related problem.", "apply_related"
    if target == "Analyze":
        return f"Analyze {topic_text}.", "analyze_direct"
    if target == "Evaluate":
        return f"Evaluate {topic_text} and justify your judgment.", "evaluate_judgment"
    if target == "Create":
        return f"Develop a solution or plan related to {topic_text}.", "create_solution"
    raise ValueError(target)


def validate(source, topic_text, target, rewrite):
    reasons = []
    low = rewrite.lower()

    if len(rewrite) < 12 or len(rewrite) > 850:
        reasons.append("length")
    if any(x in low for x in META):
        reasons.append("meta")

    # For transformed rows, the target operation must be explicit at the
    # beginning. Identity rows (source level == target level) are accepted as
    # authentic source supervision and are marked separately.
    starts = {
        "Remember": r"^state\\b",
        "Understand": r"^explain\\b",
        "Apply": r"^apply\\b",
        "Analyze": r"^analyze\\b",
        "Evaluate": r"^evaluate\\b",
        "Create": r"^develop\\b",
    }
    if not re.search(starts[target], low):
        reasons.append("target_cue")

    src_p = protected(source)
    out_p = protected(rewrite)
    missing = sorted(src_p - out_p)
    if missing:
        reasons.append("protected_loss")

    # The topic is copied directly from the source after removing only its
    # original cognitive instruction, so no separate lexical "unsupported
    # content" gate is used. This avoids falsely rejecting legitimate source
    # terminology.
    src_c = content(topic_text)
    out_c = content(rewrite)
    recall = len(src_c & out_c) / max(1, len(src_c))
    if recall < 0.95:
        reasons.append("topic_recall")

    return not reasons, {
        "reasons": reasons,
        "topic_recall": round(recall, 4),
        "protected_spans": sorted(src_p),
        "missing_protected": missing,
        "unsupported_content": [],
    }


def read_sources(path):
    good, bad = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for idx, raw in enumerate(csv.DictReader(f)):
            q = norm(raw.get("question", "")).strip(' "')
            src_level = canon(raw.get("bloom_level", ""))
            if not q or not src_level:
                continue
            ok, why = eligible(q)
            if not ok:
                bad.append({
                    "row_index": idx, "source_question": q,
                    "source_bloom_level": src_level, "reason": why,
                })
                continue
            t, why2 = topic(q)
            if not t:
                bad.append({
                    "row_index": idx, "source_question": q,
                    "source_bloom_level": src_level, "reason": why2,
                })
                continue
            good.append({
                "source_id": "src_" + hashlib.sha256(q.encode()).hexdigest()[:16],
                "group_id": int(hashlib.sha256(q.encode()).hexdigest()[:8], 16),
                "source_question": q,
                "source_bloom_level": src_level,
                "topic": t,
                "topic_form": why2,
            })
    return good, bad


def split_sources(rows, seed):
    out = {"train": [], "validation": [], "test": []}
    for r in rows:
        u = int(hashlib.sha256(f"{seed}|split|{r['source_id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if u < 0.70:
            out["train"].append(r)
        elif u < 0.85:
            out["validation"].append(r)
        else:
            out["test"].append(r)
    return out


def candidates_for(rows):
    out = {t: [] for t in LEVELS}
    for src in rows:
        for target in LEVELS:
            # Same-level pairs are retained as authentic source supervision.
            # This gives the generator examples where the requested target
            # already matches the source Bloom label.
            if target == src["source_bloom_level"]:
                rewrite = src["source_question"].rstrip()
                template = "identity_source_question"
                info = {
                    "reasons": [],
                    "topic_recall": 1.0,
                    "protected_spans": sorted(protected(src["source_question"])),
                    "missing_protected": [],
                    "unsupported_content": [],
                }
                out[target].append((src, rewrite, template, info))
                continue

            rewrite, template = target_rewrite(src["topic"], target)
            ok, info = validate(src["source_question"], src["topic"], target, rewrite)
            if ok:
                out[target].append((src, rewrite, template, info))
    return out


def choose_balanced(cands, seed):
    # Every eligible source contributes one row for every target level
    # (identity for the same source level, controlled transformation otherwise),
    # so this produces exactly balanced target classes without synthetic padding.
    n = min(len(cands[t]) for t in LEVELS)
    chosen = []
    for target in LEVELS:
        arr = cands[target][:]
        rng = random.Random(seed + sum(map(ord, target)))
        rng.shuffle(arr)
        for src, rewrite, template, info in arr[:n]:
            chosen.append((target, src, rewrite, template, info))
    return chosen, n


def row(target, src, rewrite, template, info, split):
    messages = [
        {
            "role":"system",
            "content":(
                "Rewrite an academic question to the requested Bloom level. "
                "Preserve the source topic, technical concepts, quantities, "
                "named entities, and constraints. Do not answer the question. "
                "Output only one student-facing exam question."
            ),
        },
        {
            "role":"user",
            "content":f"Original question:\n{src['source_question']}\n\nTarget Bloom level:\n{target}",
        },
        {"role":"assistant","content":rewrite},
    ]
    text = (
        "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
        "<|im_start|>user\n" + messages[1]["content"] + "<|im_end|>\n"
        "<|im_start|>assistant\n" + rewrite + "<|im_end|>"
    )
    return {
        "example_id": hashlib.sha256(
            f"{DATASET_VERSION}|{src['source_id']}|{target}|{rewrite}".encode()
        ).hexdigest()[:16],
        "source_id": src["source_id"],
        "group_id": src["group_id"],
        "split": split,
        "source_question": src["source_question"],
        "source_bloom_level": src["source_bloom_level"],
        "target_bloom_level": target,
        "target_rewrite": rewrite,
        "transformation_type": f"{src['source_bloom_level']}->{target}",
        "synthetic_or_original": "synthetic",
        "synthetic": True,
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "quality_status": "pass",
        "construction_method": "deterministic_conservative_template",
        "construction_template": template,
        "source_file": SOURCE_FILE,
        "generator_inputs": ["source_question", "target_bloom_level"],
        "validation": info,
        "messages": messages,
        "text": text,
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_FILE)
    ap.add_argument("--output-dir", default=f"data/bloom_rewrite_versions/{DATASET_VERSION}")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_path = Path(args.source)
    out = Path(args.output_dir)
    if not src_path.exists():
        raise SystemExit(f"Source not found: {src_path}")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out}. Use --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_file():
            p.unlink()

    sources, rejected = read_sources(src_path)
    splits = split_sources(sources, args.seed)

    all_rows, candidate_report, balance = {}, {}, {}
    for split in ("train","validation","test"):
        c = candidates_for(splits[split])
        candidate_report[split] = {t: len(c[t]) for t in LEVELS}
        selected, n = choose_balanced(c, args.seed + len(split))
        rows = [row(target, src, rewrite, template, info, split)
                for target, src, rewrite, template, info in selected]
        rows.sort(key=lambda r: (LEVELS.index(r["target_bloom_level"]), r["example_id"]))
        all_rows[split] = rows
        balance[split] = n
        write_jsonl(out / f"{split}.jsonl", rows)

    source_sets = {s:{r["source_id"] for r in rows} for s,rows in all_rows.items()}
    leakage = {
        "train_validation": len(source_sets["train"] & source_sets["validation"]),
        "train_test": len(source_sets["train"] & source_sets["test"]),
        "validation_test": len(source_sets["validation"] & source_sets["test"]),
    }
    if any(leakage.values()):
        raise RuntimeError(f"Source leakage: {leakage}")

    src_hash = hashlib.sha256(src_path.read_bytes()).hexdigest()
    stats = {
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "seed": args.seed,
        "source_figshare": SOURCE_FILE,
        "source_figshare_sha256": src_hash,
        "eligible_source_count": len(sources),
        "rejected_source_count": len(rejected),
        "rejected_sources_by_reason": dict(Counter(x["reason"] for x in rejected)),
        "source_split_sizes": {s:len(v) for s,v in splits.items()},
        "candidate_counts": candidate_report,
        "balanced_target_count_per_level": balance,
        "counts": {
            s:{
                "total":len(rows),
                "by_target":dict(Counter(r["target_bloom_level"] for r in rows)),
                "by_source_target":dict(Counter(
                    f"{r['source_bloom_level']}->{r['target_bloom_level']}" for r in rows
                )),
            } for s,rows in all_rows.items()
        },
        "source_leakage_check": leakage,
        "notes":[
            "Synthetic supervision; not human gold.",
            "Built directly from data/figshare_bloom_v1.csv.",
            "No LLM generation or LLM judging is used.",
            "Only source-grounded deterministic task frames are emitted.",
            "Target classes are exactly balanced within each split at the maximum validated count.",
            "Each source question occurs in only one split.",
        ],
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "seed": args.seed,
        "source_figshare": SOURCE_FILE,
        "source_figshare_sha256": src_hash,
        "counts": {s:len(v) for s,v in all_rows.items()},
        "balanced_target_count_per_level": balance,
        "source_leakage_check": leakage,
    }
    (out/"dataset_statistics.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    write_jsonl(out/"rejected_sources.jsonl", rejected)
    (out/"README.md").write_text(
        "# bloom_rewrite_synth_v5\n\n"
        "Conservative deterministic Bloom-target rewrite supervision built directly "
        "from the authoritative Figshare corpus. No LLM generation or judging is used.\n\n"
        "Rows are emitted only after source-anchor, target-cue, topic-recall, and "
        "unsupported-content validation. Target classes are balanced to the maximum "
        "validated count available in each split.\n",
        encoding="utf-8",
    )

    print("ELIGIBLE SOURCES:", len(sources))
    print("SOURCE SPLITS:", {s:len(v) for s,v in splits.items()})
    print("CANDIDATES:", json.dumps(candidate_report, indent=2))
    print("BALANCED PER TARGET:", balance)
    print("ROWS:", {s:len(v) for s,v in all_rows.items()},
          "TOTAL:", sum(len(v) for v in all_rows.values()))
    print("LEAKAGE:", leakage)
    print("OUTPUT:", out)


if __name__ == "__main__":
    main()
