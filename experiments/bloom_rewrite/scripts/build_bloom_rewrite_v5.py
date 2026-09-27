#!/usr/bin/env python3
"""Deterministic, controlled Bloom-target rewrite dataset builder.

Builds synthetic rewrite supervision directly from data/figshare_bloom_v1.csv.
No LLM generation or judging is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
CANONICAL = {
    "knowledge": "Remember", "remembering": "Remember", "recall": "Remember",
    "comprehension": "Understand", "understanding": "Understand",
    "application": "Apply", "applying": "Apply",
    "analysis": "Analyze", "analyzing": "Analyze", "analysing": "Analyze",
    "evaluation": "Evaluate", "evaluating": "Evaluate",
    "synthesis": "Create", "creating": "Create",
}
DATASET_VERSION = "bloom_rewrite_synth_v5"
POLICY_VERSION = "bloom_target_policy_v5_controlled"
SOURCE_FILE = "data/figshare_bloom_v1.csv"
SEED = 42

# Same approximate training footprint as the old v3, but with exactly balanced
# target classes and source-group isolation.
TARGET_COUNTS = {"train": 430, "validation": 110, "test": 180}

STOP = set("""
a an the and or but if then than that this these those it its of in on at
to for from with by as is are was were be been being do does did can could
may might will would should must into over under after before during through
about against between among within without using used user question task answer
students student
""".split())

COGNITIVE = set("""
define explain describe list name state identify recall recognize recognise
summarize summarise interpret classify illustrate apply use calculate compute
determine solve implement demonstrate analyze analyse compare contrast
differentiate distinguish examine evaluate assess critique criticize criticise
judge justify defend design develop construct formulate propose create devise
produce generate write build show discuss select choose suggest recommend
predict recite outline label draw sketch
""".split())

GENERIC = set("""
key information facts meaning purpose explanation account description summary
interpretation differences difference similarities similarity comparison
relationships relationship interactions interaction components component parts
structure patterns causes effects effectiveness suitability quality strengths
limitations advantage disadvantages judgment conclusion evidence criteria
appropriate relevant related task problem work outcome result results
knowledge procedure approach solution plan strategy model framework method
design project artifact proposal system process original new creative
""".split())

META_FORBIDDEN = (
    "the rewritten question", "rewritten question:", "original question:",
    "bloom level", "target level", "the student should", "as an ai",
    "academic artifact", "given constraints", "provided code", "provided data",
    "provided passage", "this question asks", "here is the question",
)

PLACEHOLDER_RE = re.compile(r"(\.\.\.|\[\.\.\.\]|\{\.\.\.\}|<\.\.\.>|\b_+\b)")
PROTECTED_NUM = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
PROTECTED_TECH = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])"
)
PROTECTED_FILE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])"
)
ACRONYM_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")

LEADING_PATTERNS = [
    r"^can you state\s+", r"^can you explain\s+", r"^can you propose\s+",
    r"^can you defend\s+", r"^how would you\s+", r"^how could someone\s+",
    r"^how could we\s+", r"^how do you\s+", r"^how does\s+", r"^how do\s+",
    r"^how can we\s+", r"^how can you\s+", r"^why don't you\s+",
    r"^why do you\s+", r"^what is\s+", r"^what are\s+",
    r"^which of the following\s+", r"^what would you\s+", r"^what factors would you\s+",
    r"^what solutions would you suggest for\s+", r"^please\s+",
    r"^(?:briefly\s+)?(?:critically\s+)?(?:carefully\s+)?(?:evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|recommend|suggest|propose|develop|design|construct|formulate|create|compose|produce|devise|build|write|draw|sketch|illustrate|calculate|compute|determine|find|solve|apply|use|demonstrate|compare|contrast|differentiate|distinguish|analyze|analyse|examine|discuss|describe|explain|summarize|summarise|interpret|classify|define|identify|name|list|state|recite|outline|predict|select|choose|specify)\s+",
]

ARTIFACT_CUES = (
    "design", "develop", "construct", "formulate", "create", "compose", "produce",
    "devise", "build", "write", "draw", "sketch", "proposal", "strategy", "plan",
    "solution", "model", "framework", "program", "algorithm", "presentation",
    "storyboard", "story", "play", "hypothesis",
)
APPLY_CUES = (
    "calculate", "compute", "determine", "find", "solve", "apply", "use",
    "implement", "demonstrate", "given", "data", "formula", "equation",
    "algorithm", "procedure", "method", "scenario", "case", "problem",
)
RELATIONAL_CUES = (
    "between", "among", "relationship", "relationships", "interaction",
    "interactions", "compare", "contrast", "differentiate", "distinguish",
    "components", "parts", "structure", "patterns", "causes", "effects",
    "similarities", "differences",
)
EVALUATIVE_CUES = (
    "evaluate", "assess", "appraise", "judge", "justify", "critique",
    "criticize", "criticise", "defend", "effectiveness", "advantage",
    "disadvantage", "strength", "limitation", "appropriate", "best", "suitable",
    "suitability", "recommend", "should",
)


def canonical_level(value: str) -> str | None:
    value = str(value or "").strip()
    if value in LEVELS:
        return value
    return CANONICAL.get(value.lower())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()


def content_tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*", (text or "").lower())
        if len(t) > 2 and t not in STOP and t not in COGNITIVE
    }


def protected_spans(text: str) -> set[str]:
    out = {m.group(0).lower() for m in PROTECTED_NUM.finditer(text or "")}
    out.update(m.group(0).lower() for m in PROTECTED_TECH.finditer(text or ""))
    out.update(m.group(0).lower() for m in PROTECTED_FILE.finditer(text or ""))
    out.update(m.group(1).lower() for m in ACRONYM_RE.finditer(text or ""))
    return out


def eligible_source(q: str) -> tuple[bool, str]:
    q = normalize(q).strip(' "')
    if len(q) < 32:
        return False, "too_short"
    if len(q) > 900:
        return False, "too_long"
    if PLACEHOLDER_RE.search(q):
        return False, "template_or_placeholder"
    if q.count("?") > 1:
        return False, "multiple_questions"
    if len(content_tokens(q)) < 5:
        return False, "insufficient_content_tokens"
    low = q.lower()
    missing = (
        "above data", "above graph", "above diagram", "following segment",
        "following code", "given below", "following passage", "this article",
        "this magazine article",
    )
    if any(x in low for x in missing):
        return False, "missing_context"
    return True, ""


def strip_suffixes(text: str) -> str:
    out = text
    suffixes = (
        r"\s+show your working(?: and calculation)?\.?$",
        r"\s+justify your answer\.?$",
        r"\s+support your answer(?:s)?\.?$",
        r"\s+support your views(?: with [^.]*)?\.?$",
        r"\s+elaborate(?: your answer)?\.?$",
    )
    for p in suffixes:
        out = re.sub(p, "", out, flags=re.I)
    return normalize(out).strip(" .;:")


def topic_for(source: str) -> str:
    q = normalize(source).strip(' "').rstrip("?").strip()

    # Preserve the useful topic in common "X and justify why Y" questions.
    m = re.match(
        r"^(?:evaluate|assess|judge|justify|critique|criticize|criticise)\s+(.+?)\s+and\s+(?:justify|explain)\s+why\s+(.+)$",
        q, flags=re.I,
    )
    if m:
        return normalize(f"{m.group(1)} and why {m.group(2)}").strip(" .;:")

    m = re.match(r"^(?:differentiate|distinguish)\s+between\s+(.+)$", q, flags=re.I)
    if m:
        return normalize("the differences between " + m.group(1)).strip(" .;:")

    m = re.match(r"^(?:compare|contrast)\s+(.+)$", q, flags=re.I)
    if m:
        return normalize("the similarities and differences in " + m.group(1)).strip(" .;:")

    m = re.match(r"^(?:define|what is|what are)\s+(.+)$", q, flags=re.I)
    if m:
        return normalize("the meaning of " + m.group(1)).strip(" .;:")

    m = re.match(r"^(?:calculate|compute|determine|find|solve)\s+(?:the|a|an)?\s*(.+)$", q, flags=re.I)
    if m:
        return strip_suffixes(m.group(1))

    for p in LEADING_PATTERNS:
        new = re.sub(p, "", q, count=1, flags=re.I)
        if new != q:
            q = new.strip()
            break

    # Remove additional pure cognitive commands while retaining their topics.
    q = re.sub(
        r"\s*;\s*(?:briefly\s+)?(?:explain|describe|discuss|justify|evaluate|assess|support|show)\b",
        ";", q, flags=re.I,
    )
    q = strip_suffixes(q)
    return normalize(q).strip(" .;:")


def artifact_for(source: str) -> str:
    low = source.lower()
    if any(x in low for x in ("diagram", "drawing", "sketch", "flow chart", "flowchart")):
        return "a diagram"
    if any(x in low for x in ("presentation", "slide", "storyboard")):
        return "a presentation"
    if any(x in low for x in ("program", "code", "algorithm", "pseudocode", "pseudo code")):
        return "a program or algorithm"
    if any(x in low for x in ("strategy", "tactic", "plan")):
        return "a strategy or plan"
    if any(x in low for x in ("model", "framework", "system")):
        return "a model or framework"
    if "hypothesis" in low:
        return "a hypothesis"
    if any(x in low for x in ("story", "book", "novel", "poem", "play", "dialogue")):
        return "a creative response"
    return "an original solution or approach"


def transform(source: str, target: str) -> tuple[str, str]:
    topic = topic_for(source)
    low_source = source.lower()
    low_topic = topic.lower()

    if target == "Remember":
        if low_topic.startswith("the meaning of "):
            rewrite = f"State the key facts about {topic[16:]}."
            template = "remember_facts"
        elif low_topic.startswith(("how ", "why ", "whether ", "when ", "where ")):
            rewrite = f"State {topic}."
            template = "remember_state_clause"
        else:
            rewrite = f"Identify the key information about {topic}."
            template = "remember_identify"

    elif target == "Understand":
        if low_topic.startswith("the meaning of "):
            rewrite = f"Explain {topic}."
        elif low_topic.startswith(("how ", "why ", "whether ", "when ", "where ")):
            rewrite = f"Describe {topic}."
        else:
            rewrite = f"Explain {topic}."
        template = "understand_explain"

    elif target == "Apply":
        if any(c in low_source for c in APPLY_CUES):
            rewrite = f"Apply the relevant procedure to determine {topic}."
            template = "apply_procedure"
        elif any(c in low_source for c in ("use", "apply", "implement", "demonstrate", "formula", "equation")):
            rewrite = f"Apply your knowledge of {topic} to a related task."
            template = "apply_knowledge"
        else:
            rewrite = f"Use your knowledge of {topic} to address a related problem."
            template = "apply_related"

    elif target == "Analyze":
        if any(c in low_topic for c in RELATIONAL_CUES) or low_topic.startswith(("how ", "why ")):
            rewrite = f"Analyze {topic}."
            template = "analyze_relation"
        else:
            rewrite = f"Analyze {topic} in terms of its key relationships and structure."
            template = "analyze_structure"

    elif target == "Evaluate":
        if any(c in low_source for c in EVALUATIVE_CUES):
            rewrite = f"Evaluate {topic} and justify your judgment."
            template = "evaluate_judgment"
        else:
            rewrite = f"Assess {topic} and justify your conclusion."
            template = "evaluate_conclusion"

    elif target == "Create":
        artifact = artifact_for(source)
        if artifact == "a diagram":
            rewrite = f"Design {artifact} that represents {topic}."
        elif artifact == "a presentation":
            rewrite = f"Develop {artifact} about {topic}."
        elif artifact == "a program or algorithm":
            rewrite = f"Develop {artifact} for {topic}."
        elif artifact == "a strategy or plan":
            rewrite = f"Develop {artifact} for {topic}."
        elif artifact == "a model or framework":
            rewrite = f"Construct {artifact} for {topic}."
        elif artifact == "a hypothesis":
            rewrite = f"Formulate {artifact} related to {topic}."
        elif artifact == "a creative response":
            rewrite = f"Create {artifact} based on {topic}."
        else:
            rewrite = f"Develop {artifact} related to {topic}."
        template = "create_controlled"

    else:
        raise ValueError(target)

    return normalize(rewrite), template


def target_ok(rewrite: str, target: str) -> bool:
    low = rewrite.lower()
    cues = {
        "Remember": ("identify", "state", "list", "name", "recall", "recognize"),
        "Understand": ("explain", "describe", "summarize", "interpret", "classify", "illustrate"),
        "Apply": ("apply", "use", "demonstrate"),
        "Analyze": ("analyze", "analyse", "examine"),
        "Evaluate": ("evaluate", "assess", "judge", "critique"),
        "Create": ("design", "develop", "construct", "formulate", "create"),
    }
    return bool(re.match(r"^(" + "|".join(cues[target]) + r")\b", low))


def validate(source: str, target: str, rewrite: str) -> tuple[bool, dict]:
    reasons: list[str] = []
    if len(rewrite) < 25:
        reasons.append("TOO_SHORT")
    if len(rewrite) > 500:
        reasons.append("TOO_LONG")
    if "?" in rewrite:
        reasons.append("UNEXPECTED_QUESTION_MARK")
    if any(x in rewrite.lower() for x in META_FORBIDDEN):
        reasons.append("META_LANGUAGE")
    if not target_ok(rewrite, target):
        reasons.append("TARGET_CUE_MISSING")
    if len(re.findall(r"[.!?]", rewrite)) > 1:
        reasons.append("MULTIPLE_SENTENCE")

    src_protected = protected_spans(source)
    out_protected = protected_spans(rewrite)
    missing = sorted(src_protected - out_protected)
    if missing:
        reasons.append("PROTECTED_SPAN_LOSS")

    src_content = content_tokens(source)
    out_content = content_tokens(rewrite)
    allowed = GENERIC | {target.lower()}
    unsupported = sorted(out_content - src_content - allowed)
    if unsupported:
        reasons.append("UNSUPPORTED_CONTENT:" + " | ".join(unsupported[:12]))

    shared = src_content & out_content
    recall = len(shared) / max(1, len(src_content))
    if recall < 0.35 and len(src_content) >= 10:
        reasons.append("LOW_SOURCE_CONTENT_RECALL")
    if target in {"Apply", "Create"} and recall < 0.45 and len(src_content) >= 8:
        reasons.append("LOW_TRANSFORMED_TOPIC_RECALL")

    return (not reasons), {
        "reasons": reasons,
        "source_content_recall": round(recall, 4),
        "protected_spans": sorted(src_protected),
        "missing_protected": missing,
        "unsupported_content": unsupported,
    }


def read_sources(path: Path) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader):
            q = normalize(raw.get("question", "")).strip(' "')
            level = canonical_level(raw.get("bloom_level", ""))
            if not q or level is None:
                continue
            ok, reason = eligible_source(q)
            if not ok:
                rejected.append({
                    "row_index": idx, "source_question": q,
                    "source_bloom_level": level, "reason": reason,
                })
                continue
            sid = "src_" + sha(q)[:16]
            accepted.append({
                "source_id": sid,
                "group_id": int(sha(sid)[:8], 16),
                "source_question": q,
                "source_bloom_level": level,
            })
    return accepted, rejected


def split_sources(sources: list[dict], seed: int) -> dict[str, list[dict]]:
    out = {"train": [], "validation": [], "test": []}
    for row in sources:
        u = int(sha(f"{seed}|split|{row['source_id']}")[:8], 16) / 0xFFFFFFFF
        if u < 0.60:
            out["train"].append(row)
        elif u < 0.75:
            out["validation"].append(row)
        else:
            out["test"].append(row)
    return out


def candidates(split_rows: list[dict]) -> dict[str, list[tuple[dict, dict]]]:
    out = {t: [] for t in LEVELS}
    for src in split_rows:
        for target in LEVELS:
            if target == src["source_bloom_level"]:
                continue
            # Do not force extremely thin prompts into Apply/Create.
            if target == "Apply" and len(content_tokens(src["source_question"])) < 8 and not any(
                x in src["source_question"].lower() for x in APPLY_CUES
            ):
                continue
            if target == "Create" and len(content_tokens(src["source_question"])) < 8 and not any(
                x in src["source_question"].lower() for x in ARTIFACT_CUES
            ):
                continue
            rewrite, template = transform(src["source_question"], target)
            ok, info = validate(src["source_question"], target, rewrite)
            if ok:
                out[target].append((src, {
                    "rewrite": rewrite,
                    "template": template,
                    "validation": info,
                }))
    return out


def select_balanced(cands: dict[str, list[tuple[dict, dict]]], n: int, seed: int) -> list[tuple[str, dict, dict]]:
    selected = []
    for target in LEVELS:
        arr = cands[target][:]
        rng = random.Random(seed + sum(map(ord, target)))
        rng.shuffle(arr)
        if len(arr) < n:
            raise RuntimeError(
                f"Insufficient validated candidates for {target}: need {n}, have {len(arr)}"
            )
        for src, info in arr[:n]:
            selected.append((target, src, info))
    return selected


def row(src: dict, target: str, info: dict, split: str) -> dict:
    rewrite = info["rewrite"]
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert academic assessment editor. Rewrite an "
                "academic question so that the student's required cognitive "
                "task matches the requested Bloom level. Preserve the source "
                "topic, technical concepts, quantities, named entities, and "
                "constraints. Do not answer the question. Output only one "
                "student-facing exam question."
            ),
        },
        {
            "role": "user",
            "content": f"Original question:\n{src['source_question']}\n\nTarget Bloom level:\n{target}",
        },
        {"role": "assistant", "content": rewrite},
    ]
    text = (
        "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
        "<|im_start|>user\n" + messages[1]["content"] + "<|im_end|>\n"
        "<|im_start|>assistant\n" + rewrite + "<|im_end|>"
    )
    return {
        "example_id": sha(f"{DATASET_VERSION}|{src['source_id']}|{target}|{rewrite}")[:16],
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
        "construction_method": "deterministic_controlled_template",
        "construction_template": info["template"],
        "source_file": SOURCE_FILE,
        "generator_inputs": ["source_question", "target_bloom_level"],
        "validation": info["validation"],
        "messages": messages,
        "text": text,
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_FILE)
    ap.add_argument("--output-dir", default=f"data/bloom_rewrite_versions/{DATASET_VERSION}")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    source_path = Path(args.source)
    out = Path(args.output_dir)
    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out}. Use --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_file():
            p.unlink()

    sources, rejected = read_sources(source_path)
    print(f"RAW ELIGIBLE SOURCES: {len(sources)}")
    print("SOURCE LABELS:", dict(Counter(x["source_bloom_level"] for x in sources)))

    split_map = split_sources(sources, args.seed)
    print("SOURCE SPLITS:", {k: len(v) for k, v in split_map.items()})

    all_rows = {}
    template_usage = Counter()
    candidate_report = {}

    for split in ("train", "validation", "test"):
        cands = candidates(split_map[split])
        candidate_report[split] = {t: len(cands[t]) for t in LEVELS}
        selected = select_balanced(cands, TARGET_COUNTS[split], args.seed + len(split))
        rows = []
        for target, src, info in selected:
            rows.append(row(src, target, info, split))
            template_usage[info["template"]] += 1
        rows.sort(key=lambda r: (LEVELS.index(r["target_bloom_level"]), r["example_id"]))
        all_rows[split] = rows
        write_jsonl(out / f"{split}.jsonl", rows)

    # Exact target balance.
    for split, rows in all_rows.items():
        counts = Counter(r["target_bloom_level"] for r in rows)
        expected = TARGET_COUNTS[split]
        assert all(counts[t] == expected for t in LEVELS), (split, counts)

    # Source-group isolation across splits.
    source_sets = {
        s: {r["source_id"] for r in rows} for s, rows in all_rows.items()
    }
    leakage = {
        "train_validation": len(source_sets["train"] & source_sets["validation"]),
        "train_test": len(source_sets["train"] & source_sets["test"]),
        "validation_test": len(source_sets["validation"] & source_sets["test"]),
    }
    assert not any(leakage.values()), leakage

    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    stats = {
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "seed": args.seed,
        "source_figshare": SOURCE_FILE,
        "source_figshare_sha256": source_hash,
        "counts": {
            s: {
                "total": len(rows),
                "by_target": dict(Counter(r["target_bloom_level"] for r in rows)),
                "by_source_target": dict(Counter(
                    f"{r['source_bloom_level']}->{r['target_bloom_level']}" for r in rows
                )),
            } for s, rows in all_rows.items()
        },
        "eligible_sources": len(sources),
        "rejected_source_count": len(rejected),
        "rejected_sources_by_reason": dict(Counter(r["reason"] for r in rejected)),
        "candidate_counts_before_sampling": candidate_report,
        "template_usage": dict(sorted(template_usage.items())),
        "source_split_sizes": {s: len(v) for s, v in split_map.items()},
        "source_leakage_check": leakage,
        "notes": [
            "Synthetic supervision only; not human gold.",
            "Built deterministically from the authoritative Figshare CSV.",
            "No LLM generation or LLM judging is used.",
            "Source Bloom level is metadata only.",
            "Each source question is assigned to exactly one split.",
            "Target classes are exactly balanced within each split.",
        ],
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "seed": args.seed,
        "source_figshare": SOURCE_FILE,
        "source_figshare_sha256": source_hash,
        "counts": {s: len(v) for s, v in all_rows.items()},
        "target_counts_per_split": TARGET_COUNTS,
        "source_leakage_check": leakage,
    }
    (out / "dataset_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_jsonl(
        out / "rejected_sources.jsonl",
        rejected,
    )
    (out / "README.md").write_text(
        "# bloom_rewrite_synth_v5\n\n"
        "Controlled Bloom-target rewrite supervision built directly from "
        "data/figshare_bloom_v1.csv.\n\n"
        "The dataset is deterministic synthetic supervision, not human gold. "
        "No LLM generation or LLM judging is used. Numeric and technical "
        "anchors are preserved, unsupported subject-matter terms are rejected, "
        "target classes are balanced, and source questions are isolated by split.\n",
        encoding="utf-8",
    )

    print("\nBUILT DATASET")
    for split, rows in all_rows.items():
        print(f"  {split}: {len(rows)}")
    print(f"  total: {sum(len(v) for v in all_rows.values())}")
    print("  target balance:", {
        s: dict(Counter(r["target_bloom_level"] for r in rows))
        for s, rows in all_rows.items()
    })
    print("  source leakage:", leakage)
    print("  output:", out)


if __name__ == "__main__":
    main()
