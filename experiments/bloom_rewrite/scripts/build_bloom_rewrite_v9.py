#!/usr/bin/env python3
"""Build conservative Bloom-target rewrite supervision from the raw Figshare corpus.

No LLM is used. v9 is designed for training quality rather than maximum row
count: ambiguous, multi-action, context-dependent, or semantically unsuitable
transformations are excluded instead of being padded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
MAP = {
    "knowledge": "Remember", "remembering": "Remember", "recall": "Remember",
    "comprehension": "Understand", "understanding": "Understand",
    "application": "Apply", "applying": "Apply",
    "analysis": "Analyze", "analysing": "Analyze", "analyzing": "Analyze",
    "evaluation": "Evaluate", "evaluating": "Evaluate",
    "synthesis": "Create", "creating": "Create",
}
DATASET_VERSION = "bloom_rewrite_synth_v9"
POLICY_VERSION = "bloom_target_policy_v9_conservative"
SOURCE_FILE = "data/figshare_bloom_v1.csv"
SEED = 42

STOP = set("""a an the and or but if then than that this these those it its of in
on at to for from with by as is are was were be been being do does did can could
may might will would should must into over under after before during through
about against between among within without using used your our their you we
someone somebody one""".split())

COGNITIVE = set("""define explain describe list name state identify recall recognize
recognise summarize summarise interpret classify illustrate apply use calculate
compute determine solve implement demonstrate analyze analyse compare contrast
differentiate distinguish examine evaluate assess critique criticize criticise
judge justify defend design develop construct formulate propose create devise
produce generate write build show discuss select choose suggest recommend predict
recite outline label draw sketch appraise specify""".split())

FORBIDDEN_CONTEXT = (
    "above data", "above graph", "above diagram", "following code",
    "following segment", "following passage", "given below", "this article",
    "the article", "this magazine article", "the figure above", "the table above",
)

PLACEHOLDER_RE = re.compile(r"(?:\.\.\.|\\[.*?\\]|<.*?>)")
NUM_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])"
)
FILE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])"
)
ACRONYM_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")

LEADING = re.compile(
    r"""^\s*(?:briefly\s+|critically\s+|carefully\s+|concisely\s+)*
    (?:please\s+)?
    (?:
        evaluate|assess|appraise|judge|justify|critique|criticize|criticise|
        compare|contrast|differentiate|distinguish|analyze|analyse|examine|
        calculate|compute|determine|find|solve|apply|use|implement|demonstrate|
        design|develop|construct|formulate|propose|create|devise|produce|build|
        write|draw|sketch|explain|describe|summarize|summarise|interpret|classify|
        illustrate|discuss|define|identify|name|list|state|recite|outline|label|
        predict|select|choose|specify|relate|modify|estimate|measure
    )\b\s*""",
    re.I | re.X,
)

HOW = re.compile(
    r"""^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+
    (?:you|we|someone|somebody|one)\s+
    (?:
        define|explain|describe|compare|contrast|differentiate|distinguish|
        analyze|analyse|examine|calculate|compute|determine|find|solve|apply|
        use|implement|demonstrate|construct|design|develop|formulate|propose|
        create|devise|produce|build|write|draw|sketch|identify|classify|
        interpret|select|choose|estimate|measure|modify|relate|rank
    )\b\s*""",
    re.I | re.X,
)

DIRECT_HOW = re.compile(
    r"^\s*how\s+(?:does|do|did|can|could|would|should|will)\s+",
    re.I,
)

WHAT = re.compile(r"^\s*what\s+(?:is|are|was|were)\s+", re.I)

PREFIX_NOISE = re.compile(
    r"""^(?:
        with\s+(?:an?\s+)?appropriate\s+(?:example|examples|diagram|diagrams|drawing|drawings)\s*[:,]?\s*|
        by\s+using\s+(?:an?\s+)?appropriate\s+(?:diagram|diagrams|example|examples)\s*[:,]?\s*|
        in\s+(?:about\s+\d+\s+words|a\s+table(?:\s+form)?|table\s+form)\s*[:,]?\s*|
        through\s+(?:role[- ]play|a\s+role[- ]play)\s*[:,]?\s*|
        using\s+(?:an?\s+)?appropriate\s+(?:diagram|diagrams|example|examples)\s*[:,]?\s*
    )""",
    re.I | re.X,
)

ANSWER_TAIL = re.compile(
    r"""(?:
        \s+show\s+your\s+working(?:\s+and\s+calculation)?|
        \s+justify\s+your\s+answer|
        \s+support\s+(?:your\s+answer|your\s+views?)|
        \s+elaborate(?:\s+your\s+answer)?|
        \s+provide\s+(?:an?|one|two|three|four|five|six|seven|eight|nine|ten)
        (?:\s+\w+){0,4}\s+examples?|
        \s+give\s+(?:an?|one|two|three|four|five|six|seven|eight|nine|ten)
        (?:\s+\w+){0,4}\s+examples?
    )\s*\.?$""",
    re.I | re.X,
)

SECOND_ACTION = re.compile(
    r"\b(?:and|then|also|followed\s+by)\s+(?:" +
    "|".join(sorted(COGNITIVE)) +
    r")\b",
    re.I,
)

RELATION_PHRASE = re.compile(
    r"(?i)\b(?:compare|contrast|differentiate|distinguish)\b"
)
EVAL_CUES = (
    "advantage", "advantages", "disadvantage", "disadvantages", "effectiveness",
    "appropriate", "suitable", "suitability", "best", "better", "worse",
    "risk", "risks", "alternative", "alternatives", "choice", "choices",
    "recommend", "recommendation", "opinion", "agree", "justify", "strength",
    "strengths", "limitation", "limitations", "quality", "validity",
)
APPLY_CUES = (
    "calculate", "compute", "determine", "find", "solve", "apply", "use",
    "implement", "demonstrate", "algorithm", "procedure", "procedures", "method",
    "methods", "formula", "equation", "modify", "problem", "scenario", "case",
)
ANALYZE_CUES = (
    "between", "among", "relationship", "relationships", "interaction",
    "interactions", "compare", "contrast", "differentiate", "distinguish",
    "component", "components", "parts", "structure", "patterns", "causes",
    "effects", "difference", "differences", "similarity", "similarities",
)
CREATE_CUES = (
    "design", "develop", "construct", "formulate", "propose", "create", "devise",
    "produce", "build", "write", "draw", "sketch", "strategy", "plan", "solution",
    "system", "model", "framework", "program", "algorithm", "presentation",
    "diagram", "hypothesis", "project", "campaign", "poster", "prototype",
    "story", "storyboard", "play", "letter",
)
BAD_TARGET_TOPIC = (
    "new song", "new ending", "letter to", "role play", "pantomime",
)


def canon(x):
    x = str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())


def norm(x):
    return re.sub(r"\s+", " ", (x or "").replace("\u00a0", " ")).strip()


def toks(x):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*", (x or "").lower())


def content(x):
    return {t for t in toks(x) if len(t) > 2 and t not in STOP and t not in COGNITIVE}


def protected(x):
    out = {m.group(0).lower() for m in NUM_RE.finditer(x or "")}
    out.update(m.group(0).lower() for m in TECH_RE.finditer(x or ""))
    out.update(m.group(0).lower() for m in FILE_RE.finditer(x or ""))
    out.update(m.group(1).lower() for m in ACRONYM_RE.finditer(x or ""))
    return out


def has_cognitive_word(text):
    return bool(re.search(
        r"\b(?:" + "|".join(sorted(COGNITIVE)) + r")\b", text or "", re.I
    ))


def clean_topic(q):
    q = norm(q).strip(' "').rstrip("?.").strip()

    # Natural-question forms.
    m = HOW.match(q)
    if m:
        rest = q[m.end():].strip()
    else:
        m = DIRECT_HOW.match(q)
        if m:
            rest = q[m.end():].strip()
        else:
            m = WHAT.match(q)
            if m:
                rest = q[m.end():].strip()
            else:
                m = LEADING.match(q)
                if not m:
                    return None, "unrecognized_form"
                rest = q[m.end():].strip()

    # Remove answer-format/illustration framing at the beginning.
    prev = None
    while prev != rest:
        prev = rest
        rest = PREFIX_NOISE.sub("", rest).strip()

    # Remove trailing grading/instruction residue.
    rest = ANSWER_TAIL.sub("", rest).strip(" .;:,")

    # Reject mixed cognitive tasks rather than trying to splice them.
    if SECOND_ACTION.search(rest):
        return None, "multiple_actions"

    # If a semicolon starts a second cognitive command, reject it.
    if re.search(r";\s*(?:" + "|".join(sorted(COGNITIVE)) + r")\b", rest, re.I):
        return None, "multiple_actions"

    # Remove accidental "you should include" instruction tails only when the
    # source question itself contains that exact phrase.
    rest = re.sub(r"\s+you\s+should\s+include\b.*$", "", rest, flags=re.I)

    rest = norm(rest).strip(" .;:,")
    if not rest:
        return None, "thin_topic"

    # Topics starting with connective/prepositional residue are usually broken
    # extractions (e.g. "with appropriate examples how ...").
    if re.match(r"(?i)^(with|by|to|for|from|on|at)\b", rest):
        return None, "prefix_residue"

    # The topic itself must not contain another Bloom action.
    if has_cognitive_word(rest):
        return None, "cognitive_residue"

    if any(x in rest.lower() for x in FORBIDDEN_CONTEXT):
        return None, "missing_context"

    if len(content(rest)) < 4:
        return None, "thin_topic"

    if rest.lower() in BAD_TARGET_TOPIC:
        return None, "creative_fragment"

    return rest, "clean"


def eligible_source(q):
    q = norm(q).strip(' "')
    if not (35 <= len(q) <= 850):
        return False, "length"
    if PLACEHOLDER_RE.search(q):
        return False, "placeholder"
    if any(x in q.lower() for x in FORBIDDEN_CONTEXT):
        return False, "missing_context"
    # Most raw records should be one coherent exam item. Allow one terminal
    # punctuation mark, but reject obvious multi-sentence material.
    sentence_like = re.split(r"(?<=[.!?])\s+(?=[A-Z])", q)
    if len(sentence_like) > 1:
        return False, "multiple_sentences"
    if q.count("?") > 1:
        return False, "multiple_questions"
    if len(content(q)) < 6:
        return False, "thin_content"
    return True, ""


def source_capability(source, topic):
    low = (source + " " + topic).lower()
    return {
        "apply": any(x in low for x in APPLY_CUES),
        "analyze": any(x in low for x in ANALYZE_CUES),
        "evaluate": any(x in low for x in EVAL_CUES),
        "create": any(x in low for x in CREATE_CUES),
        "relation": bool(RELATION_PHRASE.search(low)) or any(
            x in low for x in ("between", "difference", "differences", "relationship")
        ),
        "artifact": any(x in low for x in (
            "program", "algorithm", "diagram", "drawing", "sketch",
            "presentation", "storyboard", "poster", "model", "framework",
            "strategy", "plan", "proposal", "hypothesis", "letter", "story",
            "play", "project", "prototype",
        )),
    }


def target_supported(source, source_level, topic, target):
    caps = source_capability(source, topic)

    if target == "Remember":
        if source_level in {"Apply", "Create"} and not caps["relation"]:
            return False
        if any(x in topic.lower() for x in BAD_TARGET_TOPIC):
            return False
        return True

    if target == "Understand":
        return True

    if target == "Apply":
        return caps["apply"]

    if target == "Analyze":
        return caps["analyze"] or (
            len(content(topic)) >= 7 and " and " in topic.lower()
        )

    if target == "Evaluate":
        return caps["evaluate"] or (
            caps["relation"] and any(x in topic.lower() for x in ("compare", "difference", "differences"))
        )

    if target == "Create":
        return caps["create"] and (
            caps["artifact"] or any(x in topic.lower() for x in (
                "problem", "solution", "strategy", "plan", "system", "model"
            ))
        )

    return False


def transform(source, topic, target):
    low = (source + " " + topic).lower()

    if target == "Remember":
        if source_capability(source, topic)["relation"]:
            return f"Identify the key differences or similarities in {topic}.", "remember_relation"
        if any(x in low for x in ("steps", "process", "procedure", "procedures", "stages")):
            return f"List the main {topic}.", "remember_list_process"
        if topic.lower().startswith("the meaning of "):
            return f"State {topic}.", "remember_definition"
        return f"Identify {topic}.", "remember_identify"

    if target == "Understand":
        return f"Explain {topic}.", "understand_explain"

    if target == "Apply":
        if any(x in low for x in ("calculate", "compute", "determine", "find", "solve")):
            return f"Apply the relevant procedure to {topic}.", "apply_procedure"
        return f"Apply your knowledge of {topic} in a practical situation.", "apply_context"

    if target == "Analyze":
        if source_capability(source, topic)["relation"]:
            return f"Analyze {topic}.", "analyze_relation"
        return f"Analyze {topic}, focusing on its key relationships and components.", "analyze_structure"

    if target == "Evaluate":
        return f"Evaluate {topic} and justify your judgment.", "evaluate_judgment"

    if target == "Create":
        if any(x in low for x in ("program", "algorithm", "code")):
            return f"Develop a program or algorithm related to {topic}.", "create_program"
        if any(x in low for x in ("diagram", "drawing", "sketch", "flow chart", "flowchart")):
            return f"Design a diagram representing {topic}.", "create_diagram"
        if any(x in low for x in ("presentation", "slide", "storyboard")):
            return f"Develop a presentation about {topic}.", "create_presentation"
        if any(x in low for x in ("strategy", "plan", "proposal")):
            return f"Develop a strategy or plan for {topic}.", "create_strategy"
        if any(x in low for x in ("model", "framework", "system")):
            return f"Construct a model or framework for {topic}.", "create_model"
        if "hypothesis" in low:
            return f"Formulate a hypothesis about {topic}.", "create_hypothesis"
        if any(x in low for x in ("story", "novel", "poem", "play", "letter")):
            return f"Create a response related to {topic}.", "create_creative"
        return f"Develop a solution for {topic}.", "create_solution"

    raise ValueError(target)


def validate(source, topic, target, rewrite):
    reasons = []
    low = rewrite.lower()

    if any(x in low for x in (
        "the rewritten question", "original question", "bloom level",
        "as an ai", "given constraints", "provided code", "provided data",
        "provided passage", "academic artifact"
    )):
        reasons.append("META_LANGUAGE")

    if protected(source) - protected(rewrite):
        reasons.append("PROTECTED_SPAN_LOSS")

    src = content(topic)
    out = content(rewrite)
    recall = len(src & out) / max(1, len(src))
    if recall < 0.95:
        reasons.append("TOPIC_CONTENT_LOSS")

    if target != "Evaluate" and SECOND_ACTION.search(rewrite):
        reasons.append("MULTIPLE_ACTIONS")

    starts = {
        "Remember": r"^(state|identify|list)\b",
        "Understand": r"^explain\b",
        "Apply": r"^apply\b",
        "Analyze": r"^analyze\b",
        "Evaluate": r"^evaluate\b",
        "Create": r"^(design|develop|construct|formulate|create)\b",
    }
    if not re.search(starts[target], low):
        reasons.append("TARGET_OPERATION_MISSING")

    return not reasons, {
        "reasons": reasons,
        "topic_content_recall": round(recall, 4),
        "protected_spans": sorted(protected(source)),
        "missing_protected": sorted(protected(source) - protected(rewrite)),
    }


def make_row(split, target, source_row, rewrite, template, info):
    source = source_row["source_question"]
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite an academic question to the requested Bloom level. "
                "Preserve the source topic, technical concepts, quantities, "
                "named entities, and constraints. Do not answer the question. "
                "Output only one student-facing exam question."
            ),
        },
        {
            "role": "user",
            "content": f"Original question:\n{source}\n\nTarget Bloom level:\n{target}",
        },
        {"role": "assistant", "content": rewrite},
    ]
    packed = (
        "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
        "<|im_start|>user\n" + messages[1]["content"] + "<|im_end|>\n"
        "<|im_start|>assistant\n" + rewrite + "<|im_end|>"
    )
    return {
        "example_id": hashlib.sha256(
            f"{DATASET_VERSION}|{source_row['source_id']}|{target}|{rewrite}".encode()
        ).hexdigest()[:16],
        "source_id": source_row["source_id"],
        "group_id": source_row["group_id"],
        "split": split,
        "source_question": source,
        "source_bloom_level": source_row["source_bloom_level"],
        "target_bloom_level": target,
        "target_rewrite": rewrite,
        "transformation_type": f"{source_row['source_bloom_level']}->{target}",
        "synthetic_or_original": "synthetic" if rewrite != source else "original_identity",
        "synthetic": rewrite != source,
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "quality_status": "pass",
        "construction_method": "deterministic_conservative",
        "construction_template": template,
        "source_file": SOURCE_FILE,
        "generator_inputs": ["source_question", "target_bloom_level"],
        "validation": info,
        "messages": messages,
        "text": packed,
    }


def read_sources(path):
    good = {}
    rejected = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for idx, raw in enumerate(csv.DictReader(f)):
            q = norm(raw.get("question", "")).strip(' "')
            level = canon(raw.get("bloom_level", ""))
            if not q or not level:
                continue

            ok, why = eligible_source(q)
            if not ok:
                rejected.append({
                    "row_index": idx, "source_question": q,
                    "source_bloom_level": level, "reason": why,
                })
                continue

            topic_text, why2 = clean_topic(q)
            if not topic_text:
                rejected.append({
                    "row_index": idx, "source_question": q,
                    "source_bloom_level": level, "reason": why2,
                })
                continue

            sid = "src_" + hashlib.sha256(q.encode()).hexdigest()[:16]
            if sid in good:
                continue
            good[sid] = {
                "source_id": sid,
                "group_id": int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16),
                "source_question": q,
                "source_bloom_level": level,
                "topic": topic_text,
            }
    return list(good.values()), rejected


def split_sources(rows, seed):
    out = {"train": [], "validation": [], "test": []}
    for row in rows:
        u = int(
            hashlib.sha256(f"{seed}|split|{row['source_id']}".encode()).hexdigest()[:8],
            16,
        ) / 0xFFFFFFFF
        out["train" if u < 0.70 else "validation" if u < 0.85 else "test"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_FILE)
    ap.add_argument("--output-dir", default=f"data/bloom_rewrite_versions/{DATASET_VERSION}")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.output_dir)

    if not src.exists():
        raise SystemExit(f"Source not found: {src}")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out}. Use --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_file():
            p.unlink()

    sources, rejected = read_sources(src)
    splits = split_sources(sources, args.seed)

    all_rows = {}
    stats = {}

    for split in ("train", "validation", "test"):
        rows = []
        per_target = {}
        for target in LEVELS:
            accepted = []
            for source_row in splits[split]:
                if source_row["source_bloom_level"] == target:
                    rewrite = source_row["source_question"]
                    info = {
                        "reasons": [],
                        "topic_content_recall": 1.0,
                        "protected_spans": sorted(protected(rewrite)),
                        "missing_protected": [],
                    }
                    accepted.append(make_row(
                        split, target, source_row, rewrite,
                        "identity_source_question", info
                    ))
                    continue

                if not target_supported(
                    source_row["source_question"],
                    source_row["source_bloom_level"],
                    source_row["topic"],
                    target,
                ):
                    continue

                rewrite, template = transform(
                    source_row["source_question"], source_row["topic"], target
                )
                ok, info = validate(
                    source_row["source_question"],
                    source_row["topic"],
                    target,
                    rewrite,
                )
                if ok:
                    info["cross_level"] = True
                    accepted.append(
                        make_row(split, target, source_row, rewrite, template, info)
                    )

            per_target[target] = {
                "total": len(accepted),
                "cross_level": sum(
                    r["source_bloom_level"] != r["target_bloom_level"]
                    for r in accepted
                ),
                "identity": sum(
                    r["source_bloom_level"] == r["target_bloom_level"]
                    for r in accepted
                ),
            }
            rows.extend(accepted)

        rows.sort(key=lambda r: (LEVELS.index(r["target_bloom_level"]), r["example_id"]))
        all_rows[split] = rows
        stats[split] = per_target
        write_jsonl(out / f"{split}.jsonl", rows)

    split_sets = {
        s: {r["source_id"] for r in rows} for s, rows in all_rows.items()
    }
    leakage = {
        "train_validation": len(split_sets["train"] & split_sets["validation"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["validation"] & split_sets["test"]),
    }
    if any(leakage.values()):
        raise RuntimeError(f"Source leakage: {leakage}")

    source_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    summary = {
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "seed": args.seed,
        "source_figshare": SOURCE_FILE,
        "source_figshare_sha256": source_hash,
        "eligible_source_count": len(sources),
        "rejected_source_count": len(rejected),
        "rejected_sources_by_reason": dict(Counter(r["reason"] for r in rejected)),
        "source_split_sizes": {s: len(v) for s, v in splits.items()},
        "counts": {
            s: {
                "total": len(all_rows[s]),
                "by_target": dict(Counter(r["target_bloom_level"] for r in all_rows[s])),
                "cross_level_rows": sum(
                    r["source_bloom_level"] != r["target_bloom_level"]
                    for r in all_rows[s]
                ),
                "identity_rows": sum(
                    r["source_bloom_level"] == r["target_bloom_level"]
                    for r in all_rows[s]
                ),
            }
            for s in all_rows
        },
        "candidate_statistics": stats,
        "source_leakage_check": leakage,
        "notes": [
            "Synthetic supervision; not human gold.",
            "Built directly from data/figshare_bloom_v1.csv.",
            "No LLM generation or LLM judging is used.",
            "Only coherent single-task source questions are admitted.",
            "Cross-level transformations require target-specific source capability.",
            "Same-level source questions are retained as identity supervision.",
            "No artificial target padding is used.",
        ],
    }

    manifest = {
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "seed": args.seed,
        "source_figshare": SOURCE_FILE,
        "source_figshare_sha256": source_hash,
        "counts": {s: len(rows) for s, rows in all_rows.items()},
        "source_leakage_check": leakage,
    }

    (out / "dataset_statistics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_jsonl(out / "rejected_sources.jsonl", rejected)
    (out / "README.md").write_text(
        "# bloom_rewrite_synth_v9\n\n"
        "Conservative deterministic Bloom-target rewrite supervision built "
        "directly from data/figshare_bloom_v1.csv. No LLM generation or judging "
        "is used. Ambiguous and unsupported transformations are excluded rather "
        "than padded. Same-level rows retain the original source question.\n",
        encoding="utf-8",
    )

    print("ELIGIBLE SOURCES:", len(sources))
    print("SOURCE SPLITS:", {s: len(v) for s, v in splits.items()})
    print("COUNTS:", json.dumps(summary["counts"], indent=2))
    print("REJECTED:", summary["rejected_sources_by_reason"])
    print("LEAKAGE:", leakage)
    print("OUTPUT:", out)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
