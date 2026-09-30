#!/usr/bin/env python3
"""Build a conservative, source-preserving Bloom rewrite dataset.

Design principle:
  raw source question -> minimal target-task rewrite
No invented facts, contexts, criteria, examples, artifacts, or numbers.
Unsupported source/target transformations are omitted.

This builder is intentionally smaller than v3/v4-style synthetic expansion.
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

LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
MAP = {
    "knowledge": "Remember", "remembering": "Remember", "recall": "Remember",
    "comprehension": "Understand", "understanding": "Understand",
    "application": "Apply", "applying": "Apply",
    "analysis": "Analyze", "analysing": "Analyze", "analyzing": "Analyze",
    "evaluation": "Evaluate", "evaluating": "Evaluate",
    "synthesis": "Create", "creating": "Create",
}
DATASET_VERSION = "bloom_rewrite_synth_v12"
POLICY_VERSION = "bloom_target_policy_v12_minimal_task_transform"
SOURCE_FILE = "data/figshare_bloom_v1.csv"
SEED = 42

COGNITIVE = set("""
define explain describe list name state identify recall recognize recognise
summarize summarise interpret classify illustrate apply use calculate compute
determine find solve implement demonstrate analyze analyse compare contrast
differentiate distinguish examine evaluate assess critique criticize criticise judge
justify defend design develop construct formulate propose create devise produce
generate write build show discuss select choose suggest recommend predict recite
outline label draw sketch appraise specify modify estimate measure relate rank
""".split())

STOP = set("""
a an the and or but if then than that this these those it its of in on at to
for from with by as is are was were be been being do does did can could may
might will would should must into over under after before during through about
against between among within without using used your our their you we someone
somebody one
""".split())

NUM_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])"
)
FILE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])"
)
ACRONYM_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")
PLACEHOLDER_RE = re.compile(r"(?:\.{2,}|\[[^\]]*\.{2,}[^\]]*\]|<[^>]*\.{2,}[^>]*>)")

MISSING_CONTEXT = (
    "above data", "above graph", "above diagram", "following code",
    "following segment", "following passage", "given below", "this article",
    "the article", "this magazine article", "the figure above", "the table above",
)

# Source forms. The matched verb is removed only when it is the primary task
# operator. The remainder of the question is preserved.
LEADING = re.compile(
    r"""^\s*(?:briefly\s+|critically\s+|carefully\s+|concisely\s+)*
    (?:please\s+)?
    (?P<verb>
      evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|
      compare|contrast|differentiate|distinguish|analyze|analyse|examine|
      calculate|compute|determine|find|solve|apply|use|implement|demonstrate|
      design|develop|construct|formulate|propose|create|devise|produce|build|
      write|draw|sketch|explain|describe|summarize|summarise|interpret|classify|
      illustrate|discuss|define|identify|name|list|state|recite|outline|label|
      predict|select|choose|specify|modify|estimate|measure|relate|rank
    )\b\s*""",
    re.I | re.X,
)

HOW = re.compile(
    r"""^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+
    (?:you|we|someone|somebody|one)\s+
    (?P<verb>
      evaluate|assess|appraise|judge|justify|critique|defend|
      compare|contrast|differentiate|distinguish|analyze|analyse|examine|
      calculate|compute|determine|find|solve|apply|use|implement|demonstrate|
      design|develop|construct|formulate|propose|create|devise|produce|build|
      write|draw|sketch|explain|describe|summarize|summarise|interpret|classify|
      illustrate|discuss|define|identify|name|list|state|modify|estimate|measure|
      relate|rank
    )\b\s+""",
    re.I | re.X,
)

WHAT = re.compile(r"^\s*what\s+(?:is|are|was|were)\s+", re.I)
HOW_DIFFER = re.compile(
    r"^\s*how\s+(?:does|do|did)\s+(.+?)\s+differ\s+from\s+(.+?)\s*$", re.I
)

ANSWER_FORMAT_TAIL = re.compile(
    r"""(?:\s+show\s+your\s+working(?:\s+and\s+calculation)?|
    \s+justify\s+your\s+answer|
    \s+support\s+(?:your\s+answer|your\s+views?)|
    \s+elaborate(?:\s+your\s+answer)?|
    \s+provide\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|a|an)
      (?:\s+\w+){0,4}\s+examples?|
    \s+give\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|a|an)
      (?:\s+\w+){0,4}\s+examples?)\s*\.?$""",
    re.I | re.X,
)

RELATIONAL = (
    "between", "among", "relationship", "relationships", "interaction",
    "interactions", "compare", "contrast", "differentiate", "distinguish",
    "components", "component", "parts", "structure", "pattern", "patterns",
    "causes", "cause", "effects", "effect", "differences", "difference",
    "similarities", "similarity", "relationship between",
)

EVALUATIVE = (
    "advantage", "advantages", "disadvantage", "disadvantages", "effectiveness",
    "appropriate", "suitable", "suitability", "best", "better", "worse", "risk",
    "risks", "alternative", "alternatives", "recommend", "recommendation",
    "opinion", "agree", "justify", "strength", "strengths", "limitation",
    "limitations", "quality", "validity", "prefer", "preference", "judge",
)

APPLY_CONTEXT = (
    "calculate", "compute", "determine", "find", "solve", "apply", "use",
    "implement", "demonstrate", "algorithm", "procedure", "procedures",
    "method", "methods", "formula", "equation", "modify", "given", "data",
    "scenario", "case", "problem", "inputs", "output",
)

CREATE_ARTIFACT = (
    "diagram", "drawing", "sketch", "flow chart", "flowchart", "program",
    "algorithm", "presentation", "storyboard", "strategy", "plan", "proposal",
    "model", "framework", "system", "hypothesis", "project", "campaign",
    "poster", "prototype", "story", "play", "letter",
)


def norm(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").replace("\u00a0", " ")).strip()


def canon(x: str) -> str | None:
    x = str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())


def content(x: str) -> set[str]:
    return {
        t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*", (x or "").lower())
        if len(t) > 2 and t not in STOP and t not in COGNITIVE
    }


def protected(x: str) -> set[str]:
    out = {m.group(0).lower() for m in NUM_RE.finditer(x or "")}
    out.update(m.group(0).lower() for m in TECH_RE.finditer(x or ""))
    out.update(m.group(0).lower() for m in FILE_RE.finditer(x or ""))
    out.update(m.group(1).lower() for m in ACRONYM_RE.finditer(x or ""))
    return out


def parse_source(q: str) -> tuple[str | None, str, str]:
    q = norm(q).strip(' "').rstrip("?.").strip()

    m = HOW_DIFFER.match(q)
    if m:
        return f"how {m.group(1)} differs from {m.group(2)}", "how_differ", "question"

    m = HOW.match(q)
    if m:
        return q[m.end():].strip(" .;:"), m.group("verb").lower(), "question"

    m = WHAT.match(q)
    if m:
        return q[m.end():].strip(" .;:"), "what", "question"

    m = LEADING.match(q)
    if m:
        rest = q[m.end():].strip(" .;:,")
        rest = ANSWER_FORMAT_TAIL.sub("", rest).strip(" .;:,")
        if not rest:
            return None, "thin_topic", "imperative"
        # Mixed task instructions are not transformed.
        if re.search(
            r"\b(?:and|then|also|followed by)\s+(?:" +
            "|".join(sorted(COGNITIVE)) + r")\b", rest, re.I
        ):
            return None, "multiple_actions", "imperative"
        if re.search(
            r";\s*(?:" + "|".join(sorted(COGNITIVE)) + r")\b", rest, re.I
        ):
            return None, "multiple_actions", "imperative"
        return rest, m.group("verb").lower(), "imperative"

    return None, "unrecognized_form", ""


def source_eligible(q: str) -> tuple[bool, str]:
    q = norm(q).strip(' "')
    if not (35 <= len(q) <= 900):
        return False, "length"
    if PLACEHOLDER_RE.search(q):
        return False, "placeholder"
    low = q.lower()
    if any(x in low for x in MISSING_CONTEXT):
        return False, "missing_context"
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", q)
    if len(parts) > 1:
        return False, "multiple_sentences"
    if q.count("?") > 1:
        return False, "multiple_questions"
    if len(content(q)) < 6:
        return False, "thin_content"
    return True, ""


def capabilities(source: str, topic: str, form: str) -> dict[str, bool]:
    low = (source + " " + topic).lower()
    return {
        "remember": form not in {
            "calculate", "compute", "determine", "find", "solve", "apply",
            "use", "implement", "demonstrate", "design", "develop", "construct",
            "formulate", "propose", "create", "devise", "produce", "build",
            "write", "draw", "sketch", "modify", "how_calculate", "how_compute",
            "how_find", "how_solve",
        },
        "understand": True,
        "apply": (
            form in {
                "calculate", "compute", "determine", "find", "solve", "apply",
                "use", "implement", "demonstrate", "modify",
            }
            or any(re.search(r"\b" + re.escape(x) + r"\b", low) for x in APPLY_CONTEXT)
        ),
        "analyze": (
            form in {"compare", "contrast", "differentiate", "distinguish",
                     "analyze", "analyse", "examine", "how_differ"}
            or any(x in low for x in RELATIONAL)
        ),
        "evaluate": (
            form in {"evaluate", "assess", "appraise", "judge", "justify",
                     "critique", "defend", "recommend"}
            or any(x in low for x in EVALUATIVE)
        ),
        "create": (
            form in {
                "design", "develop", "construct", "formulate", "propose",
                "create", "devise", "produce", "build", "write", "draw",
                "sketch",
            }
            or any(x in low for x in CREATE_ARTIFACT)
        ),
    }


def supported(source: str, topic: str, form: str, target: str) -> bool:
    cap = capabilities(source, topic, form)
    low = topic.lower()

    if target == "Remember":
        # Do not turn inherently procedural/generative tasks into fact recall.
        if form in {
            "calculate", "compute", "determine", "find", "solve", "apply",
            "use", "implement", "demonstrate", "design", "develop", "construct",
            "formulate", "propose", "create", "devise", "produce", "build",
            "write", "draw", "sketch", "modify",
        }:
            # Relational factual prompts can still be recalled as differences.
            return cap["analyze"] and any(
                x in low for x in ("between", "difference", "differences", "similar")
            )
        return True

    if target == "Understand":
        return True

    if target == "Apply":
        return cap["apply"]

    if target == "Analyze":
        return cap["analyze"]

    if target == "Evaluate":
        return cap["evaluate"]

    if target == "Create":
        # Require an explicit artifact/creation concept; generic "system/model"
        # words alone are not enough unless they appear as an actual artifact.
        return cap["create"] and any(
            re.search(r"\b" + re.escape(x) + r"\b", low)
            for x in CREATE_ARTIFACT
        )

    return False


def make_rewrite(source: str, topic: str, form: str, target: str) -> tuple[str, str]:
    low = (source + " " + topic).lower()

    if target == "Remember":
        m = re.match(r"(?i)^the differences between\s+(.+)$", topic)
        if m:
            return f"State the differences between {m.group(1)}.", "remember_differences"
        m = re.match(r"(?i)^how\s+(.+?)\s+differ\s+from\s+(.+)$", topic)
        if m:
            return f"State how {m.group(1)} differs from {m.group(2)}.", "remember_difference"
        if form == "list" or re.search(r"\b\d+\s+(?:types?|advantages?|differences?|factors?|steps?|phases?|stages?|methods?)\b", topic, re.I):
            return f"List {topic}.", "remember_list"
        if form in {"define", "what"}:
            return f"State {topic}.", "remember_state"
        return f"Identify {topic}.", "remember_identify"

    if target == "Understand":
        return f"Explain {topic}.", "understand_explain"

    if target == "Apply":
        if re.search(r"\b(?:calculate|compute|determine|find|solve)\b", low):
            return f"Apply the relevant procedure to {topic}.", "apply_procedure"
        if re.search(r"\b(?:method|procedure|algorithm|formula|equation)\b", low):
            return f"Apply {topic} to a practical problem.", "apply_method"
        return f"Apply {topic} in a practical situation.", "apply_context"

    if target == "Analyze":
        if re.search(r"(?i)\bthe differences between\b", topic):
            return f"Analyze {topic}.", "analyze_differences"
        if re.search(r"(?i)\b(?:relationship|relationships|interactions?|between)\b", topic):
            return f"Analyze {topic}.", "analyze_relationship"
        if re.search(r"(?i)\b(?:components?|parts?|structure|patterns?|causes?|effects?)\b", topic):
            return f"Analyze {topic}.", "analyze_structure"
        return f"Analyze {topic}.", "analyze_direct"

    if target == "Evaluate":
        if any(x in low for x in ("advantages and disadvantages", "strengths and limitations")):
            return f"Evaluate {topic} and justify your judgment.", "evaluate_tradeoffs"
        if any(x in low for x in ("effectiveness", "suitability", "appropriate", "best", "better", "worse")):
            return f"Evaluate {topic} and justify your judgment.", "evaluate_effectiveness"
        if any(x in low for x in ("recommend", "recommendation", "should", "agree", "opinion")):
            return f"Evaluate {topic} and justify your judgment.", "evaluate_recommendation"
        return f"Evaluate {topic} and justify your judgment.", "evaluate_judgment"

    if target == "Create":
        if re.search(r"\b(?:diagram|drawing|sketch|flow chart|flowchart)\b", low):
            # Avoid "diagram representing a diagram"; preserve the original object.
            return f"Design {topic}.", "create_diagram"
        if re.search(r"\b(?:program|algorithm)\b", low):
            return f"Develop {topic}.", "create_computational"
        if re.search(r"\b(?:presentation|storyboard)\b", low):
            return f"Develop {topic}.", "create_presentation"
        if re.search(r"\b(?:strategy|plan|proposal)\b", low):
            return f"Develop {topic}.", "create_strategy"
        if re.search(r"\b(?:model|framework|system)\b", low):
            return f"Construct {topic}.", "create_model"
        if re.search(r"\bhypothesis\b", low):
            return f"Formulate {topic}.", "create_hypothesis"
        if re.search(r"\b(?:story|poem|play|letter)\b", low):
            return f"Create {topic}.", "create_creative"
        return f"Develop {topic}.", "create_solution"

    raise ValueError(target)


def validate(source: str, target: str, rewrite: str, topic: str) -> tuple[bool, dict]:
    reasons = []
    low = rewrite.lower()

    if any(x in low for x in (
        "the rewritten question", "original question", "bloom level",
        "as an ai", "given constraints", "provided code", "provided data",
        "academic artifact",
    )):
        reasons.append("META_LANGUAGE")

    if target != "Evaluate" and re.search(
        r"\b(?:and|then|also|followed by)\s+(?:define|explain|describe|list|name|state|identify|"
        r"calculate|compute|determine|find|solve|apply|use|implement|demonstrate|analyze|"
        r"analyse|compare|contrast|differentiate|examine|evaluate|assess|judge|justify|"
        r"design|develop|construct|formulate|propose|create|devise|produce|build|write|draw|"
        r"sketch|discuss)\b", low, re.I
    ):
        reasons.append("MULTIPLE_ACTIONS")

    if protected(source) - protected(rewrite):
        reasons.append("PROTECTED_SPAN_LOSS")

    # For cross-level pairs, the lexical content from the source topic must
    # remain intact. Generic cognitive words are excluded from content().
    src_c = content(topic)
    out_c = content(rewrite)
    recall = len(src_c & out_c) / max(1, len(src_c))
    if recall < 0.95:
        reasons.append("TOPIC_CONTENT_LOSS")

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


def row(split, target, s, rewrite, template, info):
    system = (
        "Rewrite an academic question to the requested Bloom level. Preserve "
        "the source content, technical concepts, quantities, named entities, "
        "and constraints. Do not invent subject matter. Output only one "
        "student-facing exam question."
    )
    user = f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": rewrite},
    ]
    packed = (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + user + "<|im_end|>\n"
        "<|im_start|>assistant\n" + rewrite + "<|im_end|>"
    )
    synthetic = rewrite != s["source_question"]
    return {
        "example_id": hashlib.sha256(
            f"{DATASET_VERSION}|{s['source_id']}|{target}|{rewrite}".encode()
        ).hexdigest()[:16],
        "source_id": s["source_id"], "group_id": s["group_id"], "split": split,
        "source_question": s["source_question"],
        "source_bloom_level": s["source_bloom_level"],
        "target_bloom_level": target,
        "target_rewrite": rewrite,
        "transformation_type": f"{s['source_bloom_level']}->{target}",
        "synthetic_or_original": "synthetic" if synthetic else "original_identity",
        "synthetic": synthetic,
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "quality_status": "pass",
        "construction_method": "deterministic_minimal_task_transform",
        "construction_template": template,
        "source_file": SOURCE_FILE,
        "generator_inputs": ["source_question", "target_bloom_level"],
        "validation": info,
        "messages": messages,
        "text": packed,
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

    sources, rejected = [], []
    seen = set()

    with src.open("r", encoding="utf-8-sig", newline="") as f:
        for idx, raw in enumerate(csv.DictReader(f)):
            q = norm(raw.get("question", "")).strip(' "')
            level = canon(raw.get("bloom_level", ""))
            if not q or not level:
                continue
            ok, why = source_eligible(q)
            if not ok:
                rejected.append({
                    "row_index": idx, "source_question": q,
                    "source_bloom_level": level, "reason": why
                })
                continue
            topic, form, style = parse_source(q)
            if not topic:
                rejected.append({
                    "row_index": idx, "source_question": q,
                    "source_bloom_level": level, "reason": form
                })
                continue
            sid = "src_" + hashlib.sha256(q.encode()).hexdigest()[:16]
            if sid in seen:
                continue
            seen.add(sid)
            sources.append({
                "source_id": sid,
                "group_id": int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16),
                "source_question": q,
                "source_bloom_level": level,
                "topic": topic,
                "form": form,
                "style": style,
            })

    splits = {"train": [], "validation": [], "test": []}
    for s in sources:
        u = int(
            hashlib.sha256(f"{args.seed}|split|{s['source_id']}".encode()).hexdigest()[:8], 16
        ) / 0xFFFFFFFF
        splits["train" if u < 0.70 else "validation" if u < 0.85 else "test"].append(s)

    all_rows = {}
    stats = {}

    for split in ("train", "validation", "test"):
        rows = []
        per_target = {}
        for target in LEVELS:
            target_rows = []
            for s in splits[split]:
                if s["source_bloom_level"] == target:
                    rewrite = s["source_question"]
                    info = {
                        "reasons": [],
                        "topic_content_recall": 1.0,
                        "protected_spans": sorted(protected(rewrite)),
                        "missing_protected": [],
                        "identity": True,
                    }
                    target_rows.append(
                        row(split, target, s, rewrite, "identity_source_question", info)
                    )
                    continue

                if not supported(s["source_question"], s["topic"], s["form"], target):
                    continue

                rewrite, template = make_rewrite(
                    s["source_question"], s["topic"], s["form"], target
                )
                ok, info = validate(
                    s["source_question"], target, rewrite, s["topic"]
                )
                if ok:
                    info["identity"] = False
                    target_rows.append(row(split, target, s, rewrite, template, info))

            per_target[target] = {
                "total": len(target_rows),
                "cross_level": sum(
                    r["source_bloom_level"] != r["target_bloom_level"]
                    for r in target_rows
                ),
                "identity": sum(
                    r["source_bloom_level"] == r["target_bloom_level"]
                    for r in target_rows
                ),
            }
            rows.extend(target_rows)

        rows.sort(key=lambda r: (LEVELS.index(r["target_bloom_level"]), r["example_id"]))
        all_rows[split] = rows
        stats[split] = per_target
        write_jsonl(out / f"{split}.jsonl", rows)

    sets = {s: {r["source_id"] for r in rows} for s, rows in all_rows.items()}
    leakage = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }
    if any(leakage.values()):
        raise RuntimeError(f"Source leakage detected: {leakage}")

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
            "Cross-level rows preserve the source remainder and alter only the task frame.",
            "Unsupported transformations are excluded rather than invented.",
            "Same-level source questions are retained as identity supervision.",
        ],
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "seed": args.seed,
        "source_figshare": SOURCE_FILE,
        "source_figshare_sha256": source_hash,
        "counts": {s: len(v) for s, v in all_rows.items()},
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
        "# bloom_rewrite_synth_v12\\n\\n"
        "Conservative deterministic Bloom-target rewrite supervision built "
        "directly from data/figshare_bloom_v1.csv. The source remainder is "
        "preserved and only the cognitive task frame is transformed. "
        "Unsupported transformations are omitted. No LLM is used.\\n",
        encoding="utf-8",
    )

    print("ELIGIBLE SOURCES:", len(sources))
    print("SOURCE SPLITS:", {s: len(v) for s, v in splits.items()})
    print("COUNTS:", json.dumps(summary["counts"], indent=2))
    print("REJECTED:", summary["rejected_sources_by_reason"])
    print("LEAKAGE:", leakage)
    print("OUTPUT:", out)


if __name__ == "__main__":
    main()
