#!/usr/bin/env python3
"""Build the final Figshare-backed Bloom target-level rewrite dataset.

Source of truth:
  data/figshare_bloom_v1_train.csv
  data/figshare_bloom_v1_val.csv
  data/figshare_bloom_v1_test.csv

For each usable source question, this produces one student-facing rewrite
for each of the six target Bloom levels.  Cross-level examples use
source-anchored deterministic templates rather than verb-only substitution.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "rewrite_dataset"
LEVELS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")
INPUTS = {
    "train": ROOT / "data" / "figshare_bloom_v1_train.csv",
    "validation": ROOT / "data" / "figshare_bloom_v1_val.csv",
    "test": ROOT / "data" / "figshare_bloom_v1_test.csv",
}

SYSTEM = (
    "Rewrite an academic question to the requested Bloom level. "
    "Preserve the original topic, technical concepts, quantities, named entities, "
    "and constraints. Do not invent subject matter. Do not merely replace the verb; "
    "change the student's cognitive operation. Output only one student-facing "
    "exam question."
)

TASK_VERBS = (
    "define|explain|describe|discuss|list|name|state|identify|recall|recognize|"
    "summarize|interpret|classify|apply|calculate|compute|solve|demonstrate|"
    "analyze|analyse|compare|contrast|differentiate|distinguish|examine|evaluate|"
    "assess|justify|critique|appraise|design|develop|create|propose|formulate|"
    "construct|suggest|recommend|highlight|outline|determine|give|show|comment|"
    "illustrate|mention|specify|write|draw|revise|participate|derive|predict|"
    "choose|select|use|implement|elaborate|retell|recite|compose|criticize|"
    "criticise|rank|place|judge|defend|correct|find|detect|fix|sketch|estimate|"
    "label|verify|paraphrase|perform|produce|devise|prepare|plan"
)

PREFIX_RE = re.compile(
    rf"^(?:(?:please|briefly|clearly|concisely|carefully|critically)\s+)+"
    rf"|^(?:based on (?:your|the) understanding|in your own words)"
    rf"[,:]?\s*|^(?:with|using|by using)\s+(?:the\s+|an?\s+)?"
    rf"(?:appropriate|relevant)\s+[^,;:]+[,:]\s*|"
    rf"^(?:with (?:the )?aid of)\s+[^,;:]+[,:]\s*",
    re.I,
)
LEAD_VERB_RE = re.compile(rf"^(?:{TASK_VERBS})\b[\s,:-]*", re.I)

META_RE = re.compile(
    r"(?:\.\\s*\\.\\s*\\.|\.{3}|_{3,}|fill\s+in\s+the\s+blank|to\s+be\s+completed)",
    re.I,
)
TRUNCATED_TAIL_RE = re.compile(
    r"(?:\b(?:what|how|why|which|when|where)\s+can\s+be$|"
    r"\b(?:when|where|because|although|if|whether)\s+[^.?!]*$)",
    re.I,
)

OUTPUT_TAIL_RE = re.compile(
    r"\s*(?:,?\s*)"
    r"(?:show|provide|support|justify|defend)\s+"
    r"(?:your|the|each|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"an?|relevant|appropriate)\b.*$",
    re.I,
)
EXAMPLE_TAIL_RE = re.compile(
    r"\s*(?:with|using|provide|give|include)\s+"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:relevant|appropriate|real[- ]life)?\s*examples?\b.*$",
    re.I,
)
FOLLOWING_TAIL_RE = re.compile(
    r"\s+(?:on the following page|on the next page|below|above)\s*$", re.I
)
PURPOSE_TAIL_RE = re.compile(
    r"\s+(?:in order to|so that|so as to|to)\s+"
    r"(?:help|allow|enable|let|make it possible|work out|answer|determine|"
    r"complete|solve|identify|find|improve|prevent)\b.*$",
    re.I,
)

GENERIC_TOPIC_RE = re.compile(
    r"^(?:this|that|these|those|it|the following|the above|the below|"
    r"something|anything|everything|the process|the method|the statement|"
    r"the answer|the problem|the question|the concept|the idea|the model)$",
    re.I,
)

SPECIALS: list[tuple[re.Pattern[str], object]] = [
    (
        re.compile(
            r"^how\s+(?:was|were|is|are)\s+(.+?)\s+similar\s+to\s+(.+)\??$",
            re.I,
        ),
        lambda m: f"the similarity between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(
            r"^how\s+(?:does|do)\s+(.+?)\s+compare\s+(?:with|to)\s+(.+)\??$",
            re.I,
        ),
        lambda m: f"the comparison between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(
            r"^how\s+(?:does|do)\s+(.+?)\s+differ\s+from\s+(.+)\??$",
            re.I,
        ),
        lambda m: f"the difference between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(
            r"^how\s+does\s+(.+?)\s+affect\s+(.+)\??$",
            re.I,
        ),
        lambda m: f"the effect of {m.group(1)} on {m.group(2)}",
    ),
    (
        re.compile(
            r"^how\s+would\s+you\s+(?:relate|link|connect)\s+(.+?)\s+(?:with|to)\s+(.+)\??$",
            re.I,
        ),
        lambda m: f"the relationship between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(
            r"^how\s+would\s+you\s+(?:differentiate|distinguish)\s+(.+?)\s+from\s+(.+)\??$",
            re.I,
        ),
        lambda m: f"the distinction between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(
            r"^how\s+would\s+you\s+(?:classify|categorize|categorise)\s+(.+)\??$",
            re.I,
        ),
        lambda m: f"the classification of {m.group(1)}",
    ),
    (
        re.compile(
            r"^how\s+(.+?)\s+(?:will|would|could|can|may|might)\s+be\s+more\s+useful\s+than\s+(.+?)(?:\s+in\s+(.+))?$",
            re.I,
        ),
        lambda m: (
            f"relative usefulness of {m.group(1)} compared with {m.group(2)}"
            + (f" in {m.group(3)}" if m.group(3) else "")
        ),
    ),
    (
        re.compile(
            r"^how\s+(.+?)\s+is\s+represented\s+by\s+(.+)$", re.I
        ),
        lambda m: f"representation of {m.group(1)} by {m.group(2)}",
    ),
    (
        re.compile(r"^how\s+(.+?)\s+(?:are|is)\s+combined\b.*$", re.I),
        lambda m: f"combination of {m.group(1)}",
    ),
    (
        re.compile(r"^how\s+(.+?)\s+(?:is|are)\s+used\b.*$", re.I),
        lambda m: f"use of {m.group(1)}",
    ),
    (
        re.compile(r"^how\s+(.+?)\s+(?:works|work)$", re.I),
        lambda m: f"functioning of {m.group(1)}",
    ),
    (
        re.compile(
            r"^what\s+(?:does|do)\s+(.+?)\s+(?:mean|refer to|stand for)\??$",
            re.I,
        ),
        lambda m: f"the meaning of {m.group(1)}",
    ),
    (
        re.compile(r"^what\s+(?:types?)\s+of\s+(.+)$", re.I),
        lambda m: f"types of {m.group(1)}",
    ),
    (
        re.compile(r"^what\s+(?:is|are)\s+(.+?)\??$", re.I),
        lambda m: m.group(1),
    ),
    (
        re.compile(r"^what\s+(?:happens|happened)\s+if\s+(.+)$", re.I),
        lambda m: f"the effects of {m.group(1)}",
    ),
    (
        re.compile(
            r"^what\s+(?:would|should)\s+you\s+do\s+if\s+(.+)$", re.I
        ),
        lambda m: f"appropriate action when {m.group(1)}",
    ),
    (
        re.compile(
            r"^what\s+solutions\s+would\s+you\s+suggest\s+for\s+(.+)$", re.I
        ),
        lambda m: f"possible solutions for {m.group(1)}",
    ),
    (
        re.compile(
            r"^what\s+criteria\s+would\s+you\s+use\s+to\s+evaluate\s+(.+)$",
            re.I,
        ),
        lambda m: f"criteria for evaluating {m.group(1)}",
    ),
    (
        re.compile(
            r"^why\s+(?:does|do|did|is|are|was|were)\s+(.+)$", re.I
        ),
        lambda m: f"the reasons for {m.group(1)}",
    ),
    (
        re.compile(r"^why\s+(.+)$", re.I),
        lambda m: f"the reasons for {m.group(1)}",
    ),
    (
        re.compile(r"^which\s+of\s+the\s+following\s+(.+)$", re.I),
        lambda m: f"the appropriate {m.group(1)}",
    ),
    (
        re.compile(r"^between\s+(.+?)\s+and\s+(.+)$", re.I),
        lambda m: f"the distinction between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(r"^among\s+(.+)$", re.I),
        lambda m: f"the distinctions among {m.group(1)}",
    ),
]

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().strip('"').strip("'").strip()

def hash_id(text: str) -> str:
    return hashlib.sha256(clean(text).encode("utf-8")).hexdigest()[:16]

def content_tokens(text: str) -> set[str]:
    return {
        x for x in re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())
        if len(x) > 2
    }

def looks_like_task(text: str) -> bool:
    s = clean(text)
    if not s or META_RE.search(s):
        return False
    if len(s.split()) > 90:
        return False
    if "?" in s:
        return True
    return bool(
        re.match(r"^(?:what|why|how|which|who|when|where)\b", s, re.I)
        or re.match(rf"^(?:please\s+)?(?:{TASK_VERBS})\b", s, re.I)
        or re.search(rf"\.\s*(?:please\s+)?(?:{TASK_VERBS})\b", s, re.I)
        or re.match(
            r"^(?:briefly|critically|concisely|carefully)\s+(?:"
            + TASK_VERBS
            + r")\b",
            s,
            re.I,
        )
    )

def extract_topic(source: str) -> str | None:
    s = clean(source)
    # Use the last clear task sentence when explanatory context precedes it.
    parts = re.split(r"(?<=[.!?])\s+", s)
    task_part = None
    for part in reversed(parts):
        if "?" in part or re.match(
            rf"^(?:please\s+)?(?:briefly\s+|critically\s+|concisely\s+|carefully\s+)?(?:{TASK_VERBS})\b",
            part,
            re.I,
        ):
            task_part = part.strip()
            break
    if task_part:
        s = task_part
    s = META_RE.sub("", s).strip()
    for _ in range(5):
        before = s
        s = PREFIX_RE.sub("", s).strip()
        s = LEAD_VERB_RE.sub("", s).strip()
        s = re.sub(r"^(?:and|or)\s+", "", s, flags=re.I).strip()
        s = re.sub(
            r"^(?:briefly|clearly|concisely|critically|carefully)\s+", "",
            s, flags=re.I
        ).strip()
        if s == before:
            break
    s = re.sub(
        r"^(?:to\s+)?(?:illustrate|show|explain|describe|indicate)\s+",
        "", s, flags=re.I,
    ).strip()
    # Specific interrogatives.
    for pat, fn in SPECIALS:
        m = pat.match(s)
        if m:
            s = clean(fn(m))
            break
    # Generic WH nominalization.
    if re.match(r"^how\s+far\s+", s, re.I):
        s = re.sub(r"^how\s+far\s+", "the extent to which ", s, flags=re.I)
    elif re.match(r"^how\s+", s, re.I):
        m = re.match(r"^how\s+(.+)$", s, re.I)
        if m:
            s = f"the way {m.group(1)}"
    elif re.match(r"^why\s+", s, re.I):
        s = re.sub(r"^why\s+", "the reasons for ", s, flags=re.I)
    elif re.match(r"^what\s+", s, re.I):
        s = re.sub(r"^what\s+", "the meaning of ", s, flags=re.I)
    elif re.match(r"^which\s+", s, re.I):
        s = re.sub(r"^which\s+", "the choice among ", s, flags=re.I)
    elif re.match(r"^(?:whether|if)\s+", s, re.I):
        s = "the question of " + s
    # Remove obvious output scaffolding after the core topic.
    s = OUTPUT_TAIL_RE.sub("", s).strip()
    s = EXAMPLE_TAIL_RE.sub("", s).strip()
    s = PURPOSE_TAIL_RE.sub("", s).strip()
    s = FOLLOWING_TAIL_RE.sub("", s).strip()
    # Remove a second instruction after an explicit conjunction.
    s = re.sub(
        rf"\s+and\s+(?:{TASK_VERBS})\b.*$",
        "",
        s,
        flags=re.I,
    ).strip()
    s = re.sub(r"\s+(?:and|or|with|of|in|on|for|to)\s*$", "", s, flags=re.I)
    s = re.sub(r'^[,:;\\-]+|[,:;\\-]+$', '', s).strip()
    s = s.rstrip(" .,:;?")
    if len(s) < 3 or GENERIC_TOPIC_RE.fullmatch(s):
        return None
    if re.search(
        r"\b(?:show your working|justify your answer|support your answer|"
        r"provide a relevant example|target bloom level|original question)\b",
        s,
        re.I,
    ):
        return None
    if re.match(
        r"^(?:how|why|what|which|whether|if|can|could|would|should|do|does|did)\b",
        s,
        re.I,
    ):
        return None
    return s

TEMPLATES = {
    "Remember": (
        "What key facts about {topic} should be recalled?",
        "State the main characteristics of {topic}.",
        "Identify the essential components or terms associated with {topic}.",
        "Which key facts about {topic} should a student be able to recall?",
        "Name the principal elements of {topic}.",
        "List the essential facts associated with {topic}.",
    ),
    "Understand": (
        "Explain {topic}.",
        "Describe the main idea and purpose of {topic}.",
        "Summarize what {topic} means in its academic context.",
        "How would you explain {topic} to a learner unfamiliar with it?",
        "Describe the role and significance of {topic}.",
        "Explain the relationship or meaning expressed by {topic}.",
    ),
    "Apply": (
        "How would you apply knowledge related to {topic} to a concrete case?",
        "Given a concrete case involving {topic}, determine the appropriate result or action.",
        "Demonstrate how knowledge of {topic} would be used in a practical situation.",
        "Use the relevant knowledge about {topic} to solve a concrete problem.",
        "Apply knowledge related to {topic} in a specified case and determine the outcome.",
        "In a practical scenario involving {topic}, determine the appropriate result or action.",
    ),
    "Analyze": (
        "Analyze {topic} by examining its components, relationships, causes, or patterns.",
        "Examine the structure of {topic} and identify the relationships among its parts.",
        "Compare relevant aspects of {topic} and explain how they relate.",
        "Break down {topic} into components and analyze how those components interact.",
        "What relationships, contrasts, or patterns can be identified within {topic}?",
        "Examine the causes and underlying structure of {topic}.",
    ),
    "Evaluate": (
        "Evaluate {topic} using explicit criteria and supporting evidence, and justify your judgment.",
        "Assess the strengths and limitations of {topic} against relevant criteria.",
        "Critique {topic} by considering its validity, suitability, and trade-offs.",
        "How effective is {topic} according to stated criteria, and why?",
        "Judge the quality or suitability of {topic} using evidence and clear standards.",
        "Defend a reasoned judgment about {topic} based on explicit criteria.",
    ),
    "Create": (
        "Design a new solution or plan related to {topic} for a defined problem.",
        "Develop an original approach incorporating {topic} under stated requirements.",
        "Propose a new strategy or artifact based on {topic} for a specified need.",
        "Formulate a new solution involving {topic} under given constraints.",
        "How would you construct an original solution using {topic}?",
        "Create a new plan or artifact that incorporates {topic} for a clearly defined purpose.",
    ),
}

CUES = {
    "Remember": ("facts", "recall", "state", "identify", "name", "list", "define", "characteristics"),
    "Understand": ("explain", "describe", "summarize", "meaning", "purpose"),
    "Apply": ("apply", "concrete", "case", "procedure", "problem", "determine", "practical", "scenario", "use"),
    "Analyze": ("analyze", "examine", "compare", "components", "relationships", "structure", "patterns", "causes"),
    "Evaluate": ("evaluate", "assess", "critique", "criteria", "evidence", "justify", "judgment", "effective", "judge"),
    "Create": ("design", "develop", "construct", "original", "artifact", "strategy", "solution", "constraints", "formulate", "propose", "create"),
}

def build_rewrite(topic: str, target: str, source: str, sid: str) -> tuple[str, int]:
    base = int(hashlib.sha256(f"{sid}|{target}".encode()).hexdigest()[:8], 16)
    source_norm = re.sub(r"[^a-z0-9\s]", " ", source.lower())
    source_norm = re.sub(r"\s+", " ", source_norm).strip()
    for off in range(len(TEMPLATES[target])):
        idx = (base + off) % len(TEMPLATES[target])
        candidate = TEMPLATES[target][idx].format(topic=topic)
        cand_norm = re.sub(r"[^a-z0-9\s]", " ", candidate.lower())
        cand_norm = re.sub(r"\s+", " ", cand_norm).strip()
        if cand_norm == source_norm:
            continue
        if target == "Remember" and not any(c in cand_norm for c in CUES[target]):
            continue
        if not any(c in cand_norm for c in CUES[target]):
            continue
        tt = content_tokens(topic)
        rr = content_tokens(candidate)
        if tt and len(tt & rr) / len(tt) < 0.45:
            continue
        return candidate, idx
    raise ValueError("no_valid_template")

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def make_row(source: dict[str, str], split: str, target: str, topic: str) -> dict:
    question = clean(source["question"])
    original_level = clean(source["bloom_level"])
    sid = "src_" + hash_id(question)
    if target == original_level:
        rewrite = question
        template = "identity"
        synthetic = False
    else:
        rewrite, idx = build_rewrite(topic, target, question, sid)
        template = f"{target.lower()}_{idx}"
        synthetic = True
    user = f"Original question:\n{question}\n\nTarget Bloom level:\n{target}"
    sft = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{rewrite}<|im_end|>"
    )
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>"
    return {
        "example_id": hashlib.sha256(f"{sid}|{target}".encode()).hexdigest()[:16],
        "task": "bloom_rewrite",
        "source_id": sid,
        "group_id": sid[4:],
        "split": split,
        "source_question": question,
        "source_bloom_level": original_level,
        "target_bloom_level": target,
        "target_rewrite": rewrite,
        "source_topic": topic,
        "synthetic": synthetic,
        "transformation_type": f"{original_level}->{target}",
        "construction_method": "deterministic_source_anchored_template_transformation",
        "construction_template": template,
        "dataset_version": "figshare_target_rewrite_v1",
        "policy_version": "figshare_bloom_target_policy_v1",
        "source_file": str(INPUTS[split].relative_to(ROOT)).replace("\\", "/"),
        "quality_status": "pass",
        "prompt_text": prompt,
        "sft_text": sft,
        "text": sft,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": rewrite},
        ],
    }

def main() -> None:
    all_rows: list[dict] = []
    rejected: list[dict] = []
    split_counts = {}
    source_sets = {}

    for split, path in INPUTS.items():
        rows = read_csv(path)
        source_sets[split] = set()
        kept = []
        for source in rows:
            q = clean(source.get("question", ""))
            source["question"] = q
            if not looks_like_task(q):
                rejected.append({"split": split, "question": q, "reason": "invalid_source_form"})
                continue
            topic = extract_topic(q)
            if topic is None:
                rejected.append({"split": split, "question": q, "reason": "unextractable_topic"})
                continue
            sid = "src_" + hash_id(q)
            if sid in source_sets[split]:
                rejected.append({"split": split, "question": q, "reason": "duplicate_source_id"})
                continue
            source_sets[split].add(sid)
            generated = []
            failed = None
            for target in LEVELS:
                try:
                    generated.append(make_row(source, split, target, topic))
                except Exception as exc:
                    failed = {
                        "split": split,
                        "source_id": sid,
                        "question": q,
                        "source_bloom_level": source.get("bloom_level", ""),
                        "reason": "generation_failed",
                        "detail": str(exc),
                        "target": target,
                        "topic": topic,
                    }
                    break
            if failed:
                rejected.append(failed)
                continue
            kept.extend(generated)
        all_rows.extend(kept)
        split_counts[split] = Counter(r["target_bloom_level"] for r in kept)

    # Hard source-level leakage check.
    split_ids = {k: {r["source_id"] for r in all_rows if r["split"] == k} for k in INPUTS}
    overlaps = {
        "train_validation": len(split_ids["train"] & split_ids["validation"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "validation_test": len(split_ids["validation"] & split_ids["test"]),
    }
    if any(overlaps.values()):
        raise SystemExit(f"Source leakage detected: {overlaps}")

    # Every accepted source must have exactly six targets.
    per_source = defaultdict(list)
    for r in all_rows:
        per_source[r["source_id"]].append(r["target_bloom_level"])
    incomplete = [sid for sid, ts in per_source.items() if set(ts) != set(LEVELS) or len(ts) != 6]
    if incomplete:
        raise SystemExit(f"Incomplete six-target source groups: {len(incomplete)}")

    # Final output checks.
    duplicate_ids = len(all_rows) - len({r["example_id"] for r in all_rows})
    if duplicate_ids:
        raise SystemExit(f"Duplicate example IDs: {duplicate_ids}")

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows.sort(key=lambda r: (r["split"], r["source_id"], LEVELS.index(r["target_bloom_level"])))

    for split in INPUTS:
        rows = [r for r in all_rows if r["split"] == split]
        (OUT / f"bloom_rewrite_{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )

    (OUT / "bloom_rewrite_all.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows),
        encoding="utf-8",
    )
    stats = {
        "source_of_truth": {k: str(v.relative_to(ROOT)).replace("\\", "/") for k, v in INPUTS.items()},
        "source_rows": {k: len(source_sets[k]) for k in INPUTS},
        "rewrite_rows": {
            k: sum(1 for r in all_rows if r["split"] == k) for k in INPUTS
        },
        "targets_per_source": 6,
        "target_levels": list(LEVELS),
        "target_counts": Counter(r["target_bloom_level"] for r in all_rows),
        "splits": {
            k: dict(split_counts[k]) for k in INPUTS
        },
        "rejected_source_rows": len(rejected),
        "rejection_reasons": Counter(r["reason"] for r in rejected),
        "source_split_overlap": overlaps,
        "incomplete_source_groups": len(incomplete),
        "duplicate_example_ids": duplicate_ids,
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "rejected_sources.json").write_text(
        json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "dataset_version": "figshare_target_rewrite_v1",
        "policy_version": "figshare_bloom_target_policy_v1",
        "source_of_truth": "Figshare Bloom dataset already checked into data/figshare_bloom_v1_*.csv",
        "generation": "deterministic source-anchored target-level transformation",
        "target_levels": list(LEVELS),
        "same_level_behavior": "identity copy",
        "cross_level_behavior": "template transformation with source-topic anchoring",
        "training_files": {
            "train": "bloom_rewrite_train.jsonl",
            "validation": "bloom_rewrite_validation.jsonl",
            "test": "bloom_rewrite_test.jsonl",
        },
        "checks": {
            "six_targets_per_accepted_source": len(incomplete) == 0,
            "source_level_split_disjoint": not any(overlaps.values()),
            "duplicate_example_ids": duplicate_ids == 0,
            "all_rows_quality_status_pass": all(r["quality_status"] == "pass" for r in all_rows),
        },
        "stats": stats,
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
