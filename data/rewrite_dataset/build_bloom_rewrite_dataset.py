#!/usr/bin/env python3
"""Build the final Figshare-backed Bloom target-level rewrite dataset."""
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
    "criticise|rank|rate|place|judge|defend|correct|find|detect|fix|sketch|"
    "estimate|label|verify|paraphrase|perform|produce|devise|prepare|plan|"
    "relate|differ|inspect|investigate|prove|test|validate|generalize|"
    "generalise|restate|restating|express|model|combine|categorize|categorise|"
    "group|argue|review|comment|appraise|synthesize|synthesise"
)

TASK_RE = re.compile(rf"\\b(?:{TASK_VERBS})\\b", re.I)
PLACEHOLDER_RE = re.compile(
    r"(?:\\.\\s*\\.\\s*\\.|\\.{3}|_{3,}|fill\\s+in\\s+the\\s+blank|"
    r"\\b(?:tbd|to be completed)\\b)",
    re.I,
)
TRUNC_RE = re.compile(r"(?:\\b(?:and|or|to|of|in|for|with|because|although)\\s*)$")
PREFIX_RE = re.compile(
    r"^(?:(?:please|briefly|clearly|concisely|carefully|critically)\\s+)+"
    r"|^(?:based on (?:your|the) understanding|in your own words)[,:]?\\s*|"
    r"^(?:with (?:the )?aid of)\\s+[^,;:]+[,:]\\s*|"
    r"^(?:with|using|by using)\\s+(?:the\\s+|an?\\s+)?"
    r"(?:appropriate|relevant)\\s+[^,;:]+[,:]\\s*",
    re.I,
)
LEAD_RE = re.compile(rf"^(?:{TASK_VERBS})\\b[\\s,:-]*", re.I)

GENERIC_REF_RE = re.compile(
    r"^(?:this statement|the statement|your answer|the answer|your position|"
    r"the following|the above|evidence|arguments?|justifications?|the decision|"
    r"the recommendation|the recommendations|the choice|the result)\\b",
    re.I,
)
OUTPUT_TAIL_RE = re.compile(
    r"\\s*(?:,?\\s+)?(?:show|provide|support|justify|defend|comment)\\s+"
    r"(?:your|the|each|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"an?|relevant|appropriate)\\b.*$",
    re.I,
)
EXAMPLE_TAIL_RE = re.compile(
    r"\\s*(?:with|using|provide|give|include)\\s+"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\\d+)\\s+"
    r"(?:relevant|appropriate|real[- ]life)?\\s*examples?\\b.*$",
    re.I,
)
SECONDARY_TASK_RE = re.compile(
    rf"\\s+(?:and|or)\\s+(?:justify|support|defend|provide|show|comment|"
    rf"explain|describe|discuss|analyze|analyse|evaluate|assess|critique|"
    rf"identify|list|give|state|illustrate)\\b.*$",
    re.I,
)

SPECIALS: list[tuple[re.Pattern[str], object]] = [
    (
        re.compile(r"^how\\s+(?:was|were|is|are)\\s+(.+?)\\s+similar\\s+to\\s+(.+)\\??$", re.I),
        lambda m: f"the similarity between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(r"^how\\s+(?:does|do)\\s+(.+?)\\s+compare\\s+(?:with|to)\\s+(.+)\\??$", re.I),
        lambda m: f"the comparison between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(r"^how\\s+(?:does|do)\\s+(.+?)\\s+differ\\s+from\\s+(.+)\\??$", re.I),
        lambda m: f"the difference between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(r"^how\\s+does\\s+(.+?)\\s+affect\\s+(.+)\\??$", re.I),
        lambda m: f"the effect of {m.group(1)} on {m.group(2)}",
    ),
    (
        re.compile(r"^how\\s+would\\s+you\\s+(?:relate|link|connect)\\s+(.+?)\\s+(?:with|to)\\s+(.+)\\??$", re.I),
        lambda m: f"the relationship between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(r"^how\\s+would\\s+you\\s+(?:differentiate|distinguish)\\s+(.+?)\\s+from\\s+(.+)\\??$", re.I),
        lambda m: f"the distinction between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(r"^how\\s+would\\s+you\\s+(?:classify|categorize|categorise)\\s+(.+)$", re.I),
        lambda m: f"the classification of {m.group(1)}",
    ),
    (
        re.compile(
            r"^how\\s+(.+?)\\s+(?:will|would|could|can|may|might)\\s+be\\s+more\\s+useful\\s+than\\s+(.+?)(?:\\s+in\\s+(.+))?$",
            re.I,
        ),
        lambda m: f"relative usefulness of {m.group(1)} compared with {m.group(2)}"
        + (f" in {m.group(3)}" if m.group(3) else ""),
    ),
    (
        re.compile(r"^how\\s+(.+?)\\s+is\\s+represented\\s+by\\s+(.+)$", re.I),
        lambda m: f"representation of {m.group(1)} by {m.group(2)}",
    ),
    (
        re.compile(r"^how\\s+(.+?)\\s+(?:are|is)\\s+combined\\b.*$", re.I),
        lambda m: f"combination of {m.group(1)}",
    ),
    (
        re.compile(r"^how\\s+(.+?)\\s+(?:is|are)\\s+used\\b.*$", re.I),
        lambda m: f"use of {m.group(1)}",
    ),
    (
        re.compile(r"^how\\s+(.+?)\\s+(?:works|work)$", re.I),
        lambda m: f"functioning of {m.group(1)}",
    ),
    (
        re.compile(r"^what\\s+(?:does|do)\\s+(.+?)\\s+(?:mean|refer to|stand for)\\??$", re.I),
        lambda m: f"the meaning of {m.group(1)}",
    ),
    (
        re.compile(r"^what\\s+types?\\s+of\\s+(.+)$", re.I),
        lambda m: f"types of {m.group(1)}",
    ),
    (
        re.compile(r"^what\\s+(?:is|are)\\s+(.+?)\\??$", re.I),
        lambda m: m.group(1),
    ),
    (
        re.compile(r"^what\\s+(?:happens|happened)\\s+if\\s+(.+)$", re.I),
        lambda m: f"the effects of {m.group(1)}",
    ),
    (
        re.compile(r"^what\\s+(?:would|should)\\s+you\\s+do\\s+if\\s+(.+)$", re.I),
        lambda m: f"appropriate action when {m.group(1)}",
    ),
    (
        re.compile(r"^what\\s+solutions\\s+would\\s+you\\s+suggest\\s+for\\s+(.+)$", re.I),
        lambda m: f"possible solutions for {m.group(1)}",
    ),
    (
        re.compile(r"^what\\s+criteria\\s+would\\s+you\\s+use\\s+to\\s+evaluate\\s+(.+)$", re.I),
        lambda m: f"criteria for evaluating {m.group(1)}",
    ),
    (
        re.compile(r"^why\\s+(?:does|do|did|is|are|was|were)\\s+(.+)$", re.I),
        lambda m: f"the reasons for {m.group(1)}",
    ),
    (
        re.compile(r"^why\\s+(.+)$", re.I),
        lambda m: f"the reasons for {m.group(1)}",
    ),
    (
        re.compile(r"^which\\s+of\\s+the\\s+following\\s+(.+)$", re.I),
        lambda m: f"the appropriate {m.group(1)}",
    ),
    (
        re.compile(r"^between\\s+(.+?)\\s+and\\s+(.+)$", re.I),
        lambda m: f"the distinction between {m.group(1)} and {m.group(2)}",
    ),
    (
        re.compile(r"^among\\s+(.+)$", re.I),
        lambda m: f"the distinctions among {m.group(1)}",
    ),
]

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
    "Remember": ("recall", "state", "identify", "name", "list", "facts", "characteristics"),
    "Understand": ("explain", "describe", "summarize", "meaning", "purpose"),
    "Apply": ("apply", "concrete", "case", "determine", "practical", "scenario", "procedure", "problem", "use"),
    "Analyze": ("analyze", "examine", "compare", "components", "relationships", "structure", "patterns", "causes"),
    "Evaluate": ("evaluate", "assess", "critique", "criteria", "evidence", "justify", "judgment", "effective", "judge"),
    "Create": ("design", "develop", "construct", "original", "artifact", "strategy", "solution", "constraints", "formulate", "propose", "create"),
}

def clean(text: str) -> str:
    return re.sub(r"\\s+", " ", str(text)).strip().strip('"').strip("'").strip()

def norm(text: str) -> str:
    return re.sub(r"\\s+", " ", re.sub(r"[^a-z0-9\\s]", " ", text.lower())).strip()

def hash_id(text: str) -> str:
    return hashlib.sha256(clean(text).encode("utf-8")).hexdigest()[:16]

def has_task_or_question(text: str) -> bool:
    s = clean(text)
    return bool(s and not PLACEHOLDER_RE.search(s) and ("?" in s or TASK_RE.search(s)))

def extract_topic(source: str) -> str | None:
    s = clean(source)
    if not has_task_or_question(s):
        return None

    sentences = re.split(r"(?<=[.!?])\\s+", s)
    task_sentence = None
    context = ""
    for sentence in reversed(sentences):
        if "?" in sentence or TASK_RE.search(sentence):
            task_sentence = sentence.strip()
            prefix = s[: s.rfind(sentence)].strip(" .,:;-")
            context = prefix
            break
    if task_sentence is None:
        task_sentence = s

    # Take the first explicit task operator in the task sentence.
    match = TASK_RE.search(task_sentence)
    if match:
        body = task_sentence[match.end():].strip(" ,:;-")
        before_task = task_sentence[:match.start()].strip()
    else:
        body = task_sentence
        before_task = ""

    # When the operator points to a generic reference, use the immediate context.
    if GENERIC_REF_RE.match(body) and context:
        ctx_parts = re.split(r"(?<=[.!?])\\s+", context)
        body = ctx_parts[-1].strip(" .,:;-") if ctx_parts else context

    body = PLACEHOLDER_RE.sub("", body).strip()
    for _ in range(6):
        before = body
        body = PREFIX_RE.sub("", body).strip()
        body = LEAD_RE.sub("", body).strip(" ,:;-")
        body = re.sub(r"^(?:and|or)\\s+", "", body, flags=re.I).strip()
        body = re.sub(r"^(?:briefly|clearly|concisely|critically|carefully)\\s+", "", body, flags=re.I).strip()
        if body == before:
            break

    for pattern, fn in SPECIALS:
        m = pattern.match(body)
        if m:
            body = clean(fn(m))
            break

    if re.match(r"^how\\s+far\\s+", body, re.I):
        body = re.sub(r"^how\\s+far\\s+", "the extent to which ", body, flags=re.I)
    elif re.match(r"^how\\s+", body, re.I):
        body = re.sub(r"^how\\s+", "the way ", body, flags=re.I)
    elif re.match(r"^why\\s+", body, re.I):
        body = re.sub(r"^why\\s+", "the reasons for ", body, flags=re.I)
    elif re.match(r"^what\\s+", body, re.I):
        body = re.sub(r"^what\\s+", "the meaning or description of ", body, flags=re.I)
    elif re.match(r"^which\\s+", body, re.I):
        body = re.sub(r"^which\\s+", "the choice among ", body, flags=re.I)
    elif re.match(r"^(?:whether|if)\\s+", body, re.I):
        body = "the question of " + body

    body = EXAMPLE_TAIL_RE.sub("", body).strip()
    body = OUTPUT_TAIL_RE.sub("", body).strip()
    body = SECONDARY_TASK_RE.sub("", body).strip()
    body = re.sub(r"\\s+(?:and|or|with|of|in|on|for|to)\\s*$", "", body, flags=re.I)
    body = re.sub(r"^[,:;\\-]+|[,:;\\-]+$", "", body).strip()
    body = clean(body).rstrip(" .,:;?")

    if len(body) < 3 or GENERIC_REF_RE.fullmatch(body):
        return None
    if PLACEHOLDER_RE.search(body) or TRUNC_RE.search(body):
        return None
    if re.match(r"^(?:how|why|what|which|whether|if|can|could|would|should|do|does|did)\\b", body, re.I):
        return None
    return body

def build_rewrite(topic: str, target: str, source: str, source_id: str) -> tuple[str, int]:
    base = int(hashlib.sha256(f"{source_id}|{target}".encode()).hexdigest()[:8], 16)
    source_n = norm(source)
    for offset in range(len(TEMPLATES[target])):
        idx = (base + offset) % len(TEMPLATES[target])
        candidate = TEMPLATES[target][idx].format(topic=topic)
        cn = norm(candidate)
        if cn == source_n:
            continue
        if not any(c in cn for c in CUES[target]):
            continue
        return candidate, idx
    raise ValueError("no_valid_template")

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def make_record(source: dict[str, str], split: str, target: str, topic: str) -> dict:
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

    user = f"Original question:\\n{question}\\n\\nTarget Bloom level:\\n{target}"
    sft = (
        f"<|im_start|>system\\n{SYSTEM}<|im_end|>\\n"
        f"<|im_start|>user\\n{user}<|im_end|>\\n"
        f"<|im_start|>assistant\\n{rewrite}<|im_end|>"
    )
    prompt = f"<|im_start|>system\\n{SYSTEM}<|im_end|>\\n<|im_start|>user\\n{user}<|im_end|>"

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
        "dataset_version": "figshare_target_rewrite_v2",
        "policy_version": "figshare_bloom_target_policy_v2",
        "source_file": str(INPUTS[split].relative_to(ROOT)).replace("\\\\", "/"),
        "quality_status": "pass",
        "validation": {
            "source_topic_nonempty": bool(topic),
            "cross_level_not_identity": (not synthetic) or (norm(question) != norm(rewrite)),
            "target_cue_present": any(c in norm(rewrite) for c in CUES[target]),
            "meta_free": not bool(re.search(r"as an ai|target bloom level|original question|rewritten question|the answer is", rewrite, re.I)),
        },
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
    source_sets: dict[str, set[str]] = {k: set() for k in INPUTS}

    for split, path in INPUTS.items():
        accepted_rows = []
        for source in read_rows(path):
            question = clean(source.get("question", ""))
            if not has_task_or_question(question):
                rejected.append({"split": split, "question": question, "reason": "invalid_source_or_placeholder"})
                continue
            topic = extract_topic(question)
            if topic is None:
                rejected.append({"split": split, "question": question, "reason": "unextractable_or_truncated_topic"})
                continue
            sid = "src_" + hash_id(question)
            if sid in source_sets[split]:
                rejected.append({"split": split, "question": question, "reason": "duplicate_source"})
                continue
            source_sets[split].add(sid)

            group = []
            failed = None
            for target in LEVELS:
                try:
                    record = make_record(source, split, target, topic)
                except Exception as exc:
                    failed = {"split": split, "source_id": sid, "question": question, "reason": "generation_failed", "target": target, "detail": str(exc)}
                    break
                if not all(record["validation"].values()):
                    failed = {"split": split, "source_id": sid, "question": question, "reason": "validation_failed", "target": target}
                    break
                group.append(record)

            if failed:
                rejected.append(failed)
                source_sets[split].discard(sid)
                continue
            accepted_rows.extend(group)

        all_rows.extend(accepted_rows)

    split_ids = {k: {r["source_id"] for r in all_rows if r["split"] == k} for k in INPUTS}
    overlaps = {
        "train_validation": len(split_ids["train"] & split_ids["validation"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "validation_test": len(split_ids["validation"] & split_ids["test"]),
    }
    if any(overlaps.values()):
        raise SystemExit(f"Source leakage detected: {overlaps}")

    per_source = defaultdict(list)
    for row in all_rows:
        per_source[row["source_id"]].append(row["target_bloom_level"])
    incomplete = [sid for sid, targets in per_source.items() if len(targets) != 6 or set(targets) != set(LEVELS)]
    if incomplete:
        raise SystemExit(f"Incomplete source groups: {len(incomplete)}")
    duplicate_ids = len(all_rows) - len({r["example_id"] for r in all_rows})
    if duplicate_ids:
        raise SystemExit(f"Duplicate example IDs: {duplicate_ids}")

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows.sort(key=lambda r: (r["split"], r["source_id"], LEVELS.index(r["target_bloom_level"])))
    for split in INPUTS:
        rows = [r for r in all_rows if r["split"] == split]
        (OUT / f"bloom_rewrite_{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\\n" for r in rows),
            encoding="utf-8",
        )
    (OUT / "bloom_rewrite_all.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\\n" for r in all_rows),
        encoding="utf-8",
    )

    stats = {
        "source_of_truth": {k: str(v.relative_to(ROOT)).replace("\\\\", "/") for k, v in INPUTS.items()},
        "source_rows_in_figshare_splits": {k: len(read_rows(v)) for k, v in INPUTS.items()},
        "accepted_source_rows": {k: len(source_sets[k]) for k in INPUTS},
        "rewrite_rows": {k: sum(1 for r in all_rows if r["split"] == k) for k in INPUTS},
        "targets_per_source": 6,
        "target_counts": dict(Counter(r["target_bloom_level"] for r in all_rows)),
        "rejected_source_rows": len(rejected),
        "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
        "source_split_overlap": overlaps,
        "incomplete_source_groups": len(incomplete),
        "duplicate_example_ids": duplicate_ids,
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "rejected_sources.json").write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "dataset_version": "figshare_target_rewrite_v2",
        "policy_version": "figshare_bloom_target_policy_v2",
        "source_of_truth": "Figshare Bloom dataset split files in data/figshare_bloom_v1_{train,val,test}.csv",
        "generation": "deterministic source-anchored target-level transformation",
        "target_levels": list(LEVELS),
        "same_level_behavior": "identity copy",
        "cross_level_behavior": "source-topic-anchored cognitive-operation transformation",
        "training_files": {
            "train": "bloom_rewrite_train.jsonl",
            "validation": "bloom_rewrite_validation.jsonl",
            "test": "bloom_rewrite_test.jsonl",
        },
        "checks": {
            "six_targets_per_accepted_source": len(incomplete) == 0,
            "source_level_split_disjoint": not any(overlaps.values()),
            "duplicate_example_ids_ok": duplicate_ids == 0,
            "all_rows_quality_status_pass": all(r["quality_status"] == "pass" for r in all_rows),
        },
        "stats": stats,
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
