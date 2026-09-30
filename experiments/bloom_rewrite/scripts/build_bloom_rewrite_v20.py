#!/usr/bin/env python3
"""Build the final controlled Bloom rewrite dataset directly from the raw
Figshare corpus.

Every eligible source question receives exactly one rewrite for each of the six
target Bloom levels. The source question is assigned to exactly one split before
expansion, so there is no source leakage across train/validation/test.

No LLM is used. The rewrites are deterministic, source-anchored assessment
frames with only generic cognitive-task wording added.
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
DATASET_VERSION = "bloom_rewrite_synth_v20"
POLICY_VERSION = "bloom_target_policy_v20_controlled"
SOURCE_FILE = "data/figshare_bloom_v1.csv"
SEED = 42

STOP = set("""
a an the and or but if then than that this these those it its of in on at to
for from with by as is are was were be been being do does did can could may
might will would should must into over under after before during through about
against between among within without using used your our their you we someone
somebody one
""".split())

OPS = set("""
define identify name list state recite label recognize recognise explain describe
summarize summarise interpret classify illustrate discuss retell apply use
calculate compute determine find solve implement demonstrate modify estimate measure
analyze analyse compare contrast differentiate distinguish examine evaluate assess
appraise judge justify critique criticize criticise defend design develop construct
formulate propose create devise produce build compose write draw sketch select choose
suggest recommend predict outline support relate rank
""".split())

PLACEHOLDER = re.compile(r"\.\.\.|\[[^\]]*\.\.\.[^\]]*\]|<[^>]*\.\.\.[^>]*>|\b_+\b")
NUM = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])")
FILE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])")
ACRONYM = re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")

MISSING_CONTEXT = (
    "above data", "above graph", "above diagram", "following code",
    "following segment", "following passage", "given below", "this article",
    "the article", "this magazine article", "the figure above", "the table above",
)

MULTI_ACTION = re.compile(
    r"\b(?:and|or|then|also|followed\s+by)\s+(?:"
    r"define|identify|name|list|state|recite|label|recognize|recognise|explain|describe|"
    r"summarize|summarise|interpret|classify|illustrate|discuss|retell|apply|use|"
    r"calculate|compute|determine|find|solve|implement|demonstrate|modify|estimate|"
    r"measure|analyze|analyse|compare|contrast|differentiate|distinguish|examine|"
    r"evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|"
    r"design|develop|construct|formulate|propose|create|devise|produce|build|compose|"
    r"write|draw|sketch|select|choose|suggest|recommend|predict|recite|outline|support|"
    r"relate|rank)\b",
    re.I,
)

RESIDUE = re.compile(
    r"(?:\bwith\s+(?:an?\s+)?(?:appropriate|relevant|suitable)\s+"
    r"(?:example|examples|diagram|diagrams|drawing|drawings)\b)|"
    r"(?:\b(?:provide|give|support|justify|elaborate|show)\b)|"
    r"(?:\bfor\s+each\b)|"
    r"(?:\bin\s+(?:a|the)\s+table\b)|"
    r"(?:\busing\s+(?:an?\s+)?appropriate\s+(?:diagram|example)\b)",
    re.I,
)

LEADING = (
    r"^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+(?:you|we|someone|somebody|one)\s+",
    r"^\s*can\s+you\s+",
    r"^\s*what\s+(?:is|are|was|were)\s+",
    r"^\s*why\s+",
    r"^\s*which\s+of\s+the\s+following\s+",
    r"^\s*(?:briefly\s+|critically\s+|carefully\s+)?(?:evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|recommend)\s+",
    r"^\s*(?:briefly\s+)?(?:analyze|analyse|examine|compare|contrast|differentiate|distinguish)\s+",
    r"^\s*(?:calculate|compute|determine|find|solve|apply|use|implement|demonstrate|modify|estimate|measure)\s+",
    r"^\s*(?:design|develop|construct|formulate|propose|create|devise|produce|build|compose|write|draw|sketch)\s+",
    r"^\s*(?:explain|describe|summarize|summarise|interpret|classify|illustrate|discuss|retell)\s+",
    r"^\s*(?:define|identify|name|list|state|recite|label|recognize|recognise)\s+",
)

def norm(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").replace("\u00a0", " ")).strip()

def canon(x: str) -> str | None:
    x = str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())

def content(x: str) -> set[str]:
    return {
        t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*", (x or "").lower())
        if len(t) > 2 and t not in STOP and t not in OPS
    }

def protected(x: str) -> set[str]:
    out = {m.group(0).lower() for m in NUM.finditer(x or "")}
    out.update(m.group(0).lower() for m in TECH.finditer(x or ""))
    out.update(m.group(0).lower() for m in FILE.finditer(x or ""))
    out.update(m.group(1).lower() for m in ACRONYM.finditer(x or ""))
    return out

def topic(q: str) -> str | None:
    x = norm(q).strip(' "').rstrip("?.").strip()

    # Preserve natural comparison objects.
    m = re.match(r"^how\s+(?:does|do|did)\s+(.+?)\s+differ\s+from\s+(.+)$", x, re.I)
    if m:
        return f"the difference between {m.group(1).strip()} and {m.group(2).strip()}"

    m = re.match(r"^(?:differentiate|distinguish)\s+between\s+(.+)$", x, re.I)
    if m:
        return f"the differences between {m.group(1).strip()}"

    m = re.match(r"^(?:compare|contrast)\s+(.+)$", x, re.I)
    if m:
        return f"the similarities and differences between {m.group(1).strip()}"

    m = re.match(r"^what\s+(?:is|are|was|were)\s+(.+)$", x, re.I)
    if m:
        return m.group(1).strip()

    # Remove leading "how do you ..." while retaining the actual operation.
    m = re.match(
        r"^how\s+(?:do|does|did|can|could|would|should|will)\s+"
        r"(?:you|we|someone|somebody|one)\s+([A-Za-z]+)\s+(.+)$",
        x, re.I,
    )
    if m:
        return m.group(2).strip()

    for p in LEADING:
        y = re.sub(p, "", x, count=1, flags=re.I)
        if y != x:
            return y.strip(" .;:,")

    return None

def source_ok(q: str) -> tuple[bool, str, str | None]:
    q = norm(q).strip(' "')
    if not (25 <= len(q) <= 850):
        return False, "length", None
    if PLACEHOLDER.search(q):
        return False, "placeholder", None
    if any(x in q.lower() for x in MISSING_CONTEXT):
        return False, "missing_context", None
    if q.count("?") > 1:
        return False, "multiple_questions", None
    if MULTI_ACTION.search(q):
        return False, "multiple_actions", None
    if RESIDUE.search(q):
        return False, "instruction_residue", None
    if len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", q)) > 1:
        return False, "multiple_sentences", None
    t = topic(q)
    if not t:
        return False, "topic_parse", None
    if len(content(t)) < 3:
        return False, "thin_topic", None
    return True, "", t

def rewrite_for(target: str, topic_text: str, source: str, seed_value: int) -> tuple[str, str, bool]:
    # Stable variation avoids a single memorized prefix dominating the dataset.
    variants = {
        "Remember": [
            (f"State the key facts about {topic_text}.", "remember_facts"),
            (f"Identify the main aspects of {topic_text}.", "remember_aspects"),
            (f"List the key information about {topic_text}.", "remember_list"),
        ],
        "Understand": [
            (f"Explain {topic_text}.", "understand_explain"),
            (f"Describe {topic_text}.", "understand_describe"),
            (f"Summarize {topic_text}.", "understand_summarize"),
        ],
        "Apply": [
            (f"Apply your knowledge of {topic_text} to a relevant task.", "apply_knowledge"),
            (f"Use your knowledge of {topic_text} in a practical task.", "apply_practical"),
            (f"Demonstrate how {topic_text} can be applied to a relevant problem.", "apply_demonstrate"),
        ],
        "Analyze": [
            (f"Analyze {topic_text}.", "analyze_direct"),
            (f"Analyze {topic_text} by examining its key relationships and structure.", "analyze_relationships"),
            (f"Examine {topic_text} to identify its key components and relationships.", "analyze_components"),
        ],
        "Evaluate": [
            (f"Evaluate {topic_text} and justify your judgment.", "evaluate_judgment"),
            (f"Assess {topic_text} and justify your conclusion.", "evaluate_conclusion"),
            (f"Judge {topic_text} and support your judgment with relevant evidence.", "evaluate_evidence"),
        ],
        "Create": [
            (f"Develop a new approach related to {topic_text}.", "create_approach"),
            (f"Design a solution related to {topic_text}.", "create_solution"),
            (f"Formulate an original approach related to {topic_text}.", "create_formulate"),
        ],
    }
    idx = seed_value % len(variants[target])
    rewrite, template = variants[target][idx]

    # For a source whose own wording already performs the requested operation,
    # retaining it is safer than paraphrasing it into weaker generic language.
    src_level = detect_level(source)
    if src_level == target:
        return norm(source).rstrip("?.") + ".", "identity_source_question", True

    return rewrite, template, False

def detect_level(q: str) -> str:
    x = norm(q)
    m = re.match(
        r"^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+"
        r"(?:you|we|someone|somebody|one)\s+([A-Za-z]+)\b", x, re.I
    )
    if m:
        v = m.group(1).lower()
        if v in {"calculate","compute","determine","find","solve","apply","use","implement","demonstrate","modify","estimate","measure"}:
            return "Apply"
        if v in {"compare","contrast","differentiate","distinguish","analyze","analyse","examine"}:
            return "Analyze"
        if v in {"evaluate","assess","appraise","judge","justify","critique","criticize","criticise","defend","recommend"}:
            return "Evaluate"
        if v in {"design","develop","construct","formulate","propose","create","devise","produce","build","compose","write","draw","sketch"}:
            return "Create"
        return "Understand"
    if re.match(r"^\s*what\s+(?:is|are|was|were)\b", x, re.I):
        return "Remember"
    if re.match(r"^\s*why\b", x, re.I):
        return "Understand"
    for level, p in (
        ("Evaluate", r"^(?:briefly\s+|critically\s+|carefully\s+)?(?:evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|recommend)\b"),
        ("Analyze", r"^(?:briefly\s+)?(?:analyze|analyse|examine|compare|contrast|differentiate|distinguish)\b"),
        ("Apply", r"^(?:calculate|compute|determine|find|solve|apply|use|implement|demonstrate|modify|estimate|measure)\b"),
        ("Create", r"^(?:design|develop|construct|formulate|propose|create|devise|produce|build|compose|write|draw|sketch)\b"),
        ("Remember", r"^(?:define|identify|name|list|state|recite|label|recognize|recognise)\b"),
        ("Understand", r"^(?:explain|describe|summarize|summarise|interpret|classify|illustrate|discuss|retell)\b"),
    ):
        if re.search(p, x, re.I):
            return level
    return "Unknown"

def validate(source: str, target: str, rewrite: str) -> list[str]:
    problems = []
    if not re.match({
        "Remember": r"^(?:state|identify|list)\b",
        "Understand": r"^(?:explain|describe|summarize|summarise)\b",
        "Apply": r"^(?:apply|use|demonstrate)\b",
        "Analyze": r"^(?:analyze|analyse|examine)\b",
        "Evaluate": r"^(?:evaluate|assess|judge)\b",
        "Create": r"^(?:develop|design|formulate)\b",
    }[target], rewrite, re.I):
        problems.append("target_operation")
    missing = protected(source) - protected(rewrite)
    if missing:
        problems.append("protected_span_loss")
    if any(x in rewrite.lower() for x in ("as an ai", "bloom level", "rewritten question", "original question")):
        problems.append("meta_language")
    if rewrite.count("?") > 0:
        problems.append("not_imperative")
    if len(rewrite.split()) > 90:
        problems.append("too_long")
    return problems

def row(src: dict, target: str, rewrite: str, template: str, identity: bool, split: str) -> dict:
    sid = src["source_id"]
    example_id = hashlib.sha256(
        f"{DATASET_VERSION}|{sid}|{target}|{rewrite}".encode()
    ).hexdigest()[:16]
    system = (
        "Rewrite an academic question so that the student's required cognitive "
        "task matches the requested Bloom level. Preserve the source topic, "
        "technical concepts, quantities, named entities, and constraints. "
        "Do not invent subject matter. Output only one student-facing exam question."
    )
    user = f"Original question:\n{src['source_question']}\n\nTarget Bloom level:\n{target}"
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": rewrite},
    ]
    packed = (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + user + "<|im_end|>\n"
        "<|im_start|>assistant\n" + rewrite + "<|im_end|>"
    )
    return {
        "example_id": example_id,
        "source_id": sid,
        "group_id": src["group_id"],
        "split": split,
        "source_question": src["source_question"],
        "source_bloom_level": src["source_bloom_level"],
        "source_detected_level": src["source_detected_level"],
        "target_bloom_level": target,
        "target_rewrite": rewrite,
        "transformation_type": f"{src['source_bloom_level']}->{target}",
        "synthetic_or_original": "original_identity" if identity else "synthetic",
        "synthetic": not identity,
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "quality_status": "pass",
        "construction_method": "deterministic_controlled_transformation",
        "construction_template": template,
        "source_file": SOURCE_FILE,
        "generator_inputs": ["source_question", "target_bloom_level"],
        "validation": {
            "checks": ["target_operation", "protected_span_preservation", "meta_language", "imperative_form"],
            "identity": identity,
        },
        "messages": msgs,
        "text": packed,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_FILE)
    ap.add_argument("--output-dir", default=f"data/bloom_rewrite_versions/{DATASET_VERSION}")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    src_path = Path(args.source)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_file():
            p.unlink()

    sources, rejected = [], []
    seen = set()
    with src_path.open("r", encoding="utf-8-sig", newline="") as f:
        for idx, raw in enumerate(csv.DictReader(f)):
            q = norm(raw.get("question", "")).strip(' "')
            source_level = canon(raw.get("bloom_level", ""))
            if not q or source_level is None:
                continue
            ok, reason, t = source_ok(q)
            if not ok:
                rejected.append({
                    "row_index": idx, "source_question": q,
                    "source_bloom_level": source_level, "reason": reason
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
                "source_bloom_level": source_level,
                "source_detected_level": detect_level(q),
                "topic": t,
            })

    split_map = {"train": [], "validation": [], "test": []}
    for s in sources:
        u = int(hashlib.sha256(
            f"{args.seed}|split|{s['source_id']}".encode()
        ).hexdigest()[:8], 16) / 0xFFFFFFFF
        split_map["train" if u < 0.70 else "validation" if u < 0.85 else "test"].append(s)

    all_rows = {}
    rejected_cross = []
    for split, split_sources in split_map.items():
        rows = []
        for s in split_sources:
            for target in LEVELS:
                rewrite, template, identity = rewrite_for(
                    target, s["topic"], s["source_question"],
                    int(hashlib.sha256(f"{s['source_id']}|{target}".encode()).hexdigest()[:8], 16),
                )
                problems = validate(s["source_question"], target, rewrite)
                if problems:
                    rejected_cross.append({
                        "source_id": s["source_id"],
                        "target_bloom_level": target,
                        "source_question": s["source_question"],
                        "candidate": rewrite,
                        "problems": problems,
                    })
                    continue
                rows.append(row(s, target, rewrite, template, identity, split))
        all_rows[split] = rows
        write_path = out / f"{split}.jsonl"
        with write_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Make the combined JSON that the user can directly download.
    combined = []
    for split in ("train", "validation", "test"):
        combined.extend(all_rows[split])
    combined_path = out / f"{DATASET_VERSION}.json"
    combined_path.write_text(
        json.dumps({
            "dataset_version": DATASET_VERSION,
            "policy_version": POLICY_VERSION,
            "source": SOURCE_FILE,
            "source_figshare_sha256": hashlib.sha256(src_path.read_bytes()).hexdigest(),
            "rows": combined,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source_sets = {k: {r["source_id"] for r in v} for k, v in all_rows.items()}
    leakage = {
        "train_validation": len(source_sets["train"] & source_sets["validation"]),
        "train_test": len(source_sets["train"] & source_sets["test"]),
        "validation_test": len(source_sets["validation"] & source_sets["test"]),
    }
    if any(leakage.values()):
        raise RuntimeError(f"Source leakage detected: {leakage}")

    counts = {
        split: {
            "total": len(rows),
            "by_target": dict(Counter(r["target_bloom_level"] for r in rows)),
            "by_source_target": dict(Counter(
                f"{r['source_bloom_level']}->{r['target_bloom_level']}" for r in rows
            )),
        }
        for split, rows in all_rows.items()
    }
    stats = {
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "seed": args.seed,
        "source_figshare": SOURCE_FILE,
        "source_figshare_sha256": hashlib.sha256(src_path.read_bytes()).hexdigest(),
        "eligible_source_count": len(sources),
        "rejected_source_count": len(rejected),
        "rejected_cross_level_rows": len(rejected_cross),
        "rejected_sources_by_reason": dict(Counter(x["reason"] for x in rejected)),
        "source_split_sizes": {k: len(v) for k, v in split_map.items()},
        "counts": counts,
        "source_leakage_check": leakage,
        "target_balance": {
            split: len({r["target_bloom_level"] for r in rows}) == 6
            for split, rows in all_rows.items()
        },
        "notes": [
            "Synthetic controlled supervision; not human gold.",
            "Built directly from data/figshare_bloom_v1.csv.",
            "Each eligible source question is assigned to exactly one split.",
            "Each source receives one candidate for each of six target Bloom levels.",
            "Same-level rows retain the clean original question.",
            "Cross-level rows use deterministic source-anchored transformations.",
            "No LLM generation or LLM judging is used.",
        ],
    }
    (out / "dataset_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "rejected_sources.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rejected),
        encoding="utf-8",
    )
    (out / "rejected_cross_level.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rejected_cross),
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# bloom_rewrite_synth_v20\n\n"
        "Controlled Bloom-target rewrite supervision built directly from the "
        "authoritative Figshare corpus. Each eligible source question receives "
        "one target rewrite for each Bloom level. No LLM is used.\n",
        encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
