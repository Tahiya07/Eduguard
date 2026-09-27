#!/usr/bin/env python3
"""Build a high-fidelity Bloom-target rewrite dataset from the raw Figshare Bloom corpus.

v17 is intentionally pattern-guarded:
- source question is the authoritative content
- same-operation rows are kept as-is
- cross-level rewrites are emitted only from explicit safe patterns
- unsupported pairs are excluded
- no LLM generation or judging
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
    "knowledge": "Remember",
    "remembering": "Remember",
    "recall": "Remember",
    "comprehension": "Understand",
    "understanding": "Understand",
    "application": "Apply",
    "applying": "Apply",
    "analysis": "Analyze",
    "analysing": "Analyze",
    "analyzing": "Analyze",
    "evaluation": "Evaluate",
    "evaluating": "Evaluate",
    "synthesis": "Create",
    "creating": "Create",
}

DATASET_VERSION = "bloom_rewrite_synth_v17"
POLICY_VERSION = "bloom_target_policy_v17_explicit_safe_patterns"
SOURCE_FILE = "data/figshare_bloom_v1.csv"
SEED = 42
IDENTITY_CAP = 120

# Only operation words are excluded from lexical-content checks.
OP_WORDS = set("""
define explain describe list name state identify recall recognize recognise
summarize summarise interpret classify illustrate apply use calculate compute
determine find solve implement demonstrate analyze analyse compare contrast
differentiate distinguish examine evaluate assess appraise judge justify critique
criticize criticise defend design develop construct formulate propose create devise
produce build compose write draw sketch select choose suggest recommend predict
recite outline label specify modify estimate measure relate rank support discuss
retell
""".split())

STOP = set("""
a an the and or but if then than that this these those it its of in on at
to for from with by as is are was were be been being do does did can could
may might will would should must into over under after before during through
about against between among within without using used your our their you we
someone somebody one
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
    "the article", "this magazine article", "the figure above", "the table above"
)

# ---------- source parsing ----------

COMPARE_BETWEEN_RE = re.compile(
    r"^\s*(?:differentiate|distinguish)\s+between\s+(.+?)\s*\.?$", re.I
)
COMPARE_DIRECT_RE = re.compile(
    r"^\s*(?:compare|contrast)\s+(.+?)\s*\.?$", re.I
)
HOW_DIFFER_RE = re.compile(
    r"^\s*how\s+(?:does|do|did)\s+(.+?)\s+differ\s+from\s+(.+?)\s*\.?$", re.I
)
WHAT_RE = re.compile(r"^\s*what\s+(?:is|are|was|were)\s+(.+?)\s*\.?$", re.I)
HOW_AUX_RE = re.compile(
    r"^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+(.+?)\s*\.?$", re.I
)

LEADING_COMMANDS = [
    ("Evaluate", re.compile(
        r"^\s*(?:briefly\s+|critically\s+|carefully\s+)?"
        r"(?:evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|recommend)\b",
        re.I
    )),
    ("Analyze", re.compile(
        r"^\s*(?:briefly\s+)?(?:analyze|analyse|examine|compare|contrast|differentiate|distinguish)\b",
        re.I
    )),
    ("Apply", re.compile(
        r"^\s*(?:calculate|compute|determine|find|solve|apply|use|implement|demonstrate|modify|estimate|measure)\b",
        re.I
    )),
    ("Create", re.compile(
        r"^\s*(?:design|develop|construct|formulate|propose|create|devise|produce|build|compose|write|draw|sketch)\b",
        re.I
    )),
    ("Remember", re.compile(
        r"^\s*(?:define|identify|name|list|state|recite|label|recognize|recognise)\b",
        re.I
    )),
    ("Understand", re.compile(
        r"^\s*(?:explain|describe|summarize|summarise|interpret|classify|illustrate|discuss|retell)\b",
        re.I
    )),
]

# ---------- narrow cross-level patterns ----------

# Remember <- Understand
R_FROM_EXPLAIN = re.compile(
    r"^\s*(?:explain|describe|discuss|illustrate|interpret)\s+(.+?)\s*\.?$",
    re.I
)
# Remember <- comparison
R_FROM_COMPARE = re.compile(
    r"^\s*(?:compare|contrast)\s+(.+?)\s*\.?$",
    re.I
)
R_FROM_DIFF = re.compile(
    r"^\s*(?:differentiate|distinguish)\s+between\s+(.+?)\s*\.?$",
    re.I
)

# Understand <- Remember
U_FROM_DEFINE = re.compile(
    r"^\s*(?:define|identify|name|state|recite|label|recognize|recognise)\s+(.+?)\s*\.?$",
    re.I
)
U_FROM_COMPARE = re.compile(
    r"^\s*compare\s+(.+?)\s*\.?$", re.I
)
U_FROM_CONTRAST = re.compile(
    r"^\s*contrast\s+(.+?)\s*\.?$", re.I
)
U_FROM_DIFF = re.compile(
    r"^\s*(?:differentiate|distinguish)\s+between\s+(.+?)\s*\.?$", re.I
)

# Apply <- Understand: only explicit method/procedure + application verb.
A_FROM_METHOD_USE = re.compile(
    r"^\s*(?:explain|describe|discuss)\s+(.+?\b(?:method|procedure|algorithm|formula|equation|technique)\b.+?)"
    r"\s+(?:is|are|can be|could be|may be)\s+(?:used|applied|implemented)\s+to\s+(.+?)\s*\.?$",
    re.I
)
A_FROM_HOW_METHOD = re.compile(
    r"^\s*how\s+(?:can|could|would|should|do|does)\s+(.+?\b(?:method|procedure|algorithm|formula|equation|technique)\b.+?)"
    r"\s+(?:be\s+)?(?:used|applied|implemented)\s+to\s+(.+?)\s*\.?$",
    re.I
)

# Analyze <- Understand: exact relation/content constructs.
N_FROM_RELATION = re.compile(
    r"^\s*(?:explain|describe|discuss)\s+(?:the\s+)?(relationship|relationships|interaction|interactions)"
    r"\s+between\s+(.+?)\s*\.?$", re.I
)
N_FROM_DIFF = re.compile(
    r"^\s*(?:explain|describe|discuss)\s+(?:the\s+)?differences?\s+between\s+(.+?)\s*\.?$", re.I
)
N_FROM_COMPARE = re.compile(
    r"^\s*(?:compare|contrast)\s+(.+?)\s*\.?$", re.I
)
N_FROM_EXAMINE = re.compile(
    r"^\s*(?:explain|describe|discuss)\s+(?:the\s+)?(components?|parts?|structure|patterns?|causes?|effects?)"
    r"\s+(?:of|in)\s+(.+?)\s*\.?$", re.I
)
N_FROM_WHY = re.compile(
    r"^\s*explain\s+why\s+(.+?)\s*\.?$", re.I
)
N_FROM_HOW = re.compile(
    r"^\s*explain\s+how\s+(.+?)\s*\.?$", re.I
)

# Evaluate <- Understand/Remember: source must already contain an explicit
# evaluative dimension such as advantages, effectiveness, suitability, best.
E_FROM_EXPLICIT = re.compile(
    r"^\s*(?:explain|describe|discuss|identify|list|state|illustrate)\s+(.+?)\s*\.?$",
    re.I
)

# Create <- existing source with an explicit artifact noun; only the frame changes.
CREATE_DIAGRAM = re.compile(
    r"^\s*(?:draw|sketch|illustrate)\s+(.+?(?:diagram|drawing|sketch|flow\s*chart|flowchart).*)\.?$",
    re.I
)
CREATE_PROGRAM = re.compile(
    r"^\s*(?:write|construct|develop|create|design|build)\s+(.+?(?:program|algorithm|source\s+code|code\b).*)\.?$",
    re.I
)
CREATE_PRESENTATION = re.compile(
    r"^\s*(?:develop|create|design|produce|propose)\s+(.+?(?:presentation|storyboard).*)\.?$",
    re.I
)
CREATE_STRATEGY = re.compile(
    r"^\s*(?:develop|construct|create|propose|design)\s+(.+?(?:strategy|plan|proposal).*)\.?$",
    re.I
)
CREATE_MODEL = re.compile(
    r"^\s*(?:develop|construct|create|design|propose)\s+(.+?(?:model|framework|prototype).*)\.?$",
    re.I
)
CREATE_HYPOTHESIS = re.compile(
    r"^\s*(?:develop|construct|formulate|propose|create)\s+(?:a\s+)?(hypothesis.+?)\.?$",
    re.I
)

def norm(x):
    return re.sub(r"\s+", " ", (x or "").replace("\u00a0", " ")).strip()

def canon(x):
    x = str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())

def content(text):
    return {
        t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*", (text or "").lower())
        if len(t) > 2 and t not in STOP and t not in OP_WORDS
    }

def protected(text):
    out = {m.group(0).lower() for m in NUM_RE.finditer(text or "")}
    out.update(m.group(0).lower() for m in TECH_RE.finditer(text or ""))
    out.update(m.group(0).lower() for m in FILE_RE.finditer(text or ""))
    out.update(m.group(1).lower() for m in ACRONYM_RE.finditer(text or ""))
    return out

def detected_level(q):
    for level, pat in LEADING_COMMANDS:
        if pat.search(q):
            return level
    if WHAT_RE.match(q):
        return "Remember"
    if HOW_AUX_RE.match(q):
        return "Understand"
    return "Unknown"

def extract_topic(q):
    q = norm(q).strip(' "').rstrip("?").rstrip(".").strip()

    m = HOW_DIFFER_RE.match(q)
    if m:
        return f"the difference between {m.group(1).strip()} and {m.group(2).strip()}"

    m = COMPARE_BETWEEN_RE.match(q)
    if m:
        return f"the differences between {m.group(1).strip()}"

    m = COMPARE_DIRECT_RE.match(q)
    if m:
        return f"the similarities and differences between {m.group(1).strip()}"

    m = WHAT_RE.match(q)
    if m:
        return m.group(1).strip()

    m = HOW_AUX_RE.match(q)
    if m:
        return m.group(1).strip()

    for _, pat in LEADING_COMMANDS:
        m = pat.match(q)
        if m:
            return q[m.end():].strip(" .;:,")

    return None

def source_ok(q):
    q = norm(q).strip(' "')
    if not (30 <= len(q) <= 900):
        return False, "length"
    if PLACEHOLDER_RE.search(q):
        return False, "placeholder"
    if any(x in q.lower() for x in MISSING_CONTEXT):
        return False, "missing_context"
    if q.count("?") > 1:
        return False, "multiple_questions"
    # Reject obvious multi-item compound tasks.
    if re.search(
        r"\b(?:and|or)\s+(?:give|provide|justify|explain|describe|discuss|"
        r"calculate|compute|determine|find|solve|draw|sketch|write|create|"
        r"design|develop|construct|propose|list|name|state)\b", q, re.I
    ):
        return False, "multiple_actions"
    if len(content(q)) < 5:
        return False, "thin_content"
    topic = extract_topic(q)
    if not topic or len(content(topic)) < 3:
        return False, "topic_parse"
    return True, ""

def cross_transform(source, src_level, topic, target):
    # -------- Remember --------
    if target == "Remember" and src_level != "Remember":
        if src_level in {"Understand","Evaluate"}:
            m = R_FROM_DIFF.match(source)
            if m:
                return f"State the differences between {m.group(1).strip()}.", "remember_difference"
            m = R_FROM_COMPARE.match(source)
            if m:
                return f"State the similarities and differences in {m.group(1).strip()}.", "remember_comparison"
            m = R_FROM_EXPLAIN.match(source)
            if m and not re.match(r"(?i)^(why|how)\b", m.group(1).strip()):
                return f"State {m.group(1).strip()}.", "remember_state"
        return None, None

    # -------- Understand --------
    if target == "Understand" and src_level != "Understand":
        if src_level == "Remember":
            m = U_FROM_DEFINE.match(source)
            if m:
                return f"Explain {m.group(1).strip()}.", "understand_explain"
        if src_level == "Analyze":
            m = U_FROM_DIFF.match(source)
            if m:
                return f"Explain the differences between {m.group(1).strip()}.", "understand_difference"
            m = U_FROM_COMPARE.match(source)
            if m:
                return f"Explain the similarities and differences between {m.group(1).strip()}.", "understand_comparison"
            m = HOW_DIFFER_RE.match(source)
            if m:
                return f"Explain the difference between {m.group(1).strip()} and {m.group(2).strip()}.", "understand_difference"
        if src_level == "Evaluate":
            # Only convert questions that explicitly contain an evaluative object
            # rather than a hidden recommendation/decision scenario.
            topic_low = topic.lower()
            if any(x in topic_low for x in (
                "advantages", "disadvantages", "effectiveness", "suitability",
                "appropriateness", "strengths", "limitations", "differences"
            )):
                m = E_FROM_EXPLICIT.match(source)
                if m:
                    return f"Explain {m.group(1).strip()}.", "understand_evaluative_topic"
        return None, None

    # -------- Apply --------
    if target == "Apply" and src_level != "Apply":
        m = A_FROM_METHOD_USE.match(source)
        if m:
            return f"Apply {m.group(1).strip()} to {m.group(2).strip()}.", "apply_method"
        m = A_FROM_HOW_METHOD.match(source)
        if m:
            return f"Apply {m.group(1).strip()} to {m.group(2).strip()}.", "apply_method"
        return None, None

    # -------- Analyze --------
    if target == "Analyze" and src_level != "Analyze":
        m = N_FROM_RELATION.match(source)
        if m:
            return f"Analyze the {m.group(1).lower()} between {m.group(2).strip()}.", "analyze_relationship"
        m = N_FROM_DIFF.match(source)
        if m:
            return f"Analyze the differences between {m.group(1).strip()}.", "analyze_difference"
        m = N_FROM_COMPARE.match(source)
        if m:
            return f"Analyze the similarities and differences between {m.group(1).strip()}.", "analyze_comparison"
        m = N_FROM_EXAMINE.match(source)
        if m:
            return f"Analyze the {m.group(1).lower()} of {m.group(2).strip()}.", "analyze_structure"
        m = N_FROM_WHY.match(source)
        if m:
            return f"Analyze why {m.group(1).strip()}.", "analyze_cause"
        m = N_FROM_HOW.match(source)
        if m:
            return f"Analyze how {m.group(1).strip()}.", "analyze_process"
        return None, None

    # -------- Evaluate --------
    if target == "Evaluate" and src_level != "Evaluate":
        low = (source + " " + topic).lower()
        if any(x in low for x in (
            "advantages and disadvantages", "advantages", "disadvantages",
            "effectiveness", "suitability", "appropriate", "best", "better",
            "worse", "recommend", "recommendation", "choice", "choices",
            "strengths", "limitations", "alternative", "alternatives",
            "agree", "opinion"
        )):
            m = E_FROM_EXPLICIT.match(source)
            if m:
                return f"Evaluate {m.group(1).strip()} and justify your judgment.", "evaluate_explicit_dimension"
        return None, None

    # -------- Create --------
    if target == "Create" and src_level != "Create":
        if CREATE_DIAGRAM.match(source):
            m = CREATE_DIAGRAM.match(source)
            body = m.group(1).strip()
            return f"Design a diagram illustrating {body}.", "create_diagram"
        if CREATE_PROGRAM.match(source):
            m = CREATE_PROGRAM.match(source)
            body = m.group(1).strip()
            return f"Develop {body}.", "create_program"
        if CREATE_PRESENTATION.match(source):
            m = CREATE_PRESENTATION.match(source)
            body = m.group(1).strip()
            return f"Develop {body}.", "create_presentation"
        if CREATE_STRATEGY.match(source):
            m = CREATE_STRATEGY.match(source)
            body = m.group(1).strip()
            return f"Develop {body}.", "create_strategy"
        if CREATE_MODEL.match(source):
            m = CREATE_MODEL.match(source)
            body = m.group(1).strip()
            return f"Construct {body}.", "create_model"
        if CREATE_HYPOTHESIS.match(source):
            m = CREATE_HYPOTHESIS.match(source)
            body = m.group(1).strip()
            return f"Formulate a {body}.", "create_hypothesis"
        return None, None

    return None, None

def validate(source, target, rewrite):
    if not rewrite:
        return False, ["empty"], 0.0

    starts = {
        "Remember": r"^(state|identify|list)\b",
        "Understand": r"^explain\b",
        "Apply": r"^apply\b",
        "Analyze": r"^analyze\b",
        "Evaluate": r"^evaluate\b",
        "Create": r"^(design|develop|construct|formulate|create)\b",
    }
    if not re.search(starts[target], rewrite, re.I):
        # Same-level source rows are separately accepted.
        return False, ["target_operation_missing"], 0.0

    if protected(source) - protected(rewrite):
        return False, ["protected_span_loss"], 0.0

    return True, [], 1.0

def make_row(split,target,s,rewrite,template,validation):
    system=(
        "Rewrite an academic question to the requested Bloom level. Preserve "
        "the source content, technical concepts, quantities, named entities, "
        "and constraints. Do not invent subject matter. Output only one "
        "student-facing exam question."
    )
    user=f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"
    messages=[
        {"role":"system","content":system},
        {"role":"user","content":user},
        {"role":"assistant","content":rewrite},
    ]
    packed=(
        "<|im_start|>system\n"+system+"<|im_end|>\n"
        "<|im_start|>user\n"+user+"<|im_end|>\n"
        "<|im_start|>assistant\n"+rewrite+"<|im_end|>"
    )
    synthetic=rewrite!=s["source_question"]
    return {
      "example_id":hashlib.sha256(
        f"{DATASET_VERSION}|{s['source_id']}|{target}|{rewrite}".encode()
      ).hexdigest()[:16],
      "source_id":s["source_id"],
      "group_id":s["group_id"],
      "split":split,
      "source_question":s["source_question"],
      "source_bloom_level":s["source_bloom_level"],
      "source_detected_level":s["source_detected_level"],
      "source_task_form":s["source_task_form"],
      "target_bloom_level":target,
      "target_rewrite":rewrite,
      "transformation_type":f"{s['source_bloom_level']}->{target}",
      "synthetic_or_original":"synthetic" if synthetic else "original_identity",
      "synthetic":synthetic,
      "dataset_version":DATASET_VERSION,
      "policy_version":POLICY_VERSION,
      "quality_status":"pass",
      "construction_method":"deterministic_explicit_safe_patterns",
      "construction_template":template,
      "source_file":SOURCE_FILE,
      "generator_inputs":["source_question","target_bloom_level"],
      "validation":validation,
      "messages":messages,
      "text":packed
    }

def write_jsonl(path,rows):
    with path.open("w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default=SOURCE_FILE)
    ap.add_argument("--output-dir",default=f"data/bloom_rewrite_versions/{DATASET_VERSION}")
    ap.add_argument("--seed",type=int,default=SEED)
    ap.add_argument("--identity-cap",type=int,default=IDENTITY_CAP)
    args=ap.parse_args()

    src=Path(args.source); out=Path(args.output_dir)
    if not src.exists(): raise SystemExit(f"Source not found: {src}")
    out.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if p.is_file(): p.unlink()

    sources=[]; rejected=[]; seen=set()
    with src.open("r",encoding="utf-8-sig",newline="") as f:
        for idx,raw in enumerate(csv.DictReader(f)):
            q=norm(raw.get("question","")).strip(' "')
            level=canon(raw.get("bloom_level",""))
            if not q or not level: continue
            ok,why=source_ok(q)
            if not ok:
                rejected.append({"row_index":idx,"source_question":q,"source_bloom_level":level,"reason":why})
                continue
            topic=extract_topic(q)
            sid="src_"+hashlib.sha256(q.encode()).hexdigest()[:16]
            if sid in seen: continue
            seen.add(sid)
            sources.append({
                "source_id":sid,
                "group_id":int(hashlib.sha256(sid.encode()).hexdigest()[:8],16),
                "source_question":q,
                "source_bloom_level":level,
                "source_detected_level":detected_level(q),
                "source_task_form":detected_level(q),
                "topic":topic,
            })

    splits={"train":[],"validation":[],"test":[]}
    for s in sources:
        u=int(hashlib.sha256(f"{args.seed}|split|{s['source_id']}".encode()).hexdigest()[:8],16)/0xFFFFFFFF
        splits["train" if u<.70 else "validation" if u<.85 else "test"].append(s)

    all_rows={}; summary={}
    for split in ("train","validation","test"):
        by_target={t:[] for t in LEVELS}

        for target in LEVELS:
            identity=[s for s in splits[split] if s["source_detected_level"]==target]
            rng=random.Random(args.seed+sum(map(ord,target))+len(split))
            rng.shuffle(identity)
            for s in identity[:args.identity_cap]:
                by_target[target].append(
                    make_row(
                        split,target,s,s["source_question"],
                        "identity_source_question",
                        {"reasons":[],"topic_content_recall":1.0,
                         "protected_spans":sorted(protected(s["source_question"])),
                         "missing_protected":[],"identity":True}
                    )
                )

            for s in splits[split]:
                if s["source_detected_level"]==target:
                    continue
                rewrite,template=cross_transform(
                    s["source_question"],s["source_detected_level"],s["topic"],target
                )
                if not rewrite:
                    continue
                ok,reasons,recall=validate(
                    s["source_question"],target,rewrite
                )
                if not ok:
                    continue
                by_target[target].append(
                    make_row(
                        split,target,s,rewrite,template,
                        {"reasons":[],"topic_content_recall":round(recall,4),
                         "protected_spans":sorted(protected(s["source_question"])),
                         "missing_protected":[],"identity":False}
                    )
                )

        rows=[r for t in LEVELS for r in by_target[t]]
        all_rows[split]=rows
        write_jsonl(out/f"{split}.jsonl",rows)
        summary[split]={
            "total":len(rows),
            "by_target":{t:len(by_target[t]) for t in LEVELS},
            "cross_level_rows":sum(r["source_detected_level"]!=r["target_bloom_level"] for r in rows),
            "identity_rows":sum(r["source_detected_level"]==r["target_bloom_level"] for r in rows),
            "by_source_target":dict(Counter(
                f"{r['source_detected_level']}->{r['target_bloom_level']}" for r in rows
            ))
        }

    source_sets={s:{r["source_id"] for r in rows} for s,rows in all_rows.items()}
    leakage={
      "train_validation":len(source_sets["train"]&source_sets["validation"]),
      "train_test":len(source_sets["train"]&source_sets["test"]),
      "validation_test":len(source_sets["validation"]&source_sets["test"])
    }
    if any(leakage.values()): raise RuntimeError(f"Source leakage: {leakage}")

    sh=hashlib.sha256(src.read_bytes()).hexdigest()
    stats={
      "dataset_version":DATASET_VERSION,
      "policy_version":POLICY_VERSION,
      "seed":args.seed,
      "source_figshare":SOURCE_FILE,
      "source_figshare_sha256":sh,
      "eligible_source_count":len(sources),
      "rejected_source_count":len(rejected),
      "rejected_sources_by_reason":dict(Counter(r["reason"] for r in rejected)),
      "source_split_sizes":{s:len(v) for s,v in splits.items()},
      "counts":summary,
      "identity_cap":args.identity_cap,
      "source_leakage_check":leakage,
      "notes":[
        "Synthetic supervision; not human gold.",
        "Built directly from data/figshare_bloom_v1.csv.",
        "Actual source task form is detected from wording and stored separately from the source label.",
        "Same-operation rows retain the original source question.",
        "Cross-level rows are produced only by explicit safe whole-question patterns.",
        "Unsupported transformations are excluded rather than invented.",
        "No LLM generation or LLM judging is used.",
      ]
    }
    (out/"dataset_statistics.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"dataset_manifest.json").write_text(json.dumps({
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":sh,
      "counts":{s:len(v) for s,v in all_rows.items()},
      "source_leakage_check":leakage
    },indent=2,ensure_ascii=False),encoding="utf-8")
    write_jsonl(out/"rejected_sources.jsonl",rejected)
    (out/"README.md").write_text(
      "# bloom_rewrite_synth_v17\n\n"
      "High-fidelity deterministic Bloom-target supervision built directly from "
      "the authoritative Figshare corpus. Same-operation source questions are "
      "retained. Cross-level transformations use only explicit safe patterns; "
      "unsupported transformations are excluded. No LLM is used.\\n",
      encoding="utf-8")
    print("ELIGIBLE SOURCES:",len(sources))
    print("SOURCE SPLITS:",{s:len(v) for s,v in splits.items()})
    print("COUNTS:",json.dumps(summary,indent=2))
    print("REJECTED:",stats["rejected_sources_by_reason"])
    print("LEAKAGE:",leakage)
    print("OUTPUT:",out)

if __name__=="__main__":
    main()
