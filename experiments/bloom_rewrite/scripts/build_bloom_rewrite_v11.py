#!/usr/bin/env python3
"""Build high-fidelity Bloom rewrite supervision from the raw Figshare corpus.

v11 principle:
- Preserve the source wording/content.
- Change only the cognitive task when the target operation is supported.
- Never invent subject matter, criteria, examples, numbers, contexts, or artifacts.
- Reject unsuitable source/target combinations instead of forcing them.
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
DATASET_VERSION = "bloom_rewrite_synth_v11"
POLICY_VERSION = "bloom_target_policy_v11_minimal_source_preserving"
SOURCE_FILE = "data/figshare_bloom_v1.csv"
SEED = 42

COGNITIVE = set("""
define explain describe list name state identify recall recognize recognise
summarize summarise interpret classify illustrate apply use calculate compute
determine find solve implement demonstrate analyze analyse compare contrast
differentiate distinguish examine evaluate assess critique criticize criticise
judge justify defend design develop construct formulate propose create devise
produce generate write build show discuss select choose suggest recommend predict
recite outline label draw sketch appraise specify modify estimate measure relate
rank
""".split())

STOP = set("""
a an the and or but if then than that this these those it its of in on at to
for from with by as is are was were be been being do does did can could may
might will would should must into over under after before during through about
against between among within without using used your our their you we someone
somebody one
""".split())

# Content tokens are used only as a consistency check. The source remainder is
# otherwise copied verbatim, so there is no model-generated topic drift.
def content(text):
    return {
        t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*", (text or "").lower())
        if len(t) > 2 and t not in STOP and t not in COGNITIVE
    }

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

LEADING = re.compile(
    r"""^\s*
    (?:
      briefly\s+|critically\s+|carefully\s+|concisely\s+
    )*
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

HOW_DIFFER = re.compile(
    r"^\s*how\s+(?:does|do|did)\s+(.+?)\s+differ\s+from\s+(.+?)\s*$", re.I
)

COMPARE_BETWEEN = re.compile(
    r"^\s*(?:differentiate|distinguish)\s+between\s+(.+)$", re.I
)

COMPARE_SIMPLE = re.compile(r"^\s*(?:compare|contrast)\s+(.+)$", re.I)

SECOND_ACTION = re.compile(
    r"\b(?:and|then|also|followed\s+by)\s+(?:" +
    "|".join(sorted(COGNITIVE)) +
    r")\b", re.I
)

TAIL_FORMAT = re.compile(
    r"""(?:\s+show\s+your\s+working(?:\s+and\s+calculation)?|
    \s+justify\s+your\s+answer|
    \s+support\s+(?:your\s+answer|your\s+views?)|
    \s+elaborate(?:\s+your\s+answer)?|
    \s+provide\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)
      (?:\s+\w+){0,4}\s+examples?|
    \s+give\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)
      (?:\s+\w+){0,4}\s+examples?)\s*\.?$""",
    re.I | re.X
)

APPLY_WORDS = (
    "calculate","compute","determine","find","solve","apply","use","implement",
    "demonstrate","algorithm","procedure","procedures","method","methods",
    "formula","equation","modify","scenario","case","problem","data",
)
ANALYZE_WORDS = (
    "between","among","relationship","relationships","interaction","interactions",
    "compare","contrast","differentiate","distinguish","component","components",
    "parts","structure","patterns","causes","effects","difference","differences",
    "similarity","similarities",
)
EVAL_WORDS = (
    "advantage","advantages","disadvantage","disadvantages","effectiveness",
    "appropriate","suitable","suitability","best","better","worse","risk","risks",
    "alternative","alternatives","choice","choices","recommend","recommendation",
    "opinion","agree","justify","strength","strengths","limitation","limitations",
    "quality","validity","prefer","preference",
)
CREATE_WORDS = (
    "design","develop","construct","formulate","propose","create","devise",
    "produce","build","write","draw","sketch","strategy","plan","solution","system",
    "model","framework","program","algorithm","presentation","diagram","hypothesis",
    "project","campaign","poster","prototype","story","storyboard","play","letter",
)
CREATE_ARTIFACT_PATTERNS = (
    r"\b(?:a|an|the)\s+(?:c\s+program|program|algorithm|diagram|drawing|sketch|"
    r"presentation|storyboard|strategy|plan|solution|model|framework|system|"
    r"hypothesis|project|campaign|poster|prototype|story|play|letter)\b",
)


def norm(x):
    return re.sub(r"\s+", " ", (x or "").replace("\u00a0", " ")).strip()


def canon(x):
    x = str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())


def protected(x):
    out = {m.group(0).lower() for m in NUM_RE.finditer(x or "")}
    out.update(m.group(0).lower() for m in TECH_RE.finditer(x or ""))
    out.update(m.group(0).lower() for m in FILE_RE.finditer(x or ""))
    out.update(m.group(1).lower() for m in ACRONYM_RE.finditer(x or ""))
    return out


def source_parts(q):
    q = norm(q).strip(' "').rstrip("?.").strip()

    m = HOW_DIFFER.match(q)
    if m:
        return f"{m.group(1)} differs from {m.group(2)}", "how_differ", "how"

    m = HOW.match(q)
    if m:
        return q[m.end():].strip(" .;:"), m.group("verb").lower(), "how"

    m = COMPARE_BETWEEN.match(q)
    if m:
        return f"the differences between {m.group(1)}", "differentiate", "imperative"

    m = COMPARE_SIMPLE.match(q)
    if m:
        return f"the similarities and differences between {m.group(1)}", m.group(0).split()[0].lower(), "imperative"

    m = LEADING.match(q)
    if m:
        rest = q[m.end():].strip(" .;:,")
        rest = TAIL_FORMAT.sub("", rest).strip(" .;:,")
        if SECOND_ACTION.search(rest):
            return None, "multiple_actions", "imperative"
        if not rest:
            return None, "thin_topic", "imperative"
        return rest, m.group("verb").lower(), "imperative"

    m = re.match(r"^\s*what\s+(?:is|are|was|were)\s+(.+)$", q, re.I)
    if m:
        return m.group(1).strip(" .;:"), "what", "what"

    return None, "unrecognized_form", ""


def source_eligible(q):
    q = norm(q).strip(' "')
    if not (35 <= len(q) <= 900):
        return False, "length"
    if PLACEHOLDER_RE.search(q):
        return False, "placeholder"
    if any(x in q.lower() for x in MISSING_CONTEXT):
        return False, "missing_context"
    # Permit ordinary abbreviations and decimals; reject clear multiple items.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", q)
    if len(parts) > 1:
        return False, "multiple_sentences"
    if q.count("?") > 1:
        return False, "multiple_questions"
    if len(content(q)) < 6:
        return False, "thin_content"
    topic, form, style = source_parts(q)
    if not topic or len(content(topic)) < 4:
        return False, form or "thin_topic"
    if any(x in topic.lower() for x in MISSING_CONTEXT):
        return False, "missing_context"
    return True, ""


def capability(source, topic, form):
    low = (source + " " + topic).lower()
    return {
        "remember": form not in {
            "calculate","compute","find","solve","apply","use","implement",
            "demonstrate","design","develop","construct","formulate","propose",
            "create","devise","produce","build","write","draw","sketch","modify",
            "how_calculate","how_compute","how_find","how_solve"
        },
        "apply": form in {
            "calculate","compute","find","solve","apply","use","implement",
            "demonstrate","modify"
        } or any(x in low for x in APPLY_WORDS),
        "analyze": (
            form in {"compare","contrast","differentiate","distinguish","analyze","analyse","examine","how_differ"}
            or any(x in low for x in ANALYZE_WORDS)
        ),
        "evaluate": (
            form in {"evaluate","assess","appraise","judge","justify","critique","defend"}
            or any(x in low for x in EVAL_WORDS)
        ),
        "create": (
            form in {"design","develop","construct","formulate","propose","create","devise",
                     "produce","build","write","draw","sketch"}
            or any(re.search(p, topic, re.I) for p in CREATE_ARTIFACT_PATTERNS)
            or any(x in low for x in ("strategy","plan","solution","model","framework",
                                      "system","program","algorithm","presentation",
                                      "diagram","hypothesis","project","campaign","poster",
                                      "prototype","storyboard"))
        ),
        "relation": any(x in low for x in ANALYZE_WORDS),
    }


def transform(source, topic, form, target):
    cap = capability(source, topic, form)

    if target == "Remember":
        if re.search(r"(?i)\bthe differences between\b", topic):
            return f"State the differences between {topic.split('between',1)[1].strip()}.", "remember_differences"
        if cap["relation"]:
            return f"State the key similarities and differences in {topic}.", "remember_relation"
        return f"State {topic}.", "remember_state"

    if target == "Understand":
        return f"Explain {topic}.", "understand_explain"

    if target == "Apply":
        if any(x in (source + " " + topic).lower() for x in ("calculate","compute","determine","find","solve")):
            return f"Apply the relevant procedure to {topic}.", "apply_procedure"
        return f"Apply {topic} in a practical situation.", "apply_context"

    if target == "Analyze":
        return f"Analyze {topic}.", "analyze_direct"

    if target == "Evaluate":
        return f"Evaluate {topic} and justify your judgment.", "evaluate_judgment"

    if target == "Create":
        low=(source+" "+topic).lower()
        if re.search(r"\b(?:diagram|drawing|sketch|flow chart|flowchart)\b",low):
            return f"Design a diagram representing {topic}.", "create_diagram"
        if re.search(r"\b(?:program|algorithm|code)\b",low) and re.search(
            r"\b(?:program|algorithm|source code|code snippet|nested loops)\b",low
        ):
            return f"Develop a program or algorithm related to {topic}.", "create_computational"
        if re.search(r"\b(?:presentation|storyboard)\b",low):
            return f"Develop a presentation about {topic}.", "create_presentation"
        if re.search(r"\b(?:strategy|plan|solution|model|framework|system)\b",low):
            return f"Develop a strategy, plan, model, framework, or system for {topic}.", "create_artifact"
        if re.search(r"\b(?:story|novel|poem|play|letter)\b",low):
            return f"Create a response related to {topic}.", "create_creative"
        return f"Develop a solution for {topic}.", "create_solution"

    raise ValueError(target)


def validate(source, topic, target, rewrite):
    reasons = []
    if protected(source) - protected(rewrite):
        reasons.append("PROTECTED_SPAN_LOSS")
    if target != "Evaluate" and SECOND_ACTION.search(rewrite):
        reasons.append("MULTIPLE_ACTIONS")
    starts = {
        "Remember": r"^(state|identify|list)\b",
        "Understand": r"^explain\b",
        "Apply": r"^apply\b",
        "Analyze": r"^analyze\b",
        "Evaluate": r"^evaluate\b",
        "Create": r"^(develop|design|construct|formulate|create)\b",
    }
    if not re.search(starts[target], rewrite, re.I):
        reasons.append("TARGET_OPERATION_MISSING")
    return not reasons, {
        "reasons": reasons,
        "topic_content_recall": round(
            len(content(topic) & content(rewrite)) / max(1, len(content(topic))), 4
        ),
        "protected_spans": sorted(protected(source)),
        "missing_protected": sorted(protected(source) - protected(rewrite)),
    }


def make_row(split, target, s, rewrite, template, info):
    system = (
        "Rewrite an academic question to the requested Bloom level. Preserve the "
        "source topic, technical concepts, quantities, named entities, and constraints. "
        "Do not invent subject matter. Do not answer the question. Output only one "
        "student-facing exam question."
    )
    user = f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"
    messages = [
        {"role":"system","content":system},
        {"role":"user","content":user},
        {"role":"assistant","content":rewrite},
    ]
    packed = (
        "<|im_start|>system\n"+system+"<|im_end|>\n"
        "<|im_start|>user\n"+user+"<|im_end|>\n"
        "<|im_start|>assistant\n"+rewrite+"<|im_end|>"
    )
    synthetic = rewrite != s["source_question"]
    return {
        "example_id": hashlib.sha256(
            f"{DATASET_VERSION}|{s['source_id']}|{target}|{rewrite}".encode()
        ).hexdigest()[:16],
        "source_id": s["source_id"],
        "group_id": s["group_id"],
        "split": split,
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
        "construction_method": "deterministic_minimal_source_preserving",
        "construction_template": template,
        "source_file": SOURCE_FILE,
        "generator_inputs": ["source_question","target_bloom_level"],
        "validation": info,
        "messages": messages,
        "text": packed,
    }


def write_jsonl(path, rows):
    with path.open("w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default=SOURCE_FILE)
    ap.add_argument("--output-dir",default=f"data/bloom_rewrite_versions/{DATASET_VERSION}")
    ap.add_argument("--seed",type=int,default=SEED)
    ap.add_argument("--overwrite",action="store_true")
    args=ap.parse_args()

    src=Path(args.source); out=Path(args.output_dir)
    if not src.exists(): raise SystemExit(f"Source not found: {src}")
    if out.exists() and not args.overwrite: raise SystemExit(f"Output exists: {out}. Use --overwrite.")
    out.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if p.is_file(): p.unlink()

    sources,rejected= [],[]
    seen=set()
    with src.open("r",encoding="utf-8-sig",newline="") as f:
        for idx,raw in enumerate(csv.DictReader(f)):
            q=norm(raw.get("question","")).strip(' "')
            level=canon(raw.get("bloom_level",""))
            if not q or not level: continue
            ok,why=source_eligible(q)
            if not ok:
                rejected.append({"row_index":idx,"source_question":q,
                                 "source_bloom_level":level,"reason":why})
                continue
            topic,form,_=source_parts(q)
            sid="src_"+hashlib.sha256(q.encode()).hexdigest()[:16]
            if sid in seen: continue
            seen.add(sid)
            sources.append({
                "source_id":sid,
                "group_id":int(hashlib.sha256(sid.encode()).hexdigest()[:8],16),
                "source_question":q,
                "source_bloom_level":level,
                "topic":topic,
                "form":form,
            })

    splits={"train":[],"validation":[],"test":[]}
    for s in sources:
        u=int(hashlib.sha256(f"{args.seed}|split|{s['source_id']}".encode()).hexdigest()[:8],16)/0xFFFFFFFF
        splits["train" if u<.70 else "validation" if u<.85 else "test"].append(s)

    all_rows={}; stats={}
    for split in ("train","validation","test"):
        rows=[]; per={}
        for target in LEVELS:
            target_rows=[]
            for s in splits[split]:
                if s["source_bloom_level"]==target:
                    rewrite=s["source_question"]
                    info={"reasons":[],"topic_content_recall":1.0,
                          "protected_spans":sorted(protected(rewrite)),"missing_protected":[]}
                    target_rows.append(make_row(split,target,s,rewrite,"identity_source_question",info))
                    continue

                if not target_supported(s["source_question"],s["topic"],s["form"],target):
                    continue

                rewrite,template=transform(s["source_question"],s["topic"],s["form"],target)
                ok,info=validate(s["source_question"],s["topic"],target,rewrite)
                if ok:
                    info["cross_level"]=True
                    target_rows.append(make_row(split,target,s,rewrite,template,info))
            rows.extend(target_rows)
            per[target]={
                "total":len(target_rows),
                "cross_level":sum(r["source_bloom_level"]!=r["target_bloom_level"] for r in target_rows),
                "identity":sum(r["source_bloom_level"]==r["target_bloom_level"] for r in target_rows),
            }
        rows.sort(key=lambda r:(LEVELS.index(r["target_bloom_level"]),r["example_id"]))
        all_rows[split]=rows
        stats[split]=per
        write_jsonl(out/f"{split}.jsonl",rows)

    sets={s:{r["source_id"] for r in rows} for s,rows in all_rows.items()}
    leakage={
      "train_validation":len(sets["train"]&sets["validation"]),
      "train_test":len(sets["train"]&sets["test"]),
      "validation_test":len(sets["validation"]&sets["test"])
    }
    if any(leakage.values()): raise RuntimeError(f"Source leakage: {leakage}")

    source_hash=hashlib.sha256(src.read_bytes()).hexdigest()
    summary={
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":source_hash,
      "eligible_source_count":len(sources),"rejected_source_count":len(rejected),
      "rejected_sources_by_reason":dict(Counter(r["reason"] for r in rejected)),
      "source_split_sizes":{s:len(v) for s,v in splits.items()},
      "counts":{
        s:{
          "total":len(all_rows[s]),
          "by_target":dict(Counter(r["target_bloom_level"] for r in all_rows[s])),
          "cross_level_rows":sum(r["source_bloom_level"]!=r["target_bloom_level"] for r in all_rows[s]),
          "identity_rows":sum(r["source_bloom_level"]==r["target_bloom_level"] for r in all_rows[s])
        } for s in all_rows
      },
      "candidate_statistics":stats,
      "source_leakage_check":leakage,
      "notes":[
        "Synthetic supervision; not human gold.",
        "Built directly from data/figshare_bloom_v1.csv.",
        "No LLM generation or LLM judging is used.",
        "Source topic wording is preserved; cross-level rows change only the task frame.",
        "Unsupported target transformations are excluded rather than invented.",
        "Same-level source questions are retained as identity supervision.",
      ]
    }
    manifest={
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":source_hash,
      "counts":{s:len(v) for s,v in all_rows.items()},
      "source_leakage_check":leakage
    }
    (out/"dataset_statistics.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    write_jsonl(out/"rejected_sources.jsonl",rejected)
    (out/"README.md").write_text(
      "# bloom_rewrite_synth_v11\n\n"
      "High-fidelity deterministic Bloom-target rewrite supervision built directly "
      "from data/figshare_bloom_v1.csv. Cross-level rows preserve the source "
      "content and change only the cognitive task when the target operation is "
      "supported. No LLM is used.\\n",encoding="utf-8")
    print("ELIGIBLE SOURCES:",len(sources))
    print("SOURCE SPLITS:",{s:len(v) for s,v in splits.items()})
    print("COUNTS:",json.dumps(summary["counts"],indent=2))
    print("REJECTED:",summary["rejected_sources_by_reason"])
    print("LEAKAGE:",leakage)
    print("OUTPUT:",out)

if __name__=="__main__":
    main()
