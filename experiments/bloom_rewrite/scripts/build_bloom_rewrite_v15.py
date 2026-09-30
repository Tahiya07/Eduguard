#!/usr/bin/env python3
"""Build final high-fidelity Bloom rewrite supervision from the raw Figshare corpus.

v15 does not try to infer subject matter with broad keywords. It:
1) detects the actual source task form,
2) keeps the source unchanged when that task already matches the requested target,
3) applies only narrow, hand-auditable cross-level transformations,
4) rejects unsupported transformations.

No LLM generation or judging is used.
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

LEVELS = ["Remember","Understand","Apply","Analyze","Evaluate","Create"]
MAP = {
    "knowledge":"Remember","remembering":"Remember","recall":"Remember",
    "comprehension":"Understand","understanding":"Understand",
    "application":"Apply","applying":"Apply",
    "analysis":"Analyze","analysing":"Analyze","analyzing":"Analyze",
    "evaluation":"Evaluate","evaluating":"Evaluate",
    "synthesis":"Create","creating":"Create"
}
DATASET_VERSION="bloom_rewrite_synth_v15"
POLICY_VERSION="bloom_target_policy_v15_task_form_guarded"
SOURCE_FILE="data/figshare_bloom_v1.csv"
SEED=42

# Primary source-task verbs. Order is from more specific/higher-information
# forms to broad explanatory forms.
FORM_RE = [
    ("Evaluate", re.compile(r"^\s*(?:briefly\s+|critically\s+)?(?:evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|recommend)\b",re.I)),
    ("Analyze", re.compile(r"^\s*(?:briefly\s+)?(?:analyze|analyse|examine|compare|contrast|differentiate|distinguish)\b",re.I)),
    ("Apply", re.compile(r"^\s*(?:calculate|compute|determine|find|solve|apply|use|implement|demonstrate|modify|estimate|measure)\b",re.I)),
    ("Create", re.compile(r"^\s*(?:design|develop|construct|formulate|propose|create|devise|produce|build|compose|write|draw|sketch)\b",re.I)),
    ("Remember", re.compile(r"^\s*(?:define|identify|name|list|state|recite|label|recognize|recognise)\b",re.I)),
    ("Understand", re.compile(r"^\s*(?:explain|describe|summarize|summarise|interpret|classify|illustrate|discuss|retell)\b",re.I)),
]

HOW_FORM = re.compile(r"^\s*how\s+(?:do|does|did|can|could|would|should|will)\b",re.I)
WHY_FORM = re.compile(r"^\s*why\b",re.I)
WHAT_FORM = re.compile(r"^\s*what\s+(?:is|are|was|were)\b",re.I)

STOP=set("""a an the and or but if then than that this these those it its of in
on at to for from with by as is are was were be been being do does did can could
may might will would should must into over under after before during through
about against between among within without using used your our their you we
someone somebody one""".split())

COGNITIVE=set("""define explain describe list name state identify recall recognize recognise
summarize summarise interpret classify illustrate apply use calculate compute determine
find solve implement demonstrate analyze analyse compare contrast differentiate distinguish
examine evaluate assess appraise judge justify critique criticize criticise defend design
develop construct formulate propose create devise produce build compose write draw sketch
select choose suggest recommend predict recite outline label specify modify estimate measure
relate rank discuss retell support""".split())

NUM_RE=re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE=re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])")
FILE_RE=re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])")
ACRONYM_RE=re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")
PLACEHOLDER_RE=re.compile(r"(?:\.{2,}|\[[^\]]*\.{2,}[^\]]*\]|<[^>]*\.{2,}[^>]*>)")

MISSING_CONTEXT=(
    "above data","above graph","above diagram","following code",
    "following segment","following passage","given below","this article",
    "the article","this magazine article","the figure above","the table above"
)

RELATION_PHRASES=(
    "between","among","difference","differences","similarity","similarities",
    "relationship","relationships","interaction","interactions","compare",
    "contrast","differentiate","distinguish","components","component","parts",
    "structure","pattern","patterns","causes","cause","effects","effect"
)

EVAL_PHRASES=(
    "advantages and disadvantages","advantages","disadvantages","effectiveness",
    "suitability","suitable","appropriate","best","better","worse","recommend",
    "recommendation","judge","justify","defend","agree","opinion","alternative",
    "alternatives","choice","choices","criteria","evidence","strengths",
    "limitations","risk","risks","prefer","preference"
)

APPLY_PHRASES=(
    "method","methods","procedure","procedures","algorithm","formula","equation",
    "calculate","compute","determine","solve","find","implement","apply","use",
    "to solve","to calculate","to determine","can be used to","can be applied to",
    "is used to","is applied to"
)

ARTIFACT_PATTERNS=(
    r"\b(?:a|an|the)\s+(?:diagram|drawing|sketch|flow\s*chart|flowchart)\b",
    r"\b(?:a|an|the)\s+(?:program|algorithm|presentation|storyboard)\b",
    r"\b(?:a|an|the)\s+(?:strategy|plan|proposal|model|framework|prototype)\b",
    r"\b(?:a|an|the)\s+(?:hypothesis|project|campaign|poster|story|play|letter|dialogue)\b",
)

MULTI_ACTION= re.compile(
    r"\b(?:and|or|then|also|followed\s+by)\s+(?:"
    r"define|explain|describe|list|name|state|identify|calculate|compute|determine|"
    r"find|solve|apply|use|implement|demonstrate|analyze|analyse|compare|contrast|"
    r"differentiate|distinguish|examine|evaluate|assess|judge|justify|design|develop|"
    r"construct|formulate|propose|create|devise|produce|build|write|draw|sketch|"
    r"discuss|illustrate|interpret|classify|select|choose|modify"
    r")\b",re.I
)

def norm(x): return re.sub(r"\s+"," ",(x or "").replace("\u00a0"," ")).strip()
def canon(x):
    x=str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())
def content(x):
    return {t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*",(x or "").lower())
            if len(t)>2 and t not in STOP and t not in COGNITIVE}
def protected(x):
    out={m.group(0).lower() for m in NUM_RE.finditer(x or "")}
    out.update(m.group(0).lower() for m in TECH_RE.finditer(x or ""))
    out.update(m.group(0).lower() for m in FILE_RE.finditer(x or ""))
    out.update(m.group(1).lower() for m in ACRONYM_RE.finditer(x or ""))
    return out
def has_any(text,phrases):
    low=text.lower()
    return any((p in low) if " " in p else bool(re.search(r"(?<![A-Za-z])"+re.escape(p)+r"(?![A-Za-z])",low)) for p in phrases)

def detect_form(q):
    for level,pattern in FORM_RE:
        if pattern.search(q):
            m=pattern.match(q)
            return level, m.group(0).strip() if m else ""
    if HOW_FORM.match(q):
        return "Understand","how"
    if WHY_FORM.match(q):
        return "Understand","why"
    if WHAT_FORM.match(q):
        return "Remember","what"
    return "Unknown",""

def strip_primary_operator(q):
    q=norm(q).strip(' "').rstrip("?.").strip()
    m=HOW_FORM.match(q)
    if m:
        return q[m.end():].strip(" .;:,")
    m=WHY_FORM.match(q)
    if m:
        return q[m.end():].strip(" .;:,")
    m=WHAT_FORM.match(q)
    if m:
        return q[m.end():].strip(" .;:,")
    for _,pattern in FORM_RE:
        m=pattern.match(q)
        if m:
            return q[m.end():].strip(" .;:,")
    return None

def normalize_topic(q,form):
    q=norm(q).strip(' "').rstrip("?.").strip()

    # Special graphical framing:
    # "Illustrate by using appropriate diagram(s) X" -> topic X
    q=re.sub(
        r"^(?:illustrate|draw|sketch)\s+(?:by\s+using|using|with\s+the\s+aid\s+of)\s+"
        r"(?:an?\s+)?(?:appropriate\s+)?(?:diagram|diagrams|drawing|drawings)\s*[:,]?\s*",
        "",
        q, flags=re.I
    )

    # Direct operator removal.
    topic=strip_primary_operator(q)
    if topic is None:
        return None

    # Remove answer-format residue only when it is clearly extraneous.
    topic=re.sub(
        r"\s+(?:show\s+your\s+working(?:\s+and\s+calculation)?|justify\s+your\s+answer|"
        r"support\s+(?:your\s+answer|your\s+views?)|elaborate(?:\s+your\s+answer)?)\s*$",
        "",
        topic, flags=re.I
    )
    topic=norm(topic).strip(" .;:,")

    # Reject an actual second action.
    if MULTI_ACTION.search(topic):
        return None
    if not topic or len(content(topic))<3:
        return None

    return topic

def source_eligible(q):
    q=norm(q).strip(' "')
    if not (30<=len(q)<=900): return False,"length"
    if PLACEHOLDER_RE.search(q): return False,"placeholder"
    if any(x in q.lower() for x in MISSING_CONTEXT): return False,"missing_context"
    if q.count("?")>1: return False,"multiple_questions"
    # Permit a single source sentence. Multi-clause questions remain possible.
    parts=re.split(r"(?<=[.!?])\s+(?=[A-Z])",q)
    if len(parts)>1: return False,"multiple_sentences"
    level,form=detect_form(q)
    topic=normalize_topic(q,form)
    if not topic: return False,"unparseable_or_multiple_action"
    return True,""

def artifact_kind(source):
    low=source.lower()
    # Require an explicit artifact noun, not merely a technical concept.
    if re.search(r"\b(?:diagram|drawing|sketch|flow\s*chart|flowchart)\b",low):
        return "diagram"
    if re.search(r"\b(?:c\s+program|program|source\s+code|code\s+snippet)\b",low):
        return "program"
    if re.search(r"\b(?:presentation|storyboard)\b",low):
        return "presentation"
    if re.search(r"\b(?:strategy|plan|proposal)\b",low):
        return "strategy"
    if re.search(r"\b(?:model|framework|prototype)\b",low):
        return "model"
    if re.search(r"\bhypothesis\b",low):
        return "hypothesis"
    if re.search(r"\b(?:story|play|letter|dialogue)\b",low):
        return "creative"
    return None

def target_supported(source,source_form,topic,target):
    low=(source+" "+topic).lower()

    # If the actual wording already performs the target operation, keep it.
    if source_form==target:
        return True

    if target=="Remember":
        # Do not convert inherently procedural, generative or explanatory-clause tasks.
        if source_form in {"Apply","Create","Evaluate"}:
            return bool(re.search(r"(?i)\b(?:difference|differences|similarity|similarities)\b",topic))
        return not re.match(r"(?i)^(how|why|whether|if)\b",topic)

    if target=="Understand":
        # Pure calculations and novel-output tasks need different supervision.
        if source_form=="Apply" and re.search(r"(?i)^(calculate|compute|determine|find|solve)\b",source.strip()):
            return False
        if source_form=="Create":
            bad=("new title","new titles","new song","another ending","sequel","diary",
                 "compose a","write a new","create a new")
            return not any(x in low for x in bad)
        return True

    if target=="Apply":
        # Strong application patterns: an explicit method/rule/algorithm/formula
        # connected to an action/object. Exclude generic relation/explanation prompts.
        if re.match(r"(?i)^(why|how|whether)\b",topic):
            # Allow "how X can be used/applied to Y" only when a real method and
            # a concrete operation are both present.
            return bool(
                re.search(r"(?i)\b(?:method|algorithm|formula|equation|procedure)\b",topic)
                and re.search(r"(?i)\b(?:used|applied|implemented|encode|calculate|solve|determine|find|modify|construct)\b",topic)
            )
        return has_any(source,APPLY_PHRASES) and (
            has_any(topic,("method","algorithm","formula","equation","procedure","calculate","compute","solve","determine","find"))
            or source_form=="Apply"
        )

    if target=="Analyze":
        return has_any(low,RELATION_PHRASES)

    if target=="Evaluate":
        return has_any(low,EVAL_PHRASES)

    if target=="Create":
        # An explicit artifact/output is necessary. Mentions of an algorithm or
        # method alone do not imply creation.
        return artifact_kind(source) is not None

    return False

def transform(source,topic,source_form,target):
    low=source.lower()

    if target=="Remember":
        m=re.match(r"(?i)^the difference between (.+?) and (.+)$",topic)
        if m:
            return f"State the difference between {m.group(1)} and {m.group(2)}.","remember_difference"
        if re.match(r"(?i)^how .+ differs from .+$",topic):
            m=re.match(r"(?i)^how (.+?) differs from (.+)$",topic)
            return f"State the difference between {m.group(1)} and {m.group(2)}.","remember_difference"
        if source_form=="Remember":
            return source,"identity"
        if re.match(r"(?i)^(why|how)\b",topic):
            return "", ""
        if re.search(r"(?i)\b(?:advantages?|disadvantages?|types?|factors?|steps?|phases?|stages?|methods?)\b",topic):
            return f"List {topic}.","remember_list"
        return f"Identify {topic}.","remember_identify"

    if target=="Understand":
        if source_form=="Understand":
            return source,"identity"
        return f"Explain {topic}.","understand_explain"

    if target=="Apply":
        if source_form=="Apply":
            return source,"identity"
        # Explain/use-to patterns can be converted into direct application.
        m=re.search(
            r"(?i)^(?:how\s+)?(?:the\s+)?(.+?)\s+can\s+be\s+used\s+to\s+(.+)$",topic
        )
        if m and re.search(r"(?i)\b(?:algorithm|method|formula|equation|procedure)\b",m.group(1)):
            return f"Apply {m.group(1)} to {m.group(2)}.","apply_method_to_task"
        m=re.search(
            r"(?i)^(?:the\s+)?(.+?)\s+(?:is|are)\s+used\s+to\s+(.+)$",topic
        )
        if m and re.search(r"(?i)\b(?:algorithm|method|formula|equation|procedure)\b",m.group(1)):
            return f"Apply {m.group(1)} to {m.group(2)}.","apply_method_to_task"
        if re.search(r"(?i)\b(?:method|algorithm|formula|equation|procedure)\b",topic):
            return f"Apply {topic}.","apply_direct"
        return f"Apply {topic}.","apply_direct"

    if target=="Analyze":
        if source_form=="Analyze":
            return source,"identity"
        return f"Analyze {topic}.","analyze_direct"

    if target=="Evaluate":
        if source_form=="Evaluate":
            return source,"identity"
        return f"Evaluate {topic} and justify your judgment.","evaluate_judgment"

    if target=="Create":
        if source_form=="Create":
            return source,"identity"
        kind=artifact_kind(source)
        if kind=="diagram":
            # For "Illustrate by using a diagram X", create an actual artifact.
            graphical_topic=topic
            return f"Design a diagram illustrating {graphical_topic}.","create_diagram"
        if kind=="program":
            return f"Develop {topic}.","create_program"
        if kind=="presentation":
            return f"Develop {topic}.","create_presentation"
        if kind=="strategy":
            return f"Develop {topic}.","create_strategy"
        if kind=="model":
            return f"Construct {topic}.","create_model"
        if kind=="hypothesis":
            return f"Formulate {topic}.","create_hypothesis"
        if kind=="creative":
            return f"Create {topic}.","create_creative"
        return "", ""

    return "", ""

def validate(source,topic,source_form,target,rewrite):
    if not rewrite:
        return False,["empty"],0.0
    reasons=[]
    if protected(source)-protected(rewrite):
        reasons.append("protected_span_loss")

    starts={
      "Remember":r"^(state|identify|list)\b",
      "Understand":r"^explain\b",
      "Apply":r"^(apply|use)\b",
      "Analyze":r"^analyze\b",
      "Evaluate":r"^evaluate\b",
      "Create":r"^(design|develop|construct|formulate|create)\b",
    }
    if not re.search(starts[target],rewrite,re.I):
        reasons.append("target_operation_missing")

    src=content(topic); out=content(rewrite)
    recall=len(src&out)/max(1,len(src))
    if target!="Evaluate" and target!="Create" and re.search(
        r"\b(?:and|then|also)\s+(?:define|explain|describe|list|name|state|identify|"
        r"calculate|compute|determine|find|solve|apply|use|implement|analyze|analyse|"
        r"compare|contrast|differentiate|examine|evaluate|assess|judge|justify|design|"
        r"develop|construct|formulate|propose|create|write|draw|sketch|discuss)\b",
        rewrite,re.I
    ):
        reasons.append("multiple_actions")

    if recall<0.90:
        reasons.append("topic_content_loss")

    return not reasons,reasons,recall

def make_row(split,target,s,rewrite,template,validation):
    system=("Rewrite an academic question to the requested Bloom level. Preserve "
            "the source content, technical concepts, quantities, named entities, "
            "and constraints. Do not invent subject matter. Output only one "
            "student-facing exam question.")
    user=f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"
    messages=[{"role":"system","content":system},
              {"role":"user","content":user},
              {"role":"assistant","content":rewrite}]
    packed=("<|im_start|>system\n"+system+"<|im_end|>\n"
            "<|im_start|>user\n"+user+"<|im_end|>\n"
            "<|im_start|>assistant\n"+rewrite+"<|im_end|>")
    synthetic=rewrite!=s["source_question"]
    return {
      "example_id":hashlib.sha256(
        f"{DATASET_VERSION}|{s['source_id']}|{target}|{rewrite}".encode()
      ).hexdigest()[:16],
      "source_id":s["source_id"],"group_id":s["group_id"],"split":split,
      "source_question":s["source_question"],
      "source_bloom_level":s["source_bloom_level"],
      "source_detected_level":s["source_detected_level"],
      "target_bloom_level":target,"target_rewrite":rewrite,
      "transformation_type":f"{s['source_bloom_level']}->{target}",
      "source_task_form":s["source_task_form"],
      "synthetic_or_original":"synthetic" if synthetic else "original_identity",
      "synthetic":synthetic,"dataset_version":DATASET_VERSION,
      "policy_version":POLICY_VERSION,"quality_status":"pass",
      "construction_method":"deterministic_strict_source_supported",
      "construction_template":template,"source_file":SOURCE_FILE,
      "generator_inputs":["source_question","target_bloom_level"],
      "validation":validation,"messages":messages,"text":packed
    }

def write_jsonl(path,rows):
    with path.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default=SOURCE_FILE)
    ap.add_argument("--output-dir",default=f"data/bloom_rewrite_versions/{DATASET_VERSION}")
    ap.add_argument("--seed",type=int,default=SEED)
    ap.add_argument("--overwrite",action="store_true")
    args=ap.parse_args()

    source=Path(args.source); out=Path(args.output_dir)
    if not source.exists(): raise SystemExit(f"Source not found: {source}")
    if out.exists() and not args.overwrite: raise SystemExit(f"Output exists: {out}. Use --overwrite.")
    out.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if p.is_file(): p.unlink()

    sources=[]; rejected=[]; seen=set()
    with source.open("r",encoding="utf-8-sig",newline="") as f:
        for idx,raw in enumerate(csv.DictReader(f)):
            q=norm(raw.get("question","")).strip(' "')
            level=canon(raw.get("bloom_level",""))
            if not q or not level: continue
            ok,why=source_eligible(q)
            if not ok:
                rejected.append({"row_index":idx,"source_question":q,
                                 "source_bloom_level":level,"reason":why})
                continue
            detected,task_form=detect_form(q)
            topic=normalize_topic(q,task_form)
            if not topic:
                rejected.append({"row_index":idx,"source_question":q,
                                 "source_bloom_level":level,"reason":"topic_parse_failed"})
                continue
            sid="src_"+hashlib.sha256(q.encode()).hexdigest()[:16]
            if sid in seen: continue
            seen.add(sid)
            sources.append({
              "source_id":sid,"group_id":int(hashlib.sha256(sid.encode()).hexdigest()[:8],16),
              "source_question":q,"source_bloom_level":level,
              "source_detected_level":detected,"source_task_form":task_form,"topic":topic
            })

    splits={"train":[],"validation":[],"test":[]}
    for s in sources:
        u=int(hashlib.sha256(f"{args.seed}|split|{s['source_id']}".encode()).hexdigest()[:8],16)/0xFFFFFFFF
        splits["train" if u<.70 else "validation" if u<.85 else "test"].append(s)

    all_rows={}; summary={}
    for split in ("train","validation","test"):
        rows=[]; per={}
        for target in LEVELS:
            accepted=[]
            for s in splits[split]:
                if s["source_detected_level"]==target:
                    rewrite=s["source_question"]
                    info={"reasons":[],"topic_content_recall":1.0,
                          "protected_spans":sorted(protected(rewrite)),
                          "missing_protected":[],"identity":True}
                    accepted.append(make_row(split,target,s,rewrite,"identity_source_question",info))
                    continue
                if not target_supported(s["source_question"],s["source_detected_level"],s["topic"],target):
                    continue
                rewrite,template=transform(s["source_question"],s["topic"],s["source_task_form"],target)
                ok,reasons,recall=validate(s["source_question"],s["topic"],s["source_task_form"],target,rewrite)
                if ok:
                    info={"reasons":[],"topic_content_recall":round(recall,4),
                          "protected_spans":sorted(protected(s["source_question"])),
                          "missing_protected":[],"identity":False}
                    accepted.append(make_row(split,target,s,rewrite,template,info))
            per[target]={
              "total":len(accepted),
              "cross_level":sum(r["source_detected_level"]!=r["target_bloom_level"] for r in accepted),
              "identity":sum(r["source_detected_level"]==r["target_bloom_level"] for r in accepted)
            }
            rows.extend(accepted)
        rows.sort(key=lambda r:(LEVELS.index(r["target_bloom_level"]),r["example_id"]))
        all_rows[split]=rows; summary[split]=per
        write_jsonl(out/f"{split}.jsonl",rows)

    sets={s:{r["source_id"] for r in rows} for s,rows in all_rows.items()}
    leakage={
      "train_validation":len(sets["train"]&sets["validation"]),
      "train_test":len(sets["train"]&sets["test"]),
      "validation_test":len(sets["validation"]&sets["test"])
    }
    if any(leakage.values()): raise RuntimeError(f"Source leakage detected: {leakage}")

    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    stats={
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":source_hash,
      "eligible_source_count":len(sources),"rejected_source_count":len(rejected),
      "rejected_sources_by_reason":dict(Counter(r["reason"] for r in rejected)),
      "source_split_sizes":{s:len(v) for s,v in splits.items()},
      "counts":{s:{
        "total":len(all_rows[s]),
        "by_target":dict(Counter(r["target_bloom_level"] for r in all_rows[s])),
        "cross_level_rows":sum(r["source_detected_level"]!=r["target_bloom_level"] for r in all_rows[s]),
        "identity_rows":sum(r["source_detected_level"]==r["target_bloom_level"] for r in all_rows[s])
      } for s in all_rows},
      "candidate_statistics":summary,
      "source_leakage_check":leakage,
      "notes":[
        "Synthetic supervision; not human gold.",
        "Built directly from data/figshare_bloom_v1.csv.",
        "No LLM generation or LLM judging is used.",
        "Actual source task form is detected independently of the original dataset label.",
        "If the source already performs the requested target operation, the original question is retained unchanged.",
        "Cross-level rewrites use only narrow source-supported transformations.",
        "Unsupported transformations are excluded rather than invented.",
      ]
    }
    manifest={
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":source_hash,
      "counts":{s:len(v) for s,v in all_rows.items()},
      "source_leakage_check":leakage
    }
    (out/"dataset_statistics.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    write_jsonl(out/"rejected_sources.jsonl",rejected)
    (out/"README.md").write_text(
      "# bloom_rewrite_synth_v15\n\n"
      "High-fidelity deterministic Bloom-target supervision built directly from "
      "data/figshare_bloom_v1.csv. Actual source task form is detected independently "
      "of the source label. Exact target matches are retained; only narrow, source-"
      "supported cross-level transformations are emitted. Unsupported transformations "
      "are excluded. No LLM is used.\\n",
      encoding="utf-8")

    print("ELIGIBLE SOURCES:",len(sources))
    print("SOURCE SPLITS:",{s:len(v) for s,v in splits.items()})
    print("COUNTS:",json.dumps(stats["counts"],indent=2))
    print("REJECTED:",stats["rejected_sources_by_reason"])
    print("LEAKAGE:",leakage)
    print("OUTPUT:",out)

if __name__=="__main__":
    main()
