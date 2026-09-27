#!/usr/bin/env python3
"""Build a high-fidelity Bloom target rewrite dataset from the raw Figshare corpus.

Quality rule: preserve the source task/content and make only a minimal, valid
change to the cognitive operation. Unsupported transformations are omitted.
No LLM is used.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, random, re
from collections import Counter
from pathlib import Path

LEVELS=["Remember","Understand","Apply","Analyze","Evaluate","Create"]
MAP={"knowledge":"Remember","remembering":"Remember","recall":"Remember",
     "comprehension":"Understand","understanding":"Understand",
     "application":"Apply","applying":"Apply","analysis":"Analyze",
     "analysing":"Analyze","analyzing":"Analyze","evaluation":"Evaluate",
     "evaluating":"Evaluate","synthesis":"Create","creating":"Create"}

DATASET_VERSION="bloom_rewrite_synth_v13"
POLICY_VERSION="bloom_target_policy_v13_source_form_strict"
SOURCE_FILE="data/figshare_bloom_v1.csv"
SEED=42

COGNITIVE=set("""
define explain describe list name state identify recall recognize recognise
summarize summarise interpret classify illustrate apply use calculate compute
determine find solve implement demonstrate analyze analyse compare contrast
differentiate distinguish examine evaluate assess critique criticize criticise judge
justify defend design develop construct formulate propose create devise produce
generate write build show discuss select choose suggest recommend predict recite
outline label draw sketch appraise specify modify estimate measure relate rank
compose
""".split())

STOP=set("""
a an the and or but if then than that this these those it its of in on at to
for from with by as is are was were be been being do does did can could may
might will would should must into over under after before during through about
against between among within without using used your our their you we someone
somebody one
""".split())

NUM_RE=re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE=re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])")
FILE_RE=re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])")
ACRONYM_RE=re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")
PLACEHOLDER_RE=re.compile(r"(?:\.{2,}|\[[^\]]*\.{2,}[^\]]*\]|<[^>]*\.{2,}[^>]*>)")

MISSING_CONTEXT=(
    "above data","above graph","above diagram","following code","following segment",
    "following passage","given below","this article","the article","this magazine article",
    "the figure above","the table above",
)

LEADING=re.compile(
r"""^\s*(?:briefly\s+|critically\s+|carefully\s+|concisely\s+)*
(?:please\s+)?
(?P<verb>
evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|
compare|contrast|differentiate|distinguish|analyze|analyse|examine|
calculate|compute|determine|find|solve|apply|use|implement|demonstrate|
design|develop|construct|formulate|propose|create|devise|produce|build|
write|draw|sketch|explain|describe|summarize|summarise|interpret|classify|
illustrate|discuss|define|identify|name|list|state|recite|outline|label|
predict|select|choose|specify|modify|estimate|measure|relate|rank|compose
)\b\s*""",re.I|re.X)

HOW=re.compile(
r"""^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+
(?:you|we|someone|somebody|one)\s+
(?P<verb>
define|explain|describe|compare|contrast|differentiate|distinguish|analyze|analyse|
examine|calculate|compute|determine|find|solve|apply|use|implement|demonstrate|
construct|design|develop|formulate|propose|create|devise|produce|build|write|
draw|sketch|identify|classify|interpret|select|choose|modify|estimate|measure|
relate|rank
)\b\s+""",re.I|re.X)

WH_RULE=re.compile(r"^\s*what\s+(?:is|are|was|were)\s+(.+)$",re.I)
HOW_DIFFER=re.compile(r"^\s*how\s+(?:does|do|did)\s+(.+?)\s+differ\s+from\s+(.+?)\s*$",re.I)
COMPARE_BETWEEN=re.compile(r"^\s*(?:differentiate|distinguish)\s+between\s+(.+)$",re.I)
COMPARE_SIMPLE=re.compile(r"^\s*(?:compare|contrast)\s+(.+)$",re.I)

LEADING_FRAME=re.compile(
r"^(?:by\s+using|using|with\s+the\s+aid\s+of|with)\s+(?:an?\s+)?"
r"(?:appropriate\s+)?(?:diagram|diagrams|drawing|drawings|example|examples)\s*[:,]?\s*",
re.I
)

TAIL_FRAME=re.compile(
r"""(?:\s+show\s+your\s+working(?:\s+and\s+calculation)?|
\s+justify\s+your\s+answer|
\s+support\s+(?:your\s+answer|your\s+views?)|
\s+elaborate(?:\s+your\s+answer)?)
\s*\.?$""",re.I|re.X)

# Exact lexical indicators for semantic support. These are intentionally
# narrower than generic word-substring heuristics.
APPLY_TERMS=("calculate","compute","determine","find","solve","apply","use",
             "implement","demonstrate","method","methods","procedure","procedures",
             "algorithm","formula","equation","process","given data","scenario",
             "case study","problem","modify")
ANALYZE_TERMS=("compare","contrast","differentiate","distinguish","difference",
               "differences","similarity","similarities","relationship",
               "relationships","interaction","interactions","components","component",
               "parts","structure","patterns","pattern","causes","cause","effects","effect")
EVALUATE_TERMS=("advantages and disadvantages","advantages","disadvantages",
                "effectiveness","suitability","appropriate","suitable","best",
                "better","worse","recommend","recommendation","judge","justify",
                "defend","agree","opinion","alternative","alternatives","choice",
                "choices","criteria","evidence","strengths","limitations")
ARTIFACT_PATTERNS=(
    r"\b(?:a|an|the)\s+(?:diagram|drawing|sketch|flow chart|flowchart)\b",
    r"\b(?:a|an|the)\s+(?:program|algorithm|presentation|storyboard)\b",
    r"\b(?:a|an|the)\s+(?:strategy|plan|proposal|model|framework|system)\b",
    r"\b(?:a|an|the)\s+(?:hypothesis|project|campaign|poster|prototype)\b",
    r"\b(?:a|an|the)\s+(?:story|play|letter|dialogue)\b",
    r"\b(?:diagram|drawing|sketch|flow chart|flowchart)\b",
    r"\b(?:program|algorithm|presentation|storyboard)\b",
    r"\b(?:strategy|plan|proposal|model|framework|prototype|project|campaign|poster)\b",
    r"\b(?:hypothesis|story|play|letter)\b",
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

def parse(q):
    q=norm(q).strip(' "').rstrip("?.").strip()

    m=HOW_DIFFER.match(q)
    if m:
        return f"{m.group(1)} differs from {m.group(2)}","how_differ"

    m=HOW.match(q)
    if m:
        rest=norm(q[m.end():])
        rest=LEADING_FRAME.sub("",rest).strip(" .;:,")
        rest=TAIL_FRAME.sub("",rest).strip(" .;:,")
        return (rest or None),m.group("verb").lower()

    m=WH_RULE.match(q)
    if m:
        return m.group(1).strip(" .;:"),"what"

    m=COMPARE_BETWEEN.match(q)
    if m:
        return f"the differences between {m.group(1)}","differentiate"

    m=COMPARE_SIMPLE.match(q)
    if m:
        return f"the similarities and differences between {m.group(1)}",m.group(0).split()[0].lower()

    m=LEADING.match(q)
    if m:
        rest=norm(q[m.end():])
        rest=LEADING_FRAME.sub("",rest).strip(" .;:,")
        rest=TAIL_FRAME.sub("",rest).strip(" .;:,")
        if not rest: return None,m.group("verb").lower()
        # Do not attempt to splice two independent cognitive tasks.
        if re.search(r"\b(?:and|then|also|followed by)\s+(?:"+
                     "|".join(sorted(COGNITIVE))+r")\b",rest,re.I):
            return None,"multiple_actions"
        return rest,m.group("verb").lower()

    return None,"unrecognized"

def eligible(q):
    q=norm(q).strip(' "')
    if not (30<=len(q)<=850): return False,"length"
    if PLACEHOLDER_RE.search(q): return False,"placeholder"
    if any(x in q.lower() for x in MISSING_CONTEXT): return False,"missing_context"
    if q.count("?")>1: return False,"multiple_questions"
    if len(re.findall(r"(?<=[.!?])\s+(?=[A-Z])",q))>1:
        return False,"multiple_sentences"
    if len(content(q))<5: return False,"thin_content"
    topic,form=parse(q)
    if not topic or len(content(topic))<3: return False,form or "thin_topic"
    # Do not let a second explicit action survive in the extracted topic.
    if re.search(r"\b(?:"+"|".join(sorted(COGNITIVE))+r")\b",topic,re.I):
        return False,"cognitive_residue"
    return True,""

def has_any_phrase(text, phrases):
    low=text.lower()
    return any(
        re.search(r"(?<![A-Za-z])"+re.escape(p)+r"(?![A-Za-z])",low)
        if " " not in p else p in low
        for p in phrases
    )

def support(source,topic,form,target):
    low=(source+" "+topic).lower()

    if target=="Remember":
        # Safe recall transformations come from conceptual/factual prompts,
        # not calculations or requests to create an artifact.
        if form in {"calculate","compute","determine","find","solve","apply","use",
                    "implement","demonstrate","design","develop","construct","formulate",
                    "propose","create","devise","produce","build","write","draw","sketch",
                    "modify","compose"}:
            return has_any_phrase(topic,("difference","differences","similarity","similarities"))
        return True

    if target=="Understand":
        # Calculation, drawing and novel-generation tasks do not become valid
        # understanding questions merely by replacing the verb.
        if form in {"calculate","compute","determine","find","solve","draw","sketch",
                    "write","compose","create"}:
            return False
        if form in {"design","develop","construct","formulate","propose","devise","produce","build"}:
            bad=("new song","new titles","another ending","sequel","diary","compose",
                 "original story","write a new","create a new")
            return not any(x in low for x in bad)
        return True

    if target=="Apply":
        # Application requires a method/procedure/rule/problem context, not an
        # arbitrary fact or explanatory "why/how" clause.
        if re.match(r"(?i)^(why|how|whether)\b",topic.strip()):
            return False
        return has_any_phrase(low,APPLY_TERMS)

    if target=="Analyze":
        return has_any_phrase(low,ANALYZE_TERMS)

    if target=="Evaluate":
        return has_any_phrase(low,EVALUATE_TERMS)

    if target=="Create":
        return any(re.search(p,low) for p in ARTIFACT_PATTERNS)

    return False

def make_rewrite(source,topic,form,target):
    low=(source+" "+topic).lower()

    if target=="Remember":
        m=re.match(r"(?i)^the differences between\s+(.+)$",topic)
        if m: return f"State the differences between {m.group(1)}.","remember_differences"
        m=re.match(r"(?i)^(.+?) differs from (.+)$",topic)
        if m: return f"State the difference between {m.group(1)} and {m.group(2)}.","remember_difference"
        if form in {"list","name","define","what","state","identify"}:
            return f"State {topic}.","remember_state"
        if re.search(r"(?i)\b(?:advantages?|disadvantages?|types?|factors?|steps?|phases?|stages?|methods?)\b",topic):
            return f"List {topic}.","remember_list"
        return f"Identify {topic}.","remember_identify"

    if target=="Understand":
        return f"Explain {topic}.","understand_explain"

    if target=="Apply":
        if re.search(r"(?i)\b(?:method|procedure|algorithm|formula|equation|process)\b",topic):
            return f"Apply {topic} to an appropriate problem.","apply_method"
        return f"Apply {topic} in a practical situation.","apply_context"

    if target=="Analyze":
        return f"Analyze {topic}.","analyze_direct"

    if target=="Evaluate":
        return f"Evaluate {topic} and justify your judgment.","evaluate_judgment"

    if target=="Create":
        if re.search(r"(?i)\b(?:diagram|drawing|sketch|flow chart|flowchart)\b",topic):
            return f"Design {topic}.","create_diagram"
        if re.search(r"(?i)\b(?:program|algorithm)\b",topic):
            return f"Develop {topic}.","create_computational"
        if re.search(r"(?i)\b(?:presentation|storyboard)\b",topic):
            return f"Develop {topic}.","create_presentation"
        if re.search(r"(?i)\b(?:strategy|plan|proposal)\b",topic):
            return f"Develop {topic}.","create_strategy"
        if re.search(r"(?i)\b(?:model|framework|system)\b",topic):
            return f"Construct {topic}.","create_model"
        if re.search(r"(?i)\bhypothesis\b",topic):
            return f"Formulate {topic}.","create_hypothesis"
        if re.search(r"(?i)\b(?:story|play|letter|dialogue)\b",topic):
            return f"Create {topic}.","create_creative"
        return f"Develop {topic}.","create_solution"

    raise ValueError(target)

def validate(source,topic,target,rewrite):
    reasons=[]
    low=rewrite.lower()

    if target!="Evaluate" and re.search(
        r"\b(?:and|then|also|followed by)\s+(?:define|explain|describe|list|name|state|identify|"
        r"calculate|compute|determine|find|solve|apply|use|implement|demonstrate|analyze|"
        r"analyse|compare|contrast|differentiate|examine|evaluate|assess|judge|justify|"
        r"design|develop|construct|formulate|propose|create|devise|produce|build|write|draw|"
        r"sketch|discuss)\b",low,re.I):
        reasons.append("multiple_actions")

    if protected(source)-protected(rewrite):
        reasons.append("protected_span_loss")

    # Since the topic is copied from the source, require near-total lexical
    # preservation of its content words.
    src=content(topic); out=content(rewrite)
    recall=len(src&out)/max(1,len(src))
    if recall<0.95:
        reasons.append("topic_content_loss")

    starts={
        "Remember":r"^(state|identify|list)\b",
        "Understand":r"^explain\b",
        "Apply":r"^apply\b",
        "Analyze":r"^analyze\b",
        "Evaluate":r"^evaluate\b",
        "Create":r"^(design|develop|construct|formulate|create)\b",
    }
    if not re.search(starts[target],low):
        reasons.append("target_operation_missing")

    return not reasons,{
        "reasons":reasons,
        "topic_content_recall":round(recall,4),
        "protected_spans":sorted(protected(source)),
        "missing_protected":sorted(protected(source)-protected(rewrite)),
    }

def make_row(split,target,s,rewrite,template,info):
    system=("Rewrite an academic question to the requested Bloom level. Preserve "
            "the source content, technical concepts, quantities, named entities, "
            "and constraints. Do not invent subject matter. Output only one "
            "student-facing exam question.")
    user=f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"
    msgs=[{"role":"system","content":system},
          {"role":"user","content":user},
          {"role":"assistant","content":rewrite}]
    packed=("<|im_start|>system\n"+system+"<|im_end|>\n"
            "<|im_start|>user\n"+user+"<|im_end|>\n"
            "<|im_start|>assistant\n"+rewrite+"<|im_end|>")
    synthetic=rewrite!=s["source_question"]
    return {
      "example_id":hashlib.sha256(f"{DATASET_VERSION}|{s['source_id']}|{target}|{rewrite}".encode()).hexdigest()[:16],
      "source_id":s["source_id"],"group_id":s["group_id"],"split":split,
      "source_question":s["source_question"],"source_bloom_level":s["source_bloom_level"],
      "target_bloom_level":target,"target_rewrite":rewrite,
      "transformation_type":f"{s['source_bloom_level']}->{target}",
      "synthetic_or_original":"synthetic" if synthetic else "original_identity",
      "synthetic":synthetic,"dataset_version":DATASET_VERSION,
      "policy_version":POLICY_VERSION,"quality_status":"pass",
      "construction_method":"deterministic_minimal_task_transform",
      "construction_template":template,"source_file":SOURCE_FILE,
      "generator_inputs":["source_question","target_bloom_level"],
      "validation":info,"messages":msgs,"text":packed
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

    src=Path(args.source); out=Path(args.output_dir)
    if not src.exists(): raise SystemExit(f"Source not found: {src}")
    if out.exists() and not args.overwrite: raise SystemExit(f"Output exists: {out}. Use --overwrite.")
    out.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if p.is_file(): p.unlink()

    sources=[]; rejected=[]; seen=set()
    with src.open("r",encoding="utf-8-sig",newline="") as f:
        for idx,raw in enumerate(csv.DictReader(f)):
            q=norm(raw.get("question","")).strip(' "')
            level=canon(raw.get("bloom_level",""))
            if not q or not level: continue
            ok,why=eligible(q)
            if not ok:
                rejected.append({"row_index":idx,"source_question":q,
                                 "source_bloom_level":level,"reason":why})
                continue
            topic,form=parse(q)
            sid="src_"+hashlib.sha256(q.encode()).hexdigest()[:16]
            if sid in seen: continue
            seen.add(sid)
            sources.append({
              "source_id":sid,
              "group_id":int(hashlib.sha256(sid.encode()).hexdigest()[:8],16),
              "source_question":q,
              "source_bloom_level":level,
              "topic":topic,"form":form
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
                          "protected_spans":sorted(protected(rewrite)),
                          "missing_protected":[],"identity":True}
                    target_rows.append(make_row(split,target,s,rewrite,"identity_source_question",info))
                    continue

                if not supported(s["source_question"],s["topic"],s["form"],target):
                    continue

                rewrite,template=make_rewrite(s["source_question"],s["topic"],s["form"],target)
                ok,info=validate(s["source_question"],s["topic"],target,rewrite)
                if ok:
                    info["identity"]=False
                    target_rows.append(make_row(split,target,s,rewrite,template,info))
            per[target]={
              "total":len(target_rows),
              "cross_level":sum(r["source_bloom_level"]!=r["target_bloom_level"] for r in target_rows),
              "identity":sum(r["source_bloom_level"]==r["target_bloom_level"] for r in target_rows)
            }
            rows.extend(target_rows)
        rows.sort(key=lambda r:(LEVELS.index(r["target_bloom_level"]),r["example_id"]))
        all_rows[split]=rows; stats[split]=per
        write_jsonl(out/f"{split}.jsonl",rows)

    sets={s:{r["source_id"] for r in rows} for s,rows in all_rows.items()}
    leakage={"train_validation":len(sets["train"]&sets["validation"]),
             "train_test":len(sets["train"]&sets["test"]),
             "validation_test":len(sets["validation"]&sets["test"])}
    if any(leakage.values()): raise RuntimeError(f"Source leakage: {leakage}")

    source_hash=hashlib.sha256(src.read_bytes()).hexdigest()
    summary={
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":source_hash,
      "eligible_source_count":len(sources),"rejected_source_count":len(rejected),
      "rejected_sources_by_reason":dict(Counter(r["reason"] for r in rejected)),
      "source_split_sizes":{s:len(v) for s,v in splits.items()},
      "counts":{s:{
        "total":len(all_rows[s]),
        "by_target":dict(Counter(r["target_bloom_level"] for r in all_rows[s])),
        "cross_level_rows":sum(r["source_bloom_level"]!=r["target_bloom_level"] for r in all_rows[s]),
        "identity_rows":sum(r["source_bloom_level"]==r["target_bloom_level"] for r in all_rows[s]),
      } for s in all_rows},
      "candidate_statistics":stats,
      "source_leakage_check":leakage,
      "notes":[
        "Synthetic supervision; not human gold.",
        "Built directly from data/figshare_bloom_v1.csv.",
        "No LLM generation or LLM judging is used.",
        "Cross-level rows use a minimal task-frame transformation only when the source form supports the target.",
        "Unsupported transformations are excluded rather than invented.",
        "Same-level source questions are retained as identity supervision.",
      ]
    }
    manifest={"dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,
              "seed":args.seed,"source_figshare":SOURCE_FILE,
              "source_figshare_sha256":source_hash,
              "counts":{s:len(v) for s,v in all_rows.items()},
              "source_leakage_check":leakage}
    (out/"dataset_statistics.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    write_jsonl(out/"rejected_sources.jsonl",rejected)
    (out/"README.md").write_text(
      "# bloom_rewrite_synth_v13\n\n"
      "High-fidelity deterministic Bloom-target supervision built directly from "
      "data/figshare_bloom_v1.csv. Source content is preserved; unsupported "
      "cross-level transformations are omitted. No LLM is used.\n",
      encoding="utf-8")
    print("ELIGIBLE SOURCES:",len(sources))
    print("SOURCE SPLITS:",{s:len(v) for s,v in splits.items()})
    print("COUNTS:",json.dumps(summary["counts"],indent=2))
    print("REJECTED:",summary["rejected_sources_by_reason"])
    print("LEAKAGE:",leakage)
    print("OUTPUT:",out)

if __name__=="__main__":
    main()
