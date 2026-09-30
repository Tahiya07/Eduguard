#!/usr/bin/env python3
"""Build conservative Bloom-target rewrite supervision from data/figshare_bloom_v1.csv.

v10 uses source-task structure, not broad keyword matching. It favors fewer
clean cross-level pairs over large numbers of questionable rewrites.
"""

from __future__ import annotations
import argparse,csv,hashlib,json,re,random
from collections import Counter
from pathlib import Path

LEVELS=["Remember","Understand","Apply","Analyze","Evaluate","Create"]
MAP={"knowledge":"Remember","remembering":"Remember","recall":"Remember",
"comprehension":"Understand","understanding":"Understand","application":"Apply",
"applying":"Apply","analysis":"Analyze","analysing":"Analyze","analyzing":"Analyze",
"evaluation":"Evaluate","evaluating":"Evaluate","synthesis":"Create","creating":"Create"}
DATASET_VERSION="bloom_rewrite_synth_v10"
POLICY_VERSION="bloom_target_policy_v10_source_structured"
SOURCE_FILE="data/figshare_bloom_v1.csv"
SEED=42

COGNITIVE=set("""define explain describe list name state identify recall recognize
recognise summarize summarise interpret classify illustrate apply use calculate
compute determine find solve implement demonstrate analyze analyse compare contrast
differentiate distinguish examine evaluate assess critique criticize criticise judge
justify defend design develop construct formulate propose create devise produce
generate write build show discuss select choose suggest recommend predict recite
outline label draw sketch appraise specify modify estimate measure relate rank
""".split())

STOP=set("""a an the and or but if then than that this these those it its of in on
at to for from with by as is are was were be been being do does did can could may
might will would should must into over under after before during through about
against between among within without using used your our their you we someone
somebody one""".split())

# Phrases describing the requested operation rather than subject matter.
META_TAIL_RE=re.compile(
    r"""(?:\s+show\s+your\s+working(?:\s+and\s+calculation)?|
    \s+justify\s+your\s+answer|
    \s+support\s+(?:your\s+answer|your\s+views?)|
    \s+elaborate(?:\s+your\s+answer)?|
    \s+provide\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|a|an)
    (?:\s+\w+){0,4}\s+examples?|
    \s+give\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|a|an)
    (?:\s+\w+){0,4}\s+examples?)\s*\.?$""",re.I|re.X)

PREFIX_NOISE_RE=re.compile(
    r"""^(?:
    with\s+(?:an?\s+)?appropriate\s+(?:example|examples|diagram|diagrams|drawing|drawings)
    \s*[:,]?\s*|
    by\s+using\s+(?:an?\s+)?appropriate\s+(?:diagram|diagrams|example|examples)
    \s*[:,]?\s*|
    using\s+(?:an?\s+)?appropriate\s+(?:diagram|diagrams|example|examples)
    \s*[:,]?\s*|
    in\s+(?:about\s+\d+\s+words|a\s+table(?:\s+form)?|table\s+form)
    \s*[:,]?\s*|
    through\s+(?:role[- ]play|a\s+role[- ]play)\s*[:,]?\s*
    )""",re.I|re.X)

PLACEHOLDER_RE=re.compile(r"(?:\.{2,}|\[[^\]]*\.{2,}[^\]]*\]|<[^>]*\.{2,}[^>]*>)")
MISSING_CONTEXT=("above data","above graph","above diagram","following code",
"following segment","following passage","given below","this article",
"the article","this magazine article","the figure above","the table above")

NUM_RE=re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE=re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])")
FILE_RE=re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])")
ACRONYM_RE=re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")

LEADING_RE=re.compile(
    r"""^\s*(?:briefly\s+|critically\s+|carefully\s+|concisely\s+)*(?:please\s+)?
    (?P<verb>
      evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|
      compare|contrast|differentiate|distinguish|analyze|analyse|examine|
      calculate|compute|determine|find|solve|apply|use|implement|demonstrate|
      design|develop|construct|formulate|propose|create|devise|produce|build|
      write|draw|sketch|explain|describe|summarize|summarise|interpret|classify|
      illustrate|discuss|define|identify|name|list|state|recite|outline|label|
      predict|select|choose|specify|modify|estimate|measure|relate|rank
    )\b\s*""",re.I|re.X)

HOW_RE=re.compile(
    r"""^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+
    (?:you|we|someone|somebody|one)\s+
    (?P<verb>define|explain|describe|compare|contrast|differentiate|distinguish|
    analyze|analyse|examine|calculate|compute|determine|find|solve|apply|use|
    implement|demonstrate|construct|design|develop|formulate|propose|create|
    devise|produce|build|write|draw|sketch|identify|classify|interpret|select|
    choose|modify|estimate|measure|relate|rank)\b\s+""",re.I|re.X)

HOW_DIFFER_RE=re.compile(
    r"^\s*how\s+(?:does|do|did)\s+(.+?)\s+differ\s+from\s+(.+)$",re.I)
COMPARE_BETWEEN_RE=re.compile(r"^\s*(?:differentiate|distinguish)\s+between\s+(.+)$",re.I)
COMPARE_RE=re.compile(r"^\s*(?:compare|contrast)\s+(.+)$",re.I)

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

def source_form(q):
    q=norm(q).strip(' "').rstrip("?.").strip()
    m=HOW_DIFFER_RE.match(q)
    if m:
        return f"how {m.group(1)} differs from {m.group(2)}","how_differ"
    m=HOW_RE.match(q)
    if m:
        return q[m.end():].strip(" .;:"),f"how_{m.group('verb').lower()}"
    m=COMPARE_BETWEEN_RE.match(q)
    if m:
        return f"the differences between {m.group(1)}","differentiate"
    m=COMPARE_RE.match(q)
    if m:
        return f"the similarities and differences between {m.group(1)}","compare"
    m=LEADING_RE.match(q)
    if m:
        rest=q[m.end():].strip(" .;:,")
        rest=PREFIX_NOISE_RE.sub("",rest).strip()
        rest=META_TAIL_RE.sub("",rest).strip(" .;:,")
        # A second cognitive action means this item is not a clean one-task source.
        if re.search(r"\b(?:and|then|also|followed by)\s+(?:"+
                     "|".join(sorted(COGNITIVE))+r")\b",rest,re.I):
            return None,"multiple_actions"
        if re.search(r";\s*(?:"+"|".join(sorted(COGNITIVE))+r")\b",rest,re.I):
            return None,"multiple_actions"
        return rest,m.group("verb").lower()
    # Clean direct "what is/are X" questions.
    m=re.match(r"^\s*what\s+(?:is|are|was|were)\s+(.+)$",q,re.I)
    if m:
        return m.group(1).strip(" .;:"),"what"
    return None,"unrecognized_form"

def source_eligible(q):
    q=norm(q).strip(' "')
    if not (35<=len(q)<=900): return False,"length"
    if PLACEHOLDER_RE.search(q): return False,"placeholder"
    if any(x in q.lower() for x in MISSING_CONTEXT): return False,"missing_context"
    # Multiple terminal sentences are not treated as one clean training task.
    parts=re.split(r"(?<=[.!?])\s+(?=[A-Z])",q)
    if len(parts)>1: return False,"multiple_sentences"
    if q.count("?")>1: return False,"multiple_questions"
    if len(content(q))<6: return False,"thin_content"
    topic,form=source_form(q)
    if not topic: return False,form
    if len(content(topic))<4: return False,"thin_topic"
    if any(x in topic.lower() for x in MISSING_CONTEXT): return False,"missing_context"
    # Do not preserve a hidden second Bloom command in the topic.
    if re.search(r"\b(?:"+"|".join(sorted(COGNITIVE))+r")\b",topic,re.I):
        return False,"cognitive_residue"
    if re.match(r"(?i)^(with|by|to|for|from|on|at)\b",topic.strip()):
        return False,"prefix_residue"
    return True,""

def capability(source,topic,form):
    low=(source+" "+topic).lower()
    return {
      "rememberable": form not in {"calculate","compute","find","solve","apply","use",
                      "implement","demonstrate","design","develop","construct","formulate",
                      "propose","create","devise","produce","build","write","draw","sketch",
                      "modify","how_calculate","how_compute","how_find","how_solve"},
      "apply": form in {"calculate","compute","find","solve","apply","use","implement",
                        "demonstrate","modify"} or any(x in low for x in (
                            "algorithm","procedure","formula","equation","method","process"
                        )),
      "analyze": form in {"compare","contrast","differentiate","distinguish","analyze","analyse",
                          "examine","how_differ"}
                or any(x in low for x in (
                    "between","differences","relationships","relationship","interaction",
                    "components","component","parts","structure","patterns","causes","effects"
                )),
      "evaluate": form in {"evaluate","assess","appraise","judge","justify","critique",
                           "defend","recommend"}
                   or any(x in low for x in (
                       "advantage","advantages","disadvantage","disadvantages","effectiveness",
                       "appropriate","suitable","best","better","worse","risk","risks",
                       "alternative","alternatives","opinion","agree","choice","choices"
                   )),
      "create": form in {"design","develop","construct","formulate","propose","create",
                         "devise","produce","build","write","draw","sketch"} or
                 any(x in low for x in (
                     "strategy","plan","solution","model","framework","system",
                     "program","algorithm","presentation","diagram","hypothesis",
                     "project","campaign","poster","prototype","storyboard"
                 )),
    }

def target_supported(source,level,topic,form,target):
    cap=capability(source,topic,form)
    if target=="Remember":
        return cap["rememberable"] and not re.match(r"(?i)^(how|why|whether|if|when|where)\b",topic)
    if target=="Understand":
        return True
    if target=="Apply":
        return cap["apply"]
    if target=="Analyze":
        return cap["analyze"]
    if target=="Evaluate":
        return cap["evaluate"]
    if target=="Create":
        return cap["create"]
    return False

def transform(source,topic,form,target):
    low=(source+" "+topic).lower()
    cap=capability(source,topic,form)

    if target=="Remember":
        if cap["analyze"] and any(x in low for x in ("between","differences","similarities")):
            return f"State the differences or similarities in {topic}.","remember_relation"
        return f"State {topic}.","remember_state"

    if target=="Understand":
        return f"Explain {topic}.","understand_explain"

    if target=="Apply":
        if any(x in low for x in ("algorithm","procedure","formula","equation","method","process")):
            return f"Apply {topic} to a practical problem.","apply_method"
        return f"Apply your knowledge of {topic} in a practical situation.","apply_context"

    if target=="Analyze":
        return f"Analyze {topic}.","analyze_direct"

    if target=="Evaluate":
        return f"Evaluate {topic} and justify your judgment.","evaluate_judgment"

    if target=="Create":
        if re.search(r"\b(?:program|algorithm)\b",low) and re.search(
            r"\b(?:c program|program|algorithm|source code|code snippet|nested loops)\b",low):
            return f"Develop a program or algorithm related to {topic}.","create_computational"
        if re.search(r"\b(?:diagram|drawing|sketch|flow chart|flowchart)\b",low):
            return f"Design a diagram representing {topic}.","create_diagram"
        if re.search(r"\b(?:presentation|slide|storyboard)\b",low):
            return f"Develop a presentation about {topic}.","create_presentation"
        if re.search(r"\b(?:strategy|plan|proposal)\b",low):
            return f"Develop a strategy or plan for {topic}.","create_strategy"
        if re.search(r"\b(?:model|framework|system)\b",low):
            return f"Construct a model or framework for {topic}.","create_model"
        if "hypothesis" in low:
            return f"Formulate a hypothesis about {topic}.","create_hypothesis"
        if re.search(r"\b(?:story|novel|poem|play|letter)\b",low):
            return f"Create a response related to {topic}.","create_creative"
        return f"Develop a solution for {topic}.","create_solution"

    raise ValueError(target)

def validate(source,topic,target,rewrite):
    low=rewrite.lower()
    reasons=[]
    if any(x in low for x in ("the rewritten question","original question","bloom level",
                               "as an ai","given constraints","provided code","provided data",
                               "academic artifact")):
        reasons.append("META_LANGUAGE")
    if protected(source)-protected(rewrite):
        reasons.append("PROTECTED_SPAN_LOSS")
    if target!="Evaluate" and re.search(
        r"\b(?:and|then|also|followed by)\s+(?:define|explain|describe|list|name|state|identify|"
        r"calculate|compute|determine|find|solve|apply|use|implement|demonstrate|analyze|"
        r"analyse|compare|contrast|differentiate|examine|evaluate|assess|judge|justify|"
        r"design|develop|construct|formulate|propose|create|devise|produce|build|write|"
        r"draw|sketch|discuss)\b",low,re.I):
        reasons.append("MULTIPLE_ACTIONS")
    starts={
      "Remember":r"^(state|identify|list)\b",
      "Understand":r"^explain\b",
      "Apply":r"^apply\b",
      "Analyze":r"^analyze\b",
      "Evaluate":r"^evaluate\b",
      "Create":r"^(develop|design|construct|formulate|create)\b",
    }
    if not re.search(starts[target],low):
        reasons.append("TARGET_OPERATION_MISSING")
    src=content(topic); out=content(rewrite)
    recall=len(src&out)/max(1,len(src))
    if recall<0.95:
        reasons.append("TOPIC_CONTENT_LOSS")
    return not reasons,{
      "reasons":reasons,
      "topic_content_recall":round(recall,4),
      "protected_spans":sorted(protected(source)),
      "missing_protected":sorted(protected(source)-protected(rewrite)),
    }

def make_row(split,target,s,rewrite,template,info):
    system=("Rewrite an academic question to the requested Bloom level. Preserve "
            "the source topic, technical concepts, quantities, named entities, and "
            "constraints. Do not answer the question. Output only one student-facing "
            "exam question.")
    user=f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"
    packed=(
      "<|im_start|>system\n"+system+"<|im_end|>\n"
      "<|im_start|>user\n"+user+"<|im_end|>\n"
      "<|im_start|>assistant\n"+rewrite+"<|im_end|>"
    )
    synthetic = rewrite != s["source_question"]
    return {
      "example_id":hashlib.sha256(
        f"{DATASET_VERSION}|{s['source_id']}|{target}|{rewrite}".encode()
      ).hexdigest()[:16],
      "source_id":s["source_id"],"group_id":s["group_id"],"split":split,
      "source_question":s["source_question"],"source_bloom_level":s["source_bloom_level"],
      "target_bloom_level":target,"target_rewrite":rewrite,
      "transformation_type":f"{s['source_bloom_level']}->{target}",
      "synthetic_or_original":"synthetic" if synthetic else "original_identity",
      "synthetic":synthetic,"dataset_version":DATASET_VERSION,
      "policy_version":POLICY_VERSION,"quality_status":"pass",
      "construction_method":"deterministic_source_structured",
      "construction_template":template,"source_file":SOURCE_FILE,
      "generator_inputs":["source_question","target_bloom_level"],
      "validation":info,"messages":[
        {"role":"system","content":system},
        {"role":"user","content":user},
        {"role":"assistant","content":rewrite}
      ],
      "text":packed
    }

def write_jsonl(path,rows):
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

    sources=[]; rejected=[]
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
            topic,form=source_form(q)
            sid="src_"+hashlib.sha256(q.encode()).hexdigest()[:16]
            if sid in seen: continue
            seen.add(sid)
            sources.append({
              "source_id":sid,"group_id":int(hashlib.sha256(sid.encode()).hexdigest()[:8],16),
              "source_question":q,"source_bloom_level":level,"topic":topic,"form":form
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

                if not target_supported(s["source_question"],s["source_bloom_level"],s["topic"],s["form"],target):
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
              "identity":sum(r["source_bloom_level"]==r["target_bloom_level"] for r in target_rows)
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
        "Only coherent single-task source questions are admitted.",
        "Target-specific source capability gates are applied before cross-level transformation.",
        "No artificial target padding is used.",
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
      "# bloom_rewrite_synth_v10\n\n"
      "Conservative deterministic Bloom-target rewrite supervision built directly "
      "from the authoritative Figshare corpus. No LLM generation or judging is used. "
      "Cross-level rows are emitted only when source task form supports the target "
      "operation. Same-level rows retain the original source question.\n",
      encoding="utf-8"
    )

    print("ELIGIBLE SOURCES:",len(sources))
    print("SOURCE SPLITS:",{s:len(v) for s,v in splits.items()})
    print("COUNTS:",json.dumps(summary["counts"],indent=2))
    print("REJECTED:",summary["rejected_sources_by_reason"])
    print("LEAKAGE:",leakage)
    print("OUTPUT:",out)

if __name__=="__main__":
    main()
