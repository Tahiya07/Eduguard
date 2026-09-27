#!/usr/bin/env python3
"""Build a clean Bloom-target rewrite dataset from the authoritative Figshare CSV.

The builder is deterministic. It does not use an LLM. It keeps the source
question content as the anchor and changes only the cognitive task frame.
Ambiguous fragments and multi-action prompts are rejected.
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
DATASET_VERSION="bloom_rewrite_synth_v8"
POLICY_VERSION="bloom_target_policy_v8_conservative"
SOURCE_FILE="data/figshare_bloom_v1.csv"
SEED=42

STOP=set("""a an the and or but if then than that this these those it its of in on
at to for from with by as is are was were be been being do does did can could
may might will would should must into over under after before during through
about against between among within without using used your our their you we
someone somebody one""".split())

COGNITIVE=set("""define explain describe list name state identify recall recognize
recognise summarize summarise interpret classify illustrate apply use calculate
compute determine solve implement demonstrate analyze analyse compare contrast
differentiate distinguish examine evaluate assess critique criticize criticise
judge justify defend design develop construct formulate propose create devise
produce generate write build show discuss select choose suggest recommend predict
recite outline label draw sketch appraise specify""".split())

GENERIC=set("""key facts information meaning purpose explanation account description
summary interpretation differences difference similarities similarity comparison
relationships relationship interactions interaction components component parts
structure patterns causes effects effectiveness suitability quality strengths
limitations advantage disadvantages judgment conclusion evidence criteria
appropriate relevant related task problem work outcome result knowledge procedure
approach solution plan strategy model framework method project artifact proposal
system process original new creative""".split())

FORBIDDEN_CONTEXT=("above data","above graph","above diagram","following code",
"following segment","following passage","given below","this article",
"the article","this magazine article","as shown in the")

PLACEHOLDER_RE=re.compile(r"(\.\.\.|\[[^\]]*\.\.\.[^\]]*\]|<[^>]*\.\.\.)")
PUNCT_RE=re.compile(r"[.!?]+")
NUM_RE=re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE=re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])")
FILE_RE=re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])")
ACRONYM_RE=re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")

PREFIX=re.compile(r"""^\s*(?:briefly\s+|critically\s+|carefully\s+)*(?:please\s+)?
(?P<verb>evaluate|assess|appraise|judge|justify|critique|criticize|criticise|
compare|contrast|differentiate|distinguish|analyze|analyse|examine|
calculate|compute|determine|find|solve|apply|use|implement|demonstrate|
design|develop|construct|formulate|propose|create|devise|produce|build|
write|draw|sketch|explain|describe|summarize|summarise|interpret|classify|
illustrate|discuss|define|identify|name|list|state|recite|outline|label|
predict|select|choose|specify|relate|modify|estimate|measure)\b\s*""",re.I|re.X)

HOW=re.compile(r"""^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+
(?:you|we|someone|somebody|one)\s+
(?P<verb>define|explain|describe|compare|contrast|differentiate|distinguish|
analyze|analyse|examine|calculate|compute|determine|find|solve|apply|use|
implement|demonstrate|construct|design|develop|formulate|propose|create|
devise|produce|build|write|draw|sketch|identify|classify|interpret|select|
choose|estimate|measure|modify|relate|rank)\b\s+""",re.I|re.X)

TABLE_PREFIX=re.compile(r"^\s*in\s+(?:a|the)\s+table\s*,?\s*",re.I)

SECOND_ACTION_RE=re.compile(
    r"\b(?:and|then|also|followed\s+by)\s+(?:"
    + "|".join(sorted(COGNITIVE))
    + r")\b",re.I)

TAIL_RE=re.compile(r"""\s+(?:justify(?:\s+your\s+answer)?|support\s+your\s+
(?:answer|views?)|show\s+your\s+working(?:\s+and\s+calculation)?|
give\s+(?:an?|one|two|three|four|five|six|seven|eight|nine|ten)\s+
(?:relevant\s+)?examples?|provide\s+(?:an?|one|two|three|four|five|six|seven|
eight|nine|ten)\s+(?:relevant\s+)?examples?|elaborate(?:\s+your\s+answer)?)
\s*\.?$""",re.I|re.X)

END_BAD=re.compile(r"\b(?:the|a|an|of|for|to|and|or|in|on|with|by|from|as|at|that|which|where|how|why|than)$",re.I)

CREATE_CUES=("design","develop","construct","formulate","propose","create","devise",
"produce","build","write","strategy","plan","solution","problem","system","model",
"framework","method","procedure","algorithm","program","presentation","diagram",
"sketch","hypothesis","approach","project","campaign","story","poster")

def canon(x):
    x=str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())

def norm(x): return re.sub(r"\s+"," ",(x or "").replace("\u00a0"," ")).strip()

def content(x):
    return {t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*",(x or "").lower())
            if len(t)>2 and t not in STOP and t not in COGNITIVE}

def protected(x):
    out={m.group(0).lower() for m in NUM_RE.finditer(x or "")}
    out.update(m.group(0).lower() for m in TECH_RE.finditer(x or ""))
    out.update(m.group(0).lower() for m in FILE_RE.finditer(x or ""))
    out.update(m.group(1).lower() for m in ACRONYM_RE.finditer(x or ""))
    return out

def one_sentence(q):
    temp=re.sub(r"\d+\.\d+","",q)
    # Decimal values are removed above. A remaining period/question mark
    # indicates multiple sentences/questions.
    return len(PUNCT_RE.findall(temp))<=1

def extract_topic(q):
    q=norm(q).strip(' "').rstrip("?.").strip()
    q=TABLE_PREFIX.sub("",q)
    m=HOW.match(q)
    if m:
        rest=q[m.end():].strip(" .;,:")
    else:
        m=PREFIX.match(q)
        if not m:
            return None,"unrecognized_form"
        rest=q[m.end():].strip(" .;,:")
    if not rest or len(content(rest))<4:
        return None,"thin_topic"
    if SECOND_ACTION_RE.search(rest):
        return None,"multiple_actions"
    rest=TAIL_RE.sub("",rest)
    rest=norm(rest).strip(" .;,:")
    if END_BAD.search(rest):
        return None,"fragment_topic"
    if any(x in rest.lower() for x in FORBIDDEN_CONTEXT):
        return None,"missing_context"
    # Do not allow an unambiguous command verb to remain as an isolated
    # instruction inside the extracted topic.
    if re.match(r"^(?:define|explain|describe|list|name|state|identify|calculate|compute|determine|solve|apply|use|implement|demonstrate|analyze|analyse|compare|contrast|differentiate|examine|evaluate|assess|judge|justify|design|develop|construct|formulate|propose|create|devise|produce|build|write|draw|sketch|discuss|classify|interpret)\b",rest,re.I):
        return None,"cognitive_residue"
    return rest,"ok"

def eligible(q):
    q=norm(q).strip(' "')
    if not (35<=len(q)<=900): return False,"length"
    if PLACEHOLDER_RE.search(q): return False,"placeholder"
    if any(x in q.lower() for x in FORBIDDEN_CONTEXT): return False,"missing_context"
    if not one_sentence(q): return False,"multiple_sentences_or_questions"
    if len(content(q))<6: return False,"thin_content"
    return True,""

def source_cues(source, topic):
    low=(source+" "+topic).lower()
    return {
        "apply": any(x in low for x in (
            "calculate","compute","determine","find","solve","apply","use ",
            "implement","demonstrate","algorithm","procedure","method",
            "formula","equation","modify","construct","given","scenario",
            "case","problem","how to"
        )),
        "analyze": any(x in low for x in (
            "compare","contrast","differentiate","distinguish","between",
            "among","relationship","relationships","interaction",
            "interactions","component","components","parts","structure",
            "pattern","patterns","cause","causes","effect","effects",
            "difference","differences","similarities","similarity"
        )),
        "evaluate": any(x in low for x in (
            "evaluate","assess","appraise","judge","justify","critique",
            "defend","recommend","appropriate","suitable","effectiveness",
            "advantage","advantages","disadvantage","disadvantages","best",
            "better","worse","should","opinion","agree","alternative",
            "alternatives","choice","choices","risk","risks","compare"
        )),
        "create": any(x in low for x in (
            "design","develop","construct","formulate","propose","create",
            "devise","produce","build","write","draw","sketch","strategy",
            "plan","solution","system","model","framework","program",
            "algorithm","presentation","diagram","hypothesis","project",
            "campaign","story","poster","prototype"
        )),
    }

def safe_for_target(source, topic, target):
    cues=source_cues(source,topic)
    low=topic.lower()

    if target=="Remember":
        # Do not turn intrinsically generative/procedural tasks into recall by
        # merely swapping the verb.
        if any(x in low for x in (
            "new song","new ending","letter to","design a","develop a",
            "construct a","create a","propose","formulate","develop an",
            "program","algorithm","strategy","plan","solution"
        )):
            return False
        return True

    if target=="Understand":
        return True

    if target=="Apply":
        return cues["apply"]

    if target=="Analyze":
        return cues["analyze"] or (
            len(content(topic)) >= 5 and
            " and " in low
        )

    if target=="Evaluate":
        return cues["evaluate"]

    if target=="Create":
        return cues["create"]

    return False

def transform(topic, target, source):
    cues=source_cues(source,topic)

    if target=="Remember":
        if re.match(r"(?i)^(how|why|whether|if|when|where)\\b",topic):
            return f"State the key facts about {topic}.","remember_facts"
        return f"Identify {topic}.","remember_identify"

    if target=="Understand":
        return f"Explain {topic}.","understand_explain"

    if target=="Apply":
        if any(x in (source+" "+topic).lower() for x in ("calculate","compute","determine","find","solve")):
            return f"Apply the relevant procedure to {topic}.","apply_procedure"
        return f"Apply your knowledge of {topic} in a practical situation.","apply_context"

    if target=="Analyze":
        if cues["analyze"]:
            return f"Analyze {topic}.","analyze_direct"
        return f"Analyze {topic}, focusing on its key relationships or components.","analyze_structure"

    if target=="Evaluate":
        return f"Evaluate {topic} and justify your judgment.","evaluate_judgment"

    if target=="Create":
        low=(source+" "+topic).lower()
        if any(x in low for x in ("program","algorithm","code")):
            artifact="a program or algorithm"
        elif any(x in low for x in ("diagram","drawing","sketch","flow chart","flowchart")):
            artifact="a diagram"
        elif any(x in low for x in ("presentation","slide","storyboard")):
            artifact="a presentation"
        elif any(x in low for x in ("strategy","plan")):
            artifact="a strategy or plan"
        elif any(x in low for x in ("model","framework","system")):
            artifact="a model or framework"
        elif "hypothesis" in low:
            artifact="a hypothesis"
        elif any(x in low for x in ("story","novel","poem","play","dialogue")):
            artifact="a creative response"
        else:
            artifact="a solution or approach"
        return f"Develop {artifact} for {topic}.","create_grounded_artifact"

    raise ValueError(target)

def validate(source, topic, target, rewrite):
    reasons=[]
    low=rewrite.lower()

    if any(x in low for x in (
        "the rewritten question","original question","bloom level",
        "as an ai","given constraints","provided code","provided data",
        "provided passage","academic artifact"
    )):
        reasons.append("META_LANGUAGE")

    # The transformed row must contain only one task operation. Evaluate's
    # integrated "justify your judgment" is explicitly allowed.
    if target!="Evaluate" and re.search(
        r"\\b(?:and|then|also)\\s+(?:define|explain|describe|list|name|state|identify|"
        r"calculate|compute|determine|solve|apply|use|implement|demonstrate|analyze|"
        r"analyse|compare|contrast|differentiate|examine|evaluate|assess|judge|justify|"
        r"design|develop|construct|formulate|propose|create|devise|produce|build|write|"
        r"draw|sketch|discuss)\\b", low):
        reasons.append("MULTIPLE_ACTIONS")

    if protected(source)-protected(rewrite):
        reasons.append("PROTECTED_SPAN_LOSS")

    src=content(topic)
    out=content(rewrite)
    recall=len(src&out)/max(1,len(src))
    if recall<0.95:
        reasons.append("TOPIC_CONTENT_LOSS")

    starts={
        "Remember":r"^(state|identify)\\b",
        "Understand":r"^explain\\b",
        "Apply":r"^apply\\b",
        "Analyze":r"^analyze\\b",
        "Evaluate":r"^evaluate\\b",
        "Create":r"^develop\\b",
    }
    if not re.search(starts[target],low):
        reasons.append("TARGET_OPERATION_MISSING")

    return not reasons,{
        "reasons":reasons,
        "topic_content_recall":round(recall,4),
        "protected_spans":sorted(protected(source)),
        "missing_protected":sorted(protected(source)-protected(rewrite)),
    }

def read_sources(path):
    good=[]; bad=[]
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        for idx,raw in enumerate(csv.DictReader(f)):
            q=norm(raw.get("question","")).strip(' "')
            level=canon(raw.get("bloom_level",""))
            if not q or not level: continue
            ok,why=eligible(q)
            if not ok:
                bad.append({"row_index":idx,"source_question":q,"source_bloom_level":level,"reason":why})
                continue
            topic_text,why2=extract_topic(q)
            if not topic_text:
                bad.append({"row_index":idx,"source_question":q,"source_bloom_level":level,"reason":why2})
                continue
            sid="src_"+hashlib.sha256(q.encode()).hexdigest()[:16]
            good.append({
                "source_id":sid,
                "group_id":int(hashlib.sha256(q.encode()).hexdigest()[:8],16),
                "source_question":q,
                "source_bloom_level":level,
                "topic":topic_text,
            })
    return good,bad

def split_sources(rows,seed):
    out={"train":[],"validation":[],"test":[]}
    for r in rows:
        u=int(hashlib.sha256(f"{seed}|split|{r['source_id']}".encode()).hexdigest()[:8],16)/0xFFFFFFFF
        out["train" if u<.70 else "validation" if u<.85 else "test"].append(r)
    return out

def make_row(split,target,s,rewrite,template,info):
    messages=[
      {"role":"system","content":(
        "Rewrite an academic question to the requested Bloom level. Preserve "
        "the source topic, technical concepts, quantities, named entities, "
        "and constraints. Do not answer the question. Output only one "
        "student-facing exam question."
      )},
      {"role":"user","content":f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"},
      {"role":"assistant","content":rewrite},
    ]
    text="<|im_start|>system\n"+messages[0]["content"]+"<|im_end|>\n<|im_start|>user\n"+messages[1]["content"]+"<|im_end|>\n<|im_start|>assistant\n"+rewrite+"<|im_end|>"
    return {
      "example_id":hashlib.sha256(f"{DATASET_VERSION}|{s['source_id']}|{target}|{rewrite}".encode()).hexdigest()[:16],
      "source_id":s["source_id"],"group_id":s["group_id"],"split":split,
      "source_question":s["source_question"],"source_bloom_level":s["source_bloom_level"],
      "target_bloom_level":target,"target_rewrite":rewrite,
      "transformation_type":f"{s['source_bloom_level']}->{target}",
      "synthetic_or_original":"synthetic","synthetic":True,
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,
      "quality_status":"pass","construction_method":"deterministic_conservative_template",
      "construction_template":template,"source_file":SOURCE_FILE,
      "generator_inputs":["source_question","target_bloom_level"],
      "validation":info,"messages":messages,"text":text
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

    sources,rejected=read_sources(src)
    splits=split_sources(sources,args.seed)
    all_rows={}; cand_stats={}

    for split in ("train","validation","test"):
        rows=[]
        cand_stats[split]={}
        for target in LEVELS:
            accepted=0; cross=0; identity=0; target_rows=[]
            shuffled=splits[split][:]
            random.Random(args.seed+sum(map(ord,target))+len(split)).shuffle(shuffled)

            for s in shuffled:
                if s["source_bloom_level"]==target:
                    rewrite=s["source_question"].rstrip()
                    template="identity_source_question"
                    info={"reasons":[],"topic_content_recall":1.0,
                          "protected_spans":sorted(protected(s["source_question"])),
                          "missing_protected":[]}
                    target_rows.append(make_row(split,target,s,rewrite,template,info))
                    accepted+=1; identity+=1
                    continue

                if not safe_for_target(s["source_question"],s["topic"],target):
                    continue

                rewrite,template=transform(s["topic"],target,s["source_question"])
                ok,info=validate(s["source_question"],s["topic"],target,rewrite)
                if ok:
                    info["cross_level"]=True
                    target_rows.append(make_row(split,target,s,rewrite,template,info))
                    accepted+=1; cross+=1

            cand_stats[split][target]={
                "accepted":accepted,
                "cross_level":cross,
                "identity":identity,
            }
            rows.extend(target_rows)

        rows.sort(key=lambda r:(LEVELS.index(r["target_bloom_level"]),r["example_id"]))
        all_rows[split]=rows
        write_jsonl(out/f"{split}.jsonl",rows)

    sets={s:{r["source_id"] for r in rows} for s,rows in all_rows.items()}
    leakage={
        "train_validation":len(sets["train"]&sets["validation"]),
        "train_test":len(sets["train"]&sets["test"]),
        "validation_test":len(sets["validation"]&sets["test"])
    }
    if any(leakage.values()): raise RuntimeError(f"Source leakage: {leakage}")

    source_hash=hashlib.sha256(src.read_bytes()).hexdigest()
    stats={
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":source_hash,
      "eligible_source_count":len(sources),"rejected_source_count":len(rejected),
      "rejected_sources_by_reason":dict(Counter(x["reason"] for x in rejected)),
      "source_split_sizes":{s:len(v) for s,v in splits.items()},
      "candidate_statistics":cand_stats,
      "counts":{s:{
        "total":len(rows),
        "by_target":dict(Counter(r["target_bloom_level"] for r in rows)),
        "cross_level_rows":sum(r["source_bloom_level"]!=r["target_bloom_level"] for r in rows),
        "identity_rows":sum(r["source_bloom_level"]==r["target_bloom_level"] for r in rows),
        "by_source_target":dict(Counter(f"{r['source_bloom_level']}->{r['target_bloom_level']}" for r in rows))
      } for s,rows in all_rows.items()},
      "source_leakage_check":leakage,
      "notes":[
        "Synthetic supervision; not human gold.",
        "Target-specific adequacy gates prevent unsupported Apply, Analyze, Evaluate, and Create transformations.",

        "Built directly from data/figshare_bloom_v1.csv.",
        "No LLM generation or LLM judging is used.",
        "Multi-action, incomplete, placeholder, and missing-context source prompts are rejected.",
        "Cross-level rows use minimal deterministic target-task frames.",
        "Same-level source questions are retained as identity supervision.",
      ]
    }
    manifest={
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":source_hash,
      "counts":{s:len(v) for s,v in all_rows.items()},"source_leakage_check":leakage
    }
    (out/"dataset_statistics.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    write_jsonl(out/"rejected_sources.jsonl",rejected)
    (out/"README.md").write_text(
      "# bloom_rewrite_synth_v8\n\n"
      "Deterministic conservative Bloom-target rewrite supervision built directly "
      "from the authoritative Figshare corpus. No LLM is used.\n",
      encoding="utf-8")
    print("ELIGIBLE SOURCES:",len(sources))
    print("SOURCE SPLITS:",{s:len(v) for s,v in splits.items()})
    print("CANDIDATES:",json.dumps(cand_stats,indent=2))
    print("ROWS:",{s:len(v) for s,v in all_rows.items()},"TOTAL:",sum(len(v) for v in all_rows.values()))
    print("LEAKAGE:",leakage)
    print("OUTPUT:",out)

if __name__=="__main__":
    main()
