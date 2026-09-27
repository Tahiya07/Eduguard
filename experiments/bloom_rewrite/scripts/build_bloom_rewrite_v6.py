#!/usr/bin/env python3
"""Build a conservative Bloom rewrite dataset from the raw Figshare corpus.

v6 is intentionally deterministic and conservative:
- raw Figshare questions are the only source content
- one coherent source task is required
- leftover Bloom/cognitive commands are rejected
- cross-level transformations use minimal generic task framing
- same-level rows may be retained as authentic identity supervision
- target classes are balanced within each split
- source questions never cross train/validation/test boundaries
"""

from __future__ import annotations
import argparse,csv,hashlib,json,random,re
from collections import Counter
from pathlib import Path

LEVELS=["Remember","Understand","Apply","Analyze","Evaluate","Create"]
MAP={"knowledge":"Remember","remembering":"Remember","recall":"Remember",
     "comprehension":"Understand","understanding":"Understand",
     "application":"Apply","applying":"Apply","analysis":"Analyze",
     "analysing":"Analyze","analyzing":"Analyze","evaluation":"Evaluate",
     "evaluating":"Evaluate","synthesis":"Create","creating":"Create"}
DATASET_VERSION="bloom_rewrite_synth_v6"
POLICY_VERSION="bloom_target_policy_v6_conservative"
SOURCE_FILE="data/figshare_bloom_v1.csv"
SEED=42

COGNITIVE=set("""define explain describe list name state identify recall recognize
recognise summarize summarise interpret classify illustrate apply use calculate
compute determine solve implement demonstrate analyze analyse compare contrast
differentiate distinguish examine evaluate assess critique criticize criticise
judge justify defend design develop construct formulate propose create devise
produce generate write build show discuss select choose suggest recommend predict
recite outline label draw sketch appraise support""".split())

STOP=set("""a an the and or but if then than that this these those it its of in
on at to for from with by as is are was were be been being do does did can could
may might will would should must into over under after before during through
about against between among within without using used your our their you we
someone somebody one""".split())

GENERIC=set("""key information facts meaning purpose explanation account description
summary interpretation differences difference similarities similarity comparison
relationships relationship interactions interaction components component parts
structure patterns causes effects effectiveness suitability quality strengths
limitations advantage disadvantages judgment conclusion evidence criteria
appropriate relevant related task problem work outcome result results knowledge
procedure approach solution plan strategy model framework method project artifact
proposal system process original new creative""".split())

FORBIDDEN_CONTEXT=("above data","above graph","above diagram","following code",
                   "following segment","following passage","given below",
                   "this article","the article","this magazine article")
PLACEHOLDER_RE=re.compile(r"(\.\.\.|\[[^\]]*\.\.\.[^\]]*\]|<[^>]*\.\.\.)")
NUM_RE=re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE=re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])")
FILE_RE=re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])")
ACRONYM_RE=re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")

PREFIX=re.compile(r"""^\s*
(?:briefly\s+|critically\s+|carefully\s+)*
(?:please\s+)?
(?P<verb>
evaluate|assess|appraise|judge|justify|critique|criticize|criticise|
compare|contrast|differentiate|distinguish|analyze|analyse|examine|
calculate|compute|determine|find|solve|apply|use|implement|demonstrate|
design|develop|construct|formulate|propose|create|devise|produce|build|
write|draw|sketch|explain|describe|summarize|summarise|interpret|classify|
illustrate|discuss|define|identify|name|list|state|recite|outline|label|
predict|select|choose|specify
)\b\s*""",re.I|re.X)

HOW=re.compile(r"""^\s*how\s+
(?:do|does|did|can|could|would|should|will)\s+
(?:you|we|someone|somebody|one)\s+
(?P<verb>define|explain|describe|compare|contrast|differentiate|distinguish|
analyze|analyse|examine|calculate|compute|determine|find|solve|apply|use|
implement|demonstrate|construct|design|develop|formulate|propose|create|
devise|produce|build|write|draw|sketch|identify|classify|interpret|select|
choose|estimate|measure|modify|rank|relate|construct)\b\s+""",re.I|re.X)

HOW_COPULA=re.compile(r"""^\s*how\s+
(?:does|do|did|can|could|would|should|will)\s+""",re.I)

WHAT=re.compile(r"^\s*what\s+(?:is|are|was|were)\s+",re.I)


def canon(x):
    x=str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())

def norm(x):
    return re.sub(r"\s+"," ",(x or "").replace("\u00a0"," ")).strip()

def content_tokens(x):
    return {t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*",(x or "").lower())
            if len(t)>2 and t not in STOP and t not in COGNITIVE}

def protected(x):
    out={m.group(0).lower() for m in NUM_RE.finditer(x or "")}
    out.update(m.group(0).lower() for m in TECH_RE.finditer(x or ""))
    out.update(m.group(0).lower() for m in FILE_RE.finditer(x or ""))
    out.update(m.group(1).lower() for m in ACRONYM_RE.finditer(x or ""))
    return out

def coherent_source(q):
    q=norm(q).strip(' "')
    if not (35<=len(q)<=850): return False,"length"
    if PLACEHOLDER_RE.search(q): return False,"placeholder"
    if any(x in q.lower() for x in FORBIDDEN_CONTEXT): return False,"missing_context"
    # Reject multi-question/multi-command material. Decimal points do not count.
    temp=re.sub(r"\d+\.\d+","",q)
    marks=re.findall(r"[.?]+",temp)
    if len(marks)>1: return False,"multiple_sentences_or_questions"
    if len(content_tokens(q))<6: return False,"thin_content"
    return True,""

def extract_topic(q):
    q=norm(q).strip(' "').rstrip("?.").strip()

    m=HOW.match(q)
    if m:
        rest=q[m.end():].strip(" .;,:")
        return rest,"how"

    m=WHAT.match(q)
    if m:
        rest=q[m.end():].strip(" .;,:")
        return rest,"what"

    m=PREFIX.match(q)
    if m:
        rest=q[m.end():].strip(" .;,:")
        # Any additional command in the remaining topic makes the pair unsafe.
        if re.search(r"\b(?:and|then|also|plus|followed by)\s+(?:"+
                     "|".join(sorted(COGNITIVE))+r")\b",rest,re.I):
            return None,"multiple_actions"
        # Remove pure answer-format residue only at the end.
        rest=re.sub(r"""\s+(?:show your working(?: and calculation)?|
                         justify your answer|support your answer|
                         support your views|elaborate(?: your answer)?)
                         \s*\.?$""","",rest,flags=re.I|re.X)
        rest=norm(rest).strip(" .;,:")
        return rest,"imperative"

    # Do not force ambiguous question fragments.
    return None,"unrecognized_form"

def create_supported(source,topic):
    low=(source+" "+topic).lower()
    cues=("design","develop","construct","formulate","propose","create","devise",
          "produce","build","strategy","plan","solution","problem","system",
          "model","framework","method","procedure","algorithm","program",
          "presentation","diagram","sketch","hypothesis","approach")
    return any(c in low for c in cues)

def make_rewrite(topic,target):
    if target=="Remember":
        if re.match(r"(?i)^(how|why|whether|if|when|where)\\b", topic.strip()):
            return f"State the key facts about {topic}.","remember_facts"
        return f"State {topic}.","remember_state"
    if target=="Understand":
        return f"Explain {topic}.","understand_explain"
    if target=="Apply":
        return f"Apply your knowledge of {topic} to a related problem.","apply_related"
    if target=="Analyze":
        return f"Analyze {topic}.","analyze_direct"
    if target=="Evaluate":
        return f"Evaluate {topic} and justify your judgment.","evaluate_judgment"
    if target=="Create":
        return f"Develop a solution or plan related to {topic}.","create_solution"
    raise ValueError(target)

def validate(source,topic,target,rewrite):
    reasons=[]
    src_p=protected(source); out_p=protected(rewrite)
    missing=sorted(src_p-out_p)
    if missing:
        reasons.append("PROTECTED_SPAN_LOSS")

    low=rewrite.lower()
    starts={
        "Remember":r"^(state|list|name|identify|recall|recognize)\\b",
        "Understand":r"^(explain|describe|summarize|interpret|classify|illustrate)\\b",
        "Apply":r"^apply\\b",
        "Analyze":r"^(analyze|analyse)\\b",
        "Evaluate":r"^evaluate\\b",
        "Create":r"^develop\\b",
    }
    if not re.search(starts[target],low):
        reasons.append("TARGET_OPERATION_MISSING")

    src_c=content_tokens(topic)
    out_c=content_tokens(rewrite)
    recall=len(src_c & out_c)/max(1,len(src_c))
    if recall<0.95:
        reasons.append("TOPIC_CONTENT_LOSS")

    return not reasons,{
        "reasons":reasons,
        "topic_content_recall":round(recall,4),
        "protected_spans":sorted(src_p),
        "missing_protected":missing
    }


def read_source(path):
    good=[]; bad=[]
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        for i,raw in enumerate(csv.DictReader(f)):
            q=norm(raw.get("question","")).strip(' "')
            sl=canon(raw.get("bloom_level",""))
            if not q or not sl: continue
            ok,why=coherent_source(q)
            if not ok:
                bad.append({"row_index":i,"source_question":q,"source_bloom_level":sl,"reason":why})
                continue
            t,why2=extract_topic(q)
            if not t:
                bad.append({"row_index":i,"source_question":q,"source_bloom_level":sl,"reason":why2})
                continue
            if len(content_tokens(t))<4:
                bad.append({"row_index":i,"source_question":q,"source_bloom_level":sl,"reason":"thin_topic"})
                continue
            if residual_cognitive(t):
                bad.append({"row_index":i,"source_question":q,"source_bloom_level":sl,"reason":"cognitive_residue"})
                continue
            good.append({
                "source_id":"src_"+hashlib.sha256(q.encode()).hexdigest()[:16],
                "group_id":int(hashlib.sha256(q.encode()).hexdigest()[:8],16),
                "source_question":q,
                "source_bloom_level":sl,
                "topic":t,
                "topic_form":why2,
            })
    return good,bad

def split_sources(rows,seed):
    out={"train":[],"validation":[],"test":[]}
    for r in rows:
        u=int(hashlib.sha256(f"{seed}|split|{r['source_id']}".encode()).hexdigest()[:8],16)/0xFFFFFFFF
        out["train" if u<.70 else "validation" if u<.85 else "test"].append(r)
    return out

def candidates(rows,target):
    out=[]
    for s in rows:
        if s["source_bloom_level"]==target:
            out.append((s,s["source_question"],"identity_source_question",
                        {"reasons":[],"topic_content_recall":1.0,
                         "protected_spans":sorted(protected(s["source_question"])),
                         "missing_protected":[],"cross_level":False}))
            continue
        rewrite,template=make_rewrite(s["topic"],target)
        ok,info=validate(s["source_question"],s["topic"],target,rewrite)
        if ok:
            info["cross_level"]=True
            out.append((s,rewrite,template,info))
    return out

def choose_all(cands, seed):
    arr=cands[:]
    random.Random(seed).shuffle(arr)
    return arr


def make_row(split,target,s,rew,template,info):
    messages=[
      {"role":"system","content":(
        "Rewrite an academic question to the requested Bloom level. Preserve "
        "the source topic, technical concepts, quantities, named entities, "
        "and constraints. Do not answer the question. Output only one "
        "student-facing exam question."
      )},
      {"role":"user","content":f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"},
      {"role":"assistant","content":rew}
    ]
    text="<|im_start|>system\n"+messages[0]["content"]+"<|im_end|>\n<|im_start|>user\n"+messages[1]["content"]+"<|im_end|>\n<|im_start|>assistant\n"+rew+"<|im_end|>"
    return {
      "example_id":hashlib.sha256(f"{DATASET_VERSION}|{s['source_id']}|{target}|{rew}".encode()).hexdigest()[:16],
      "source_id":s["source_id"],"group_id":s["group_id"],"split":split,
      "source_question":s["source_question"],"source_bloom_level":s["source_bloom_level"],
      "target_bloom_level":target,"target_rewrite":rew,
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

    sources,rejected=read_source(src)
    splits=split_sources(sources,args.seed)
    all_rows={}; cand_stats={}; selected_stats={}

    for split in ("train","validation","test"):
        all_selected=[]
        cand_stats[split]={}
        selected_stats[split]={}
        for target in LEVELS:
            c=candidates(splits[split],target)
            cand_stats[split][target]={
                "total_candidates":len(c),
                "cross_level_candidates":sum(1 for x in c if x[3].get("cross_level")),
                "identity_candidates":sum(1 for x in c if not x[3].get("cross_level")),
            }
            chosen=choose(c,TARGET_PER_SPLIT[split],args.seed+len(split)+sum(map(ord,target)))
            if len(chosen)<TARGET_PER_SPLIT[split]:
                raise RuntimeError(
                    f"Insufficient candidates for {split}/{target}: "
                    f"need {TARGET_PER_SPLIT[split]}, have {len(chosen)}"
                )
            selected_stats[split][target]={
                "total":len(chosen),
                "cross_level":sum(1 for x in chosen if x[3].get("cross_level")),
                "identity":sum(1 for x in chosen if not x[3].get("cross_level")),
            }
            for s,rew,templ,info in chosen:
                all_selected.append(make_row(split,target,s,rew,templ,info))
        all_selected.sort(key=lambda r:(LEVELS.index(r["target_bloom_level"]),r["example_id"]))
        all_rows[split]=all_selected
        write_jsonl(out/f"{split}.jsonl",all_selected)

    # Ensure each target is exactly balanced and source sets are split-isolated.
    for split,rows in all_rows.items():
        counts=Counter(r["target_bloom_level"] for r in rows)
        if any(counts[t]!=TARGET_PER_SPLIT[split] for t in LEVELS):
            raise RuntimeError(f"Target imbalance in {split}: {counts}")

    sets={s:{r["source_id"] for r in rows} for s,rows in all_rows.items()}
    leakage={
        "train_validation":len(sets["train"]&sets["validation"]),
        "train_test":len(sets["train"]&sets["test"]),
        "validation_test":len(sets["validation"]&sets["test"]),
    }
    if any(leakage.values()): raise RuntimeError(f"Source leakage: {leakage}")

    source_hash=hashlib.sha256(src.read_bytes()).hexdigest()
    stats={
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":source_hash,
      "eligible_source_count":len(sources),"rejected_source_count":len(rejected),
      "rejected_sources_by_reason":dict(Counter(x["reason"] for x in rejected)),
      "source_split_sizes":{s:len(v) for s,v in splits.items()},
      "candidate_statistics":cand_stats,"selected_statistics":selected_stats,
      "counts":{s:{
        "total":len(rows),
        "by_target":dict(Counter(r["target_bloom_level"] for r in rows)),
        "cross_level_rows":sum(1 for r in rows if r["source_bloom_level"]!=r["target_bloom_level"]),
        "identity_rows":sum(1 for r in rows if r["source_bloom_level"]==r["target_bloom_level"])
      } for s,rows in all_rows.items()},
      "source_leakage_check":leakage,
      "notes":[
        "Synthetic supervision; not human gold.",
        "Built directly from the authoritative Figshare corpus.",
        "No LLM generation or LLM judging is used.",
        "Cross-level rows use deterministic conservative task frames.",
        "Rows with leftover cognitive instructions, multiple tasks, missing context, or unsupported Create framing are rejected.",
        "Same-level source questions may be retained as identity supervision.",
      ]
    }
    manifest={
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":source_hash,
      "counts":{s:len(v) for s,v in all_rows.items()},
      "target_count_per_level":TARGET_PER_SPLIT,
      "source_leakage_check":leakage
    }
    (out/"dataset_statistics.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    write_jsonl(out/"rejected_sources.jsonl",rejected)
    (out/"README.md").write_text(
      "# bloom_rewrite_synth_v6\n\n"
      "Conservative deterministic Bloom-target rewrite supervision built directly "
      "from data/figshare_bloom_v1.csv. No LLM is used to generate or judge rows.\n\n"
      "Cross-level transformations are emitted only when the source provides a "
      "coherent topic and the target task can be framed without introducing "
      "subject-matter facts, numbers, criteria, examples, or context. Same-level "
      "source questions may be included as identity supervision.\n",
      encoding="utf-8"
    )

    print("ELIGIBLE SOURCES:",len(sources))
    print("SOURCE SPLITS:",{s:len(v) for s,v in splits.items()})
    print("CANDIDATES:",json.dumps(cand_stats,indent=2))
    print("SELECTED:",json.dumps(selected_stats,indent=2))
    print("ROWS:",{s:len(v) for s,v in all_rows.items()})
    print("TOTAL:",sum(len(v) for v in all_rows.values()))
    print("LEAKAGE:",leakage)
    print("OUTPUT:",out)

if __name__=="__main__":
    main()
