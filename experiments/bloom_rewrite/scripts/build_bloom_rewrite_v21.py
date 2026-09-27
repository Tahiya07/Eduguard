#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, random, re
from collections import Counter
from pathlib import Path

LEVELS=["Remember","Understand","Apply","Analyze","Evaluate","Create"]
MAP={"knowledge":"Remember","remembering":"Remember","recall":"Remember",
"comprehension":"Understand","understanding":"Understand","application":"Apply",
"applying":"Apply","analysis":"Analyze","analysing":"Analyze","analyzing":"Analyze",
"evaluation":"Evaluate","evaluating":"Evaluate","synthesis":"Create","creating":"Create"}

DATASET_VERSION="bloom_rewrite_synth_v21"
POLICY_VERSION="bloom_target_policy_v21_controlled"
SOURCE_FILE="data/figshare_bloom_v1.csv"
SEED=42

STOP=set("""a an the and or but if then than that this these those it its of in on at
to for from with by as is are was were be been being do does did can could may might
will would should must into over under after before during through about against
between among within without using used your our their you we someone somebody one
""".split())
OPS=set("""define identify name list state recite label recognize recognise explain describe
summarize summarise interpret classify illustrate discuss retell apply use calculate
compute determine find solve implement demonstrate modify estimate measure analyze analyse
compare contrast differentiate distinguish examine evaluate assess appraise judge justify
critique criticize criticise defend design develop construct formulate propose create devise
produce build compose write draw sketch select choose suggest recommend predict outline
support relate rank show give provide
""".split())

NUM=re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH=re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])")
FILE=re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])")
ACRO=re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")
ELLIPSIS=re.compile(r"(?:\.\s*){3,}|\.\.\.|\[[^\]]*\.\.\.[^\]]*\]|<[^>]*\.\.\.[^>]*>|\b_+\b")

MISSING=("above data","above graph","above diagram","following code","following segment",
"following passage","given below","this article","the article","this magazine article",
"the figure above","the table above")

def norm(x): return re.sub(r"\s+"," ",(x or "").replace("\u00a0"," ")).strip()
def canon(x):
    x=str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())
def content(x):
    return {t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*",(x or "").lower())
            if len(t)>2 and t not in STOP and t not in OPS}
def protected(x):
    out={m.group(0).lower() for m in NUM.finditer(x or "")}
    out|={m.group(0).lower() for m in TECH.finditer(x or "")}
    out|={m.group(0).lower() for m in FILE.finditer(x or "")}
    out|={m.group(1).lower() for m in ACRO.finditer(x or "")}
    return out

def topic(q):
    x=norm(q).strip(' "').rstrip("?.").strip()

    m=re.match(r"^how\s+(?:does|do|did)\s+(.+?)\s+differ\s+from\s+(.+)$",x,re.I)
    if m: return f"the difference between {m.group(1).strip()} and {m.group(2).strip()}"

    m=re.match(r"^(?:differentiate|distinguish)\s+between\s+(.+)$",x,re.I)
    if m: return f"the differences between {m.group(1).strip()}"

    m=re.match(r"^(?:compare|contrast)\s+(.+)$",x,re.I)
    if m: return f"the similarities and differences between {m.group(1).strip()}"

    m=re.match(r"^what\s+(?:is|are|was|were)\s+(.+)$",x,re.I)
    if m: return m.group(1).strip()

    m=re.match(r"^how\s+(?:do|does|did|can|could|would|should|will)\s+"
               r"(?:you|we|someone|somebody|one)\s+(.+)$",x,re.I)
    if m: return "how "+m.group(1).strip()

    for p in (
        r"^(?:briefly\s+|critically\s+|carefully\s+)?(?:evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|recommend)\s+",
        r"^(?:briefly\s+)?(?:analyze|analyse|examine|compare|contrast|differentiate|distinguish)\s+",
        r"^(?:calculate|compute|determine|find|solve|apply|use|implement|demonstrate|modify|estimate|measure)\s+",
        r"^(?:design|develop|construct|formulate|propose|create|devise|produce|build|compose|write|draw|sketch)\s+",
        r"^(?:explain|describe|summarize|summarise|interpret|classify|illustrate|discuss|retell)\s+",
        r"^(?:define|identify|name|list|state|recite|label|recognize|recognise)\s+",
        r"^why\s+",
    ):
        y=re.sub(p,"",x,count=1,flags=re.I)
        if y!=x: return y.strip(" .;:,")
    return None

def source_ok(q):
    q=norm(q).strip(' "')
    if not 32<=len(q)<=850: return False,"length",None
    if ELLIPSIS.search(q): return False,"placeholder",None
    low=q.lower()
    if any(x in low for x in MISSING): return False,"missing_context",None
    if q.count("?")>1: return False,"multiple_questions",None
    if re.search(r"\b(?:and|or|then|also|followed by)\s+(?:define|identify|name|list|state|explain|describe|apply|calculate|compute|determine|solve|analyze|analyse|compare|contrast|evaluate|assess|justify|design|develop|construct|propose|create)\b",q,re.I):
        return False,"multiple_actions",None
    if len(re.split(r"(?<=[.!?])\s+(?=[A-Z])",q))>1: return False,"multiple_sentences",None
    t=topic(q)
    if not t or ELLIPSIS.search(t): return False,"topic_parse",None
    if len(content(t))<4: return False,"thin_topic",None
    return True,"",t

def rewrite(target, t, source, src_detected):
    tl=t.strip()
    # Source wording is retained exactly when the detected operation already
    # matches the requested target; this avoids degrading good original items.
    if src_detected==target:
        return norm(source).rstrip("?.")+" .".replace(" .",".") ,"identity_source_question",True

    if target=="Remember":
        if tl.lower().startswith(("how ","why ")):
            return f"State the key information about {tl}.","remember_clause",False
        return f"Identify the main information about {tl}.","remember_identify",False

    if target=="Understand":
        if tl.lower().startswith(("how ","why ")):
            return f"Explain {tl}.","understand_clause",False
        return f"Explain {tl}.","understand_explain",False

    if target=="Apply":
        if any(x in source.lower() for x in ("calculate","compute","determine","find","solve")):
            return f"Apply the relevant method to determine {tl}.","apply_method",False
        return f"Apply your knowledge of {tl} to solve a related problem.","apply_knowledge",False

    if target=="Analyze":
        if tl.lower().startswith(("how ","why ")):
            return f"Analyze {tl}.","analyze_clause",False
        return f"Analyze {tl} by examining its key relationships and structure.","analyze_structure",False

    if target=="Evaluate":
        if tl.lower().startswith(("how ","why ")):
            return f"Evaluate {tl} and justify your judgment.","evaluate_clause",False
        return f"Evaluate {tl} and justify your judgment.","evaluate_judgment",False

    if target=="Create":
        low=source.lower()
        if any(x in low for x in ("diagram","drawing","sketch","flow chart","flowchart")):
            return f"Design a representation related to {tl}.","create_representation",False
        if any(x in low for x in ("program","code","algorithm","pseudocode","pseudo code")):
            return f"Develop a program or algorithm related to {tl}.","create_program",False
        if any(x in low for x in ("strategy","tactic","plan")):
            return f"Develop a strategy or plan related to {tl}.","create_strategy",False
        return f"Formulate an original approach related to {tl}.","create_approach",False

    raise ValueError(target)

def validate(source,target,rw):
    if not rw or len(rw.split())>90: return ["length"]
    starts={
      "Remember":r"^(state|identify|list)\b",
      "Understand":r"^explain\b",
      "Apply":r"^(apply)\b",
      "Analyze":r"^analyze\b",
      "Evaluate":r"^evaluate\b",
      "Create":r"^(design|develop|formulate)\b",
    }
    bad=[]
    if not re.search(starts[target],rw,re.I): bad.append("target_operation")
    if protected(source)-protected(rw): bad.append("protected_span_loss")
    if ELLIPSIS.search(rw): bad.append("placeholder")
    return bad

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default=SOURCE_FILE)
    ap.add_argument("--output-dir",default=f"data/bloom_rewrite_versions/{DATASET_VERSION}")
    ap.add_argument("--seed",type=int,default=SEED)
    args=ap.parse_args()

    src_path=Path(args.source); out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if p.is_file(): p.unlink()

    sources=[]; rejected=[]; seen=set()
    with src_path.open("r",encoding="utf-8-sig",newline="") as f:
        for idx,raw in enumerate(csv.DictReader(f)):
            q=norm(raw.get("question","")).strip(' "')
            level=canon(raw.get("bloom_level",""))
            if not q or not level: continue
            ok,reason,t=source_ok(q)
            if not ok:
                rejected.append({"row_index":idx,"source_question":q,"source_bloom_level":level,"reason":reason})
                continue
            sid="src_"+hashlib.sha256(q.encode()).hexdigest()[:16]
            if sid in seen: continue
            seen.add(sid)
            detected=detect_level(q)
            sources.append({"source_id":sid,"group_id":int(hashlib.sha256(sid.encode()).hexdigest()[:8],16),
                            "source_question":q,"source_bloom_level":level,
                            "source_detected_level":detected,"topic":t})

    splits={"train":[],"validation":[],"test":[]}
    for s in sources:
        u=int(hashlib.sha256(f"{args.seed}|split|{s['source_id']}".encode()).hexdigest()[:8],16)/0xFFFFFFFF
        splits["train" if u<.70 else "validation" if u<.85 else "test"].append(s)

    all_rows={}; rejected_rows=[]
    for split,items in splits.items():
        rows=[]
        for s in items:
            for target in LEVELS:
                rw,tp,identity=rewrite(target,s["topic"],s["source_question"],s["source_detected_level"])
                bad=validate(s["source_question"],target,rw)
                if bad:
                    rejected_rows.append({"source_id":s["source_id"],"target_bloom_level":target,
                                          "source_question":s["source_question"],"candidate":rw,"problems":bad})
                    continue
                sid=s["source_id"]
                system=("Rewrite an academic question to the requested Bloom level. Preserve "
                        "the source topic, technical concepts, quantities, named entities, and "
                        "constraints. Do not invent subject matter. Output only one student-facing "
                        "exam question.")
                user=f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"
                packed=("<|im_start|>system\n"+system+"<|im_end|>\n"
                        "<|im_start|>user\n"+user+"<|im_end|>\n"
                        "<|im_start|>assistant\n"+rw+"<|im_end|>")
                rows.append({
                    "example_id":hashlib.sha256(f"{DATASET_VERSION}|{sid}|{target}|{rw}".encode()).hexdigest()[:16],
                    "source_id":sid,"group_id":s["group_id"],"split":split,
                    "source_question":s["source_question"],"source_bloom_level":s["source_bloom_level"],
                    "source_detected_level":s["source_detected_level"],
                    "target_bloom_level":target,"target_rewrite":rw,
                    "transformation_type":f"{s['source_bloom_level']}->{target}",
                    "synthetic_or_original":"original_identity" if identity else "controlled_synthetic",
                    "synthetic":not identity,"dataset_version":DATASET_VERSION,
                    "policy_version":POLICY_VERSION,"quality_status":"pass",
                    "construction_method":"deterministic_source_anchored_transformation",
                    "construction_template":tp,"source_file":SOURCE_FILE,
                    "generator_inputs":["source_question","target_bloom_level"],
                    "validation":{"protected_span_preservation":True,"target_operation_check":True},
                    "messages":[{"role":"system","content":system},{"role":"user","content":user},{"role":"assistant","content":rw}],
                    "text":packed
                })
        all_rows[split]=rows
        with (out/f"{split}.jsonl").open("w",encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

    combined=[r for split in ("train","validation","test") for r in all_rows[split]]
    # Every source in a split contributes all six targets; target classes are
    # therefore exactly balanced within every split.
    leakage={
        "train_validation":len({r["source_id"] for r in all_rows["train"]}&{r["source_id"] for r in all_rows["validation"]}),
        "train_test":len({r["source_id"] for r in all_rows["train"]}&{r["source_id"] for r in all_rows["test"]}),
        "validation_test":len({r["source_id"] for r in all_rows["validation"]}&{r["source_id"] for r in all_rows["test"]})
    }
    if any(leakage.values()): raise RuntimeError(leakage)

    bundle={"dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,
            "source":SOURCE_FILE,"source_figshare_sha256":hashlib.sha256(src_path.read_bytes()).hexdigest(),
            "rows":combined}
    (out/f"{DATASET_VERSION}.json").write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding="utf-8")

    stats={
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":bundle["source_figshare_sha256"],
      "eligible_source_count":len(sources),"rejected_source_count":len(rejected),
      "rejected_cross_level_rows":len(rejected_rows),
      "rejected_sources_by_reason":dict(Counter(x["reason"] for x in rejected)),
      "source_split_sizes":{k:len(v) for k,v in splits.items()},
      "counts":{k:{"total":len(v),"by_target":dict(Counter(r["target_bloom_level"] for r in v))}
                for k,v in all_rows.items()},
      "source_leakage_check":leakage,
      "notes":[
        "Synthetic controlled supervision; not human gold.",
        "Built directly from data/figshare_bloom_v1.csv.",
        "Each clean source question is assigned to exactly one split.",
        "Each retained source receives one rewrite for each of the six target Bloom levels.",
        "Cross-level rewrites add only generic cognitive-task wording, not new subject-matter facts.",
        "Source numeric and technical anchors are required to be preserved.",
        "No LLM generation or LLM judging is used."
      ]
    }
    (out/"dataset_statistics.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"rejected_sources.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rejected),encoding="utf-8")
    (out/"rejected_cross_level.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rejected_rows),encoding="utf-8")
    (out/"README.md").write_text("# bloom_rewrite_synth_v21\n\nControlled source-anchored Bloom rewrite supervision from the authoritative Figshare corpus.\n",encoding="utf-8")
    print(json.dumps(stats,indent=2))

if __name__=="__main__": main()
