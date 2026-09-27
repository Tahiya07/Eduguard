#!/usr/bin/env python3
"""Build a high-fidelity Bloom rewrite dataset directly from Figshare.

v14 is intentionally conservative. It never invents a scenario, criterion,
artifact, example, number, method, or subject-matter detail. Cross-level rows
are created only when the source task structure supports the requested target.
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

DATASET_VERSION="bloom_rewrite_synth_v14"
POLICY_VERSION="bloom_target_policy_v14_strict_source_supported"
SOURCE_FILE="data/figshare_bloom_v1.csv"
SEED=42

COGNITIVE=set("""define explain describe list name state identify recall recognize recognise
summarize summarise interpret classify illustrate apply use calculate compute determine
find solve implement demonstrate analyze analyse compare contrast differentiate distinguish
examine evaluate assess appraise critique judge justify defend design develop construct
formulate propose create devise produce build write draw sketch compose select choose
suggest recommend predict recite outline label specify modify estimate measure relate rank
support""".split())

STOP=set("""a an the and or but if then than that this these those it its of in on at
to for from with by as is are was were be been being do does did can could may might
will would should must into over under after before during through about against between
among within without using used your our their you we someone somebody one""".split())

NUM_RE=re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE=re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])")
FILE_RE=re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])")
ACRONYM_RE=re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")
PLACEHOLDER_RE=re.compile(r"(?:\.{2,}|\[[^\]]*\.{2,}[^\]]*\]|<[^>]*\.{2,}[^>]*>)")

MISSING_CONTEXT=("above data","above graph","above diagram","following code",
"following segment","following passage","given below","this article",
"the article","this magazine article","the figure above","the table above")

LEADING=re.compile(
r"""^\s*(?:briefly\s+|critically\s+|carefully\s+|concisely\s+)*
(?:please\s+)?
(?P<verb>
evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|
compare|contrast|differentiate|distinguish|analyze|analyse|examine|
calculate|compute|determine|find|solve|apply|use|implement|demonstrate|
design|develop|construct|formulate|propose|create|devise|produce|build|
write|draw|sketch|compose|explain|describe|summarize|summarise|interpret|classify|
illustrate|discuss|define|identify|name|list|state|recite|outline|label|
predict|select|choose|specify|modify|estimate|measure|relate|rank|support
)\b\s*""",re.I|re.X)

HOW=re.compile(
r"""^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+
(?:you|we|someone|somebody|one)\s+
(?P<verb>
define|explain|describe|compare|contrast|differentiate|distinguish|analyze|analyse|
examine|calculate|compute|determine|find|solve|apply|use|implement|demonstrate|
construct|design|develop|formulate|propose|create|devise|produce|build|write|draw|
sketch|identify|classify|interpret|select|choose|modify|estimate|measure|relate|rank
)\b\s+""",re.I|re.X)

WHAT=re.compile(r"^\s*what\s+(?:is|are|was|were)\s+(.+)$",re.I)
HOW_DIFFER=re.compile(r"^\s*how\s+(?:does|do|did)\s+(.+?)\s+differ\s+from\s+(.+?)\s*$",re.I)

FRAME_PREFIX=re.compile(
r"^(?:by\s+using|using|with\s+the\s+aid\s+of|with)\s+(?:an?\s+)?"
r"(?:appropriate\s+)?(?:diagram|diagrams|drawing|drawings|example|examples)\s*[:,]?\s*",
re.I
)

ANSWER_TAIL=re.compile(
r"""(?:\s+show\s+your\s+working(?:\s+and\s+calculation)?|
\s+justify\s+your\s+answer|
\s+support\s+(?:your\s+answer|your\s+views?)|
\s+elaborate(?:\s+your\s+answer)?)\s*\.?$""",re.I|re.X)

ALT_ACTION=re.compile(
r"\b(?:or|and)\s+(?:take|write|draw|sketch|compose|create|design|develop|"
r"construct|propose|calculate|compute|determine|explain|describe|discuss|"
r"identify|list|state|name|justify|evaluate|assess)\b",re.I)

RELATIONAL=("between","among","difference","differences","similarity","similarities",
"relationship","relationships","interaction","interactions","compare","contrast",
"differentiate","distinguish","components","component","parts","structure","pattern",
"patterns","cause","causes","effect","effects")

EVALUATIVE=("advantages and disadvantages","advantages","disadvantages","effectiveness",
"suitability","suitable","appropriate","best","better","worse","recommend",
"recommendation","judge","justify","defend","agree","opinion","alternative",
"alternatives","choice","choices","criteria","evidence","strength","strengths",
"limitation","limitations","prefer","preference")

APPLY_EXACT=("calculate","compute","determine","find","solve","apply","use","implement",
"demonstrate","method","methods","procedure","procedures","algorithm","formula",
"equation","modify","given data","scenario","case","problem")

ARTIFACT_NOUNS=("diagram","drawing","sketch","flow chart","flowchart","program",
"algorithm","presentation","storyboard","strategy","plan","proposal","model",
"framework","hypothesis","project","campaign","poster","prototype","story","play",
"letter","dialogue")

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

def contains_phrase(text, phrase):
    low=text.lower()
    return phrase in low if " " in phrase else re.search(r"(?<![A-Za-z])"+re.escape(phrase)+r"(?![A-Za-z])",low) is not None

def parse_source(q):
    q=norm(q).strip(' "').rstrip("?.").strip()

    m=HOW_DIFFER.match(q)
    if m:
        return f"the difference between {m.group(1)} and {m.group(2)}","how_differ"

    m=HOW.match(q)
    if m:
        return q[m.end():].strip(" .;:,").strip(),m.group("verb").lower()

    m=WHAT.match(q)
    if m:
        return m.group(1).strip(" .;:"),"what"

    m=LEADING.match(q)
    if m:
        rest=q[m.end():].strip(" .;:,")
        rest=FRAME_PREFIX.sub("",rest).strip(" .;:,")
        rest=ANSWER_TAIL.sub("",rest).strip(" .;:,")
        if ALT_ACTION.search(rest):
            return None,"multiple_actions"
        return rest,m.group("verb").lower()

    return None,"unrecognized"

def source_eligible(q):
    q=norm(q).strip(' "')
    if not (35<=len(q)<=900): return False,"length"
    if PLACEHOLDER_RE.search(q): return False,"placeholder"
    if any(x in q.lower() for x in MISSING_CONTEXT): return False,"missing_context"
    if q.count("?")>1: return False,"multiple_questions"
    # Keep only one coherent source item.
    parts=re.split(r"(?<=[.!?])\s+(?=[A-Z])",q)
    if len(parts)>1: return False,"multiple_sentences"
    if len(content(q))<5: return False,"thin_content"
    topic,form=parse_source(q)
    if not topic or len(content(topic))<3: return False,form
    if ALT_ACTION.search(topic): return False,"multiple_actions"
    return True,""

def supported(source,topic,form,target):
    low=(source+" "+topic).lower()

    if target=="Remember":
        if form in {"calculate","compute","determine","find","solve","apply","use",
                    "implement","demonstrate","design","develop","construct","formulate",
                    "propose","create","devise","produce","build","write","draw","sketch",
                    "compose","modify"}:
            return bool(re.search(r"(?i)\b(?:difference|differences|similarity|similarities)\b",topic))
        return not re.match(r"(?i)^(how|why|whether|if|when|where)\b",topic)

    if target=="Understand":
        if form in {"calculate","compute","determine","find","solve","draw","sketch",
                    "write","compose"}:
            return False
        if form in {"create","design","develop","construct","formulate","propose","devise",
                    "produce","build"}:
            bad=("new title","new titles","new song","another ending","sequel",
                 "diary","original story","novel response")
            return not any(x in low for x in bad)
        return True

    if target=="Apply":
        if re.match(r"(?i)^(why|how|whether|if)\b",topic):
            return False
        return any(contains_phrase(source, x) for x in APPLY_EXACT)

    if target=="Analyze":
        return any(contains_phrase(low,x) for x in RELATIONAL)

    if target=="Evaluate":
        return any(contains_phrase(low,x) for x in EVALUATIVE)

    if target=="Create":
        # Only create when an actual output artifact is present in the source
        # wording. Generic conceptual/algorithmic content is not enough.
        return any(re.search(r"\b(?:a|an|the)\s+"+re.escape(x)+r"\b",low)
                   or re.search(r"\b"+re.escape(x)+r"\b",low)
                   for x in ARTIFACT_NOUNS)

    return False

def transform(source,topic,form,target):
    if target=="Remember":
        m=re.match(r"(?i)^the difference between (.+?) and (.+)$",topic)
        if m: return f"State the difference between {m.group(1)} and {m.group(2)}.","remember_difference"
        if contains_phrase(topic,"differences") or contains_phrase(topic,"similarities"):
            return f"State {topic}.","remember_relation"
        if form=="list": return f"List {topic}.","remember_list"
        return f"Identify {topic}.","remember_identify"

    if target=="Understand":
        return f"Explain {topic}.","understand_explain"

    if target=="Apply":
        return f"Apply {topic}.","apply_direct"

    if target=="Analyze":
        return f"Analyze {topic}.","analyze_direct"

    if target=="Evaluate":
        return f"Evaluate {topic} and justify your judgment.","evaluate_judgment"

    if target=="Create":
        low=topic.lower()
        if re.search(r"\b(?:diagram|drawing|sketch|flow chart|flowchart)\b",low):
            return f"Design {topic}.","create_diagram"
        if re.search(r"\b(?:program|algorithm)\b",low):
            return f"Develop {topic}.","create_program"
        if re.search(r"\b(?:presentation|storyboard)\b",low):
            return f"Develop {topic}.","create_presentation"
        if re.search(r"\b(?:strategy|plan|proposal)\b",low):
            return f"Develop {topic}.","create_strategy"
        if re.search(r"\b(?:model|framework|system)\b",low):
            return f"Construct {topic}.","create_model"
        if re.search(r"\bhypothesis\b",low):
            return f"Formulate {topic}.","create_hypothesis"
        return f"Create {topic}.","create_artifact"

    raise ValueError(target)

def validate(source,topic,target,rewrite):
    reasons=[]
    low=rewrite.lower()

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
      "Remember":r"^(identify|list|state)\b",
      "Understand":r"^explain\b",
      "Apply":r"^apply\b",
      "Analyze":r"^analyze\b",
      "Evaluate":r"^evaluate\b",
      "Create":r"^(design|develop|construct|formulate|create)\b",
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
    system=("Rewrite an academic question to the requested Bloom level. Preserve the "
            "source content, technical concepts, quantities, named entities, and "
            "constraints. Do not invent subject matter. Output only one student-facing "
            "exam question.")
    user=f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"
    messages=[{"role":"system","content":system},
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
      "construction_method":"deterministic_strict_source_supported",
      "construction_template":template,"source_file":SOURCE_FILE,
      "generator_inputs":["source_question","target_bloom_level"],
      "validation":info,"messages":messages,"text":packed
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
            topic,form=parse_source(q)
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

    all_rows={}; statistics={}
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
                rewrite,template=transform(s["source_question"],s["topic"],s["form"],target)
                ok,info=validate(s["source_question"],s["topic"],target,rewrite)
                if ok:
                    info["identity"]=False
                    target_rows.append(make_row(split,target,s,rewrite,template,info))
            rows.extend(target_rows)
            per[target]={
              "total":len(target_rows),
              "cross_level":sum(r["source_bloom_level"]!=r["target_bloom_level"] for r in target_rows),
              "identity":sum(r["source_bloom_level"]==r["target_bloom_level"] for r in target_rows)
            }
        rows.sort(key=lambda r:(LEVELS.index(r["target_bloom_level"]),r["example_id"]))
        all_rows[split]=rows; statistics[split]=per
        write_jsonl(out/f"{split}.jsonl",rows)

    sets={s:{r["source_id"] for r in rows} for s,rows in all_rows.items()}
    leakage={"train_validation":len(sets["train"]&sets["validation"]),
             "train_test":len(sets["train"]&sets["test"]),
             "validation_test":len(sets["validation"]&sets["test"])}
    if any(leakage.values()): raise RuntimeError(f"Source leakage: {leakage}")

    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
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
      "candidate_statistics":statistics,
      "source_leakage_check":leakage,
      "notes":[
        "Synthetic supervision; not human gold.",
        "Built directly from data/figshare_bloom_v1.csv.",
        "No LLM generation or LLM judging is used.",
        "Cross-level transformations are allowed only when source task form supports the target operation.",
        "Unsupported transformations are omitted rather than invented.",
        "Same-level source questions are retained as identity supervision.",
      ]
    }
    (out/"dataset_statistics.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"dataset_manifest.json").write_text(json.dumps({
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":source_hash,
      "counts":{s:len(v) for s,v in all_rows.items()},
      "source_leakage_check":leakage
    },indent=2,ensure_ascii=False),encoding="utf-8")
    write_jsonl(out/"rejected_sources.jsonl",rejected)
    (out/"README.md").write_text(
      "# bloom_rewrite_synth_v14\n\n"
      "High-fidelity deterministic Bloom-target supervision built directly from "
      "data/figshare_bloom_v1.csv. Cross-level rows are emitted only when the "
      "source task structure supports the requested Bloom operation. No LLM is "
      "used to generate or judge rows; unsupported transformations are excluded.\\n",
      encoding="utf-8")
    print("ELIGIBLE SOURCES:",len(sources))
    print("SOURCE SPLITS:",{s:len(v) for s,v in splits.items()})
    print("COUNTS:",json.dumps(summary["counts"],indent=2))
    print("REJECTED:",summary["rejected_sources_by_reason"])
    print("LEAKAGE:",leakage)
    print("OUTPUT:",out)

if __name__=="__main__":
    main()
