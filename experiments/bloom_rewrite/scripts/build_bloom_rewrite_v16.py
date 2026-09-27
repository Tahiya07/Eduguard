#!/usr/bin/env python3
"""High-fidelity Bloom-target rewrite supervision.

Every cross-level row is produced by a narrow whole-question pattern. The
builder never invents a scenario, criterion, artifact, example, number, or
subject-matter fact. Unsupported source/target pairs are omitted.
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
DATASET_VERSION="bloom_rewrite_synth_v16"
POLICY_VERSION="bloom_target_policy_v16_explicit_patterns"
SOURCE_FILE="data/figshare_bloom_v1.csv"
SEED=42

STOP=set("""a an the and or but if then than that this these those it its of in on
at to for from with by as is are was were be been being do does did can could may
might will would should must into over under after before during through about
against between among within without using used your our their you we someone somebody
one""".split())

NUM_RE=re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE=re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])")
FILE_RE=re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])")
ACRONYM_RE=re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")
PLACEHOLDER_RE=re.compile(r"(?:\.{2,}|\[[^\]]*\.{2,}[^\]]*\]|<[^>]*\.{2,}[^>]*>)")

MISSING=("above data","above graph","above diagram","following code","following segment",
"following passage","given below","this article","the article","this magazine article",
"the figure above","the table above")

# Exact source command families.
REMEMBER_V=re.compile(r"^\s*(?:define|identify|name|list|state|recite|label|recognize|recognise)\b",re.I)
UNDERSTAND_V=re.compile(r"^\s*(?:explain|describe|summarize|summarise|interpret|classify|illustrate|discuss|retell)\b",re.I)
APPLY_V=re.compile(r"^\s*(?:calculate|compute|determine|find|solve|apply|use|implement|demonstrate|modify|estimate|measure)\b",re.I)
ANALYZE_V=re.compile(r"^\s*(?:analyze|analyse|examine|compare|contrast|differentiate|distinguish)\b",re.I)
EVAL_V=re.compile(r"^\s*(?:evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|recommend)\b",re.I)
CREATE_V=re.compile(r"^\s*(?:design|develop|construct|formulate|propose|create|devise|produce|build|compose|write|draw|sketch)\b",re.I)
WHAT_V=re.compile(r"^\s*what\s+(?:is|are|was|were)\b",re.I)
WHY_V=re.compile(r"^\s*why\b",re.I)
HOW_V=re.compile(r"^\s*how\b",re.I)

def norm(x): return re.sub(r"\s+"," ",(x or "").replace("\u00a0"," ")).strip()
def canon(x):
    x=str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())
def content(x):
    return {t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*",(x or "").lower())
            if len(t)>2 and t not in STOP}
def protected(x):
    out={m.group(0).lower() for m in NUM_RE.finditer(x or "")}
    out.update(m.group(0).lower() for m in TECH_RE.finditer(x or ""))
    out.update(m.group(0).lower() for m in FILE_RE.finditer(x or ""))
    out.update(m.group(1).lower() for m in ACRONYM_RE.finditer(x or ""))
    return out

def detected_level(q):
    if REMEMBER_V.match(q) or WHAT_V.match(q): return "Remember"
    if APPLY_V.match(q): return "Apply"
    if CREATE_V.match(q): return "Create"
    if ANALYZE_V.match(q): return "Analyze"
    if EVAL_V.match(q): return "Evaluate"
    if UNDERSTAND_V.match(q) or WHY_V.match(q) or HOW_V.match(q): return "Understand"
    return "Unknown"

def strip_source_verb(q):
    q=norm(q).strip(' "').rstrip("?.").strip()
    # Special relation forms.
    m=re.match(r"^\s*(?:differentiate|distinguish)\s+between\s+(.+)$",q,re.I)
    if m: return "the differences between "+m.group(1).strip()
    m=re.match(r"^\s*(?:compare|contrast)\s+(.+)$",q,re.I)
    if m: return "the similarities and differences between "+m.group(1).strip()
    m=re.match(r"^\s*how\s+(?:does|do|did)\s+(.+?)\s+differ\s+from\s+(.+)$",q,re.I)
    if m: return "the difference between "+m.group(1).strip()+" and "+m.group(2).strip()
    m=re.match(r"^\s*(?:what\s+(?:is|are|was|were))\s+(.+)$",q,re.I)
    if m: return m.group(1).strip()
    patterns=[
      r"^\s*(?:briefly\s+|critically\s+|carefully\s+|concisely\s+)?(?:evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|recommend)\s+",
      r"^\s*(?:briefly\s+)?(?:analyze|analyse|examine|compare|contrast|differentiate|distinguish)\s+",
      r"^\s*(?:calculate|compute|determine|find|solve|apply|use|implement|demonstrate|modify|estimate|measure)\s+",
      r"^\s*(?:design|develop|construct|formulate|propose|create|devise|produce|build|compose|write|draw|sketch)\s+",
      r"^\s*(?:explain|describe|summarize|summarise|interpret|classify|illustrate|discuss|retell)\s+",
      r"^\s*(?:define|identify|name|list|state|recite|label|recognize|recognise)\s+",
      r"^\s*(?:how\s+(?:do|does|did|can|could|would|should|will))\s+(?:you|we|someone|somebody|one)\s+",
      r"^\s*(?:why)\s+",
    ]
    for p in patterns:
        n=re.sub(p,"",q,count=1,flags=re.I)
        if n!=q:
            return norm(n).strip(" .;:")
    return None

def normalize_topic(q):
    q=norm(q).strip(' "')
    # Remove diagram framing before the content clause.
    q=re.sub(r"^\s*(?:illustrate|draw|sketch)\s+(?:by\s+using|using|with\s+the\s+aid\s+of)\s+(?:an?\s+)?(?:appropriate\s+)?(?:diagram|diagrams|drawing|drawings)\s*[:,]?\s*","",q,flags=re.I)
    topic=strip_source_verb(q)
    if not topic: return None
    topic=re.sub(r"\s+(?:show\s+your\s+working(?:\s+and\s+calculation)?|justify\s+your\s+answer|support\s+(?:your\s+answer|your\s+views?)|elaborate(?:\s+your\s+answer)?)\s*$","",topic,flags=re.I)
    topic=norm(topic).strip(" .;:,")
    return topic or None

def source_ok(q):
    q=norm(q).strip(' "')
    if not (30<=len(q)<=900): return False,"length"
    if PLACEHOLDER_RE.search(q): return False,"placeholder"
    if any(x in q.lower() for x in MISSING): return False,"missing_context"
    if q.count("?")>1: return False,"multiple_questions"
    # Reject clear alternatives/multi-task source items.
    if re.search(r"\b(?:or|and)\s+(?:take|write|draw|sketch|calculate|compute|determine|"
                 r"explain|describe|discuss|justify|evaluate|assess|design|develop|"
                 r"construct|propose|create|list|name|state)\b",q,re.I):
        return False,"multiple_actions"
    if len(content(q))<5: return False,"thin_content"
    topic=normalize_topic(q)
    if not topic or len(content(topic))<3: return False,"topic_parse"
    return True,""

def eval_cue(topic):
    low=topic.lower()
    phrases=("advantages and disadvantages","advantages","disadvantages","effectiveness",
             "suitability","suitable","appropriate","best","better","worse","recommend",
             "recommendation","opinion","agree","alternative","alternatives","choice",
             "choices","justify","defend")
    return any(p in low for p in phrases)

def apply_cue(topic):
    low=topic.lower()
    if re.match(r"(?i)^(why|how)\b",topic.strip()):
        return bool(re.search(r"(?i)\b(?:method|procedure|algorithm|formula|equation|technique)\b",topic)
                    and re.search(r"(?i)\b(?:used|applied|implemented|encode|calculate|solve|determine|find|modify)\b",topic))
    return any(re.search(r"(?i)\b"+re.escape(p)+r"\b",low) for p in
               ("method","procedure","algorithm","formula","equation","technique","process")) or detected_level(topic)=="Apply"

def analyze_cue(topic):
    low=topic.lower()
    return any(p in low for p in (
        "difference","differences","similarity","similarities","relationship","relationships",
        "interaction","interactions","between","among","compare","contrast","differentiate",
        "distinguish","components","component","parts","structure","patterns","pattern",
        "causes","cause","effects","effect","why "))

def artifact_kind(source,topic):
    low=(source+" "+topic).lower()
    if re.search(r"\b(?:a|an|the)\s+(?:diagram|drawing|sketch|flow\s*chart|flowchart)\b",low):
        return "diagram"
    if re.search(r"\b(?:a|an|the)\s+(?:c\s+program|program|source\s+code|code\s+snippet)\b",low):
        return "program"
    if re.search(r"\b(?:a|an|the)\s+(?:presentation|storyboard)\b",low):
        return "presentation"
    if re.search(r"\b(?:a|an|the)\s+(?:strategy|plan|proposal)\b",low):
        return "strategy"
    if re.search(r"\b(?:a|an|the)\s+(?:model|framework|prototype)\b",low):
        return "model"
    if re.search(r"\b(?:a|an|the)\s+hypothesis\b",low):
        return "hypothesis"
    if re.search(r"\b(?:a|an|the)\s+(?:story|play|letter|dialogue)\b",low):
        return "creative"
    return None

def supported(source,src_form,topic,target):
    if target==src_form: return True
    if target=="Remember":
        # Safe only for conceptual/factual source topics, never for calculation,
        # novel creation, or causal/how clauses.
        if src_form in {"Apply","Create","Evaluate"}:
            return bool(re.search(r"(?i)\b(?:difference|differences|similarity|similarities)\b",topic))
        return not re.match(r"(?i)^(how|why|whether|if)\b",topic)
    if target=="Understand":
        if src_form=="Apply" and re.match(r"(?i)^(?:calculate|compute|determine|find|solve)\b",source):
            return False
        if src_form=="Create":
            low=source.lower()
            return not any(x in low for x in (
                "new title","new titles","new song","another ending","sequel",
                "diary","compose a","write a new","create a new"
            ))
        return True
    if target=="Apply":
        return apply_cue(topic)
    if target=="Analyze":
        return analyze_cue(topic)
    if target=="Evaluate":
        return eval_cue(topic)
    if target=="Create":
        return artifact_kind(source,topic) is not None
    return False

def transform(source,topic,src_form,target):
    if target=="Remember":
        m=re.match(r"(?i)^the difference between (.+?) and (.+)$",topic)
        if m: return f"State the difference between {m.group(1)} and {m.group(2)}.","remember_difference"
        if src_form=="Remember": return source,"identity"
        if re.search(r"(?i)\b(?:advantages?|disadvantages?|types?|factors?|steps?|phases?|stages?|methods?)\b",topic):
            return f"List {topic}.","remember_list"
        return f"Identify {topic}.","remember_identify"

    if target=="Understand":
        if src_form=="Understand": return source,"identity"
        m=re.match(r"(?i)^the differences between (.+)$",topic)
        if m: return f"Explain the differences between {m.group(1)}.","understand_differences"
        m=re.match(r"(?i)^the similarities and differences between (.+)$",topic)
        if m: return f"Explain the similarities and differences between {m.group(1)}.","understand_comparison"
        return f"Explain {topic}.","understand_explain"

    if target=="Apply":
        if src_form=="Apply": return source,"identity"
        m=re.search(r"(?i)^(.+?)\s+can\s+be\s+used\s+to\s+(.+)$",topic)
        if m and re.search(r"(?i)\b(?:method|procedure|algorithm|formula|equation|technique)\b",m.group(1)):
            return f"Apply {m.group(1)} to {m.group(2)}.","apply_method_to_task"
        m=re.search(r"(?i)^(.+?)\s+(?:is|are)\s+used\s+to\s+(.+)$",topic)
        if m and re.search(r"(?i)\b(?:method|procedure|algorithm|formula|equation|technique)\b",m.group(1)):
            return f"Apply {m.group(1)} to {m.group(2)}.","apply_method_to_task"
        return f"Apply {topic}.","apply_direct"

    if target=="Analyze":
        if src_form=="Analyze": return source,"identity"
        return f"Analyze {topic}.","analyze_direct"

    if target=="Evaluate":
        if src_form=="Evaluate": return source,"identity"
        return f"Evaluate {topic} and justify your judgment.","evaluate_judgment"

    if target=="Create":
        if src_form=="Create": return source,"identity"
        kind=artifact_kind(source,topic)
        if kind=="diagram":
            return f"Design a diagram illustrating {topic}.","create_diagram"
        return f"Develop {topic}.","create_artifact"

    return "", ""

def validate(source,topic,src_form,target,rewrite):
    if not rewrite: return False,["empty"]
    starts={
      "Remember":r"^(state|identify|list)\b","Understand":r"^explain\b",
      "Apply":r"^apply\b","Analyze":r"^analyze\b",
      "Evaluate":r"^evaluate\b","Create":r"^(design|develop|construct|formulate|create)\b"
    }
    if not re.search(starts[target],rewrite,re.I):
        return False,["target_operation"]
    if protected(source)-protected(rewrite):
        return False,["protected_span_loss"]
    if len(content(topic)&content(rewrite))/max(1,len(content(topic)))<0.90:
        return False,["topic_content_loss"]
    return True,[]

def row(split,target,s,rewrite,template,validation):
    system=("Rewrite an academic question to the requested Bloom level. Preserve "
            "the source content, technical concepts, quantities, named entities, "
            "and constraints. Do not invent subject matter. Output only one "
            "student-facing exam question.")
    user=f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"
    msgs=[{"role":"system","content":system},{"role":"user","content":user},{"role":"assistant","content":rewrite}]
    packed=("<|im_start|>system\n"+system+"<|im_end|>\n"
            "<|im_start|>user\n"+user+"<|im_end|>\n"
            "<|im_start|>assistant\n"+rewrite+"<|im_end|>")
    return {
      "example_id":hashlib.sha256(f"{DATASET_VERSION}|{s['source_id']}|{target}|{rewrite}".encode()).hexdigest()[:16],
      "source_id":s["source_id"],"group_id":s["group_id"],"split":split,
      "source_question":s["source_question"],"source_bloom_level":s["source_bloom_level"],
      "source_detected_level":s["source_detected_level"],
      "source_task_form":s["source_task_form"],
      "target_bloom_level":target,"target_rewrite":rewrite,
      "transformation_type":f"{s['source_bloom_level']}->{target}",
      "synthetic_or_original":"synthetic" if rewrite!=s["source_question"] else "original_identity",
      "synthetic":rewrite!=s["source_question"],
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,
      "quality_status":"pass","construction_method":"deterministic_explicit_pattern",
      "construction_template":template,"source_file":SOURCE_FILE,
      "generator_inputs":["source_question","target_bloom_level"],
      "validation":validation,"messages":msgs,"text":packed
    }

def write_jsonl(path,rows):
    with path.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default=SOURCE_FILE)
    ap.add_argument("--output-dir",default=f"data/bloom_rewrite_versions/{DATASET_VERSION}")
    ap.add_argument("--seed",type=int,default=SEED)
    ap.add_argument("--identity-cap",type=int,default=100,
                    help="Maximum same-detected-target rows per target/split.")
    args=ap.parse_args()

    src=Path(args.source); out=Path(args.output_dir)
    if not src.exists(): raise SystemExit(f"Source not found: {src}")
    if out.exists(): 
        for p in out.iterdir():
            if p.is_file(): p.unlink()
    out.mkdir(parents=True,exist_ok=True)

    sources=[]; rejected=[]; seen=set()
    with src.open("r",encoding="utf-8-sig",newline="") as f:
        for idx,raw in enumerate(csv.DictReader(f)):
            q=norm(raw.get("question","")).strip(' "')
            label=canon(raw.get("bloom_level",""))
            if not q or not label: continue
            ok,why=source_eligible(q)
            if not ok:
                rejected.append({"row_index":idx,"source_question":q,"source_bloom_level":label,"reason":why})
                continue
            form=detected_level(q)
            topic=normalize_topic(q)
            sid="src_"+hashlib.sha256(q.encode()).hexdigest()[:16]
            if sid in seen: continue
            seen.add(sid)
            sources.append({
              "source_id":sid,"group_id":int(hashlib.sha256(sid.encode()).hexdigest()[:8],16),
              "source_question":q,"source_bloom_level":label,
              "source_detected_level":form,"source_task_form":form,
              "topic":topic
            })

    splits={"train":[],"validation":[],"test":[]}
    for s in sources:
        u=int(hashlib.sha256(f"{args.seed}|split|{s['source_id']}".encode()).hexdigest()[:8],16)/0xFFFFFFFF
        splits["train" if u<.70 else "validation" if u<.85 else "test"].append(s)

    all_rows={}; summary={}
    for split in ("train","validation","test"):
        by_target={t:[] for t in LEVELS}
        for target in LEVELS:
            identity=[]; cross=[]
            for s in splits[split]:
                if s["source_detected_level"]==target:
                    identity.append(s)
                    continue
                if not supported(s["source_question"],s["source_detected_level"],s["topic"],target):
                    continue
                rewrite,template=transform(s["source_question"],s["topic"],s["source_task_form"],target)
                ok,reasons=validate(s["source_question"],s["topic"],s["source_task_form"],target,rewrite)
                if ok:
                    cross.append((s,rewrite,template,{"reasons":[],"identity":False}))
            rng=random.Random(args.seed+sum(map(ord,target))+len(split))
            rng.shuffle(identity); rng.shuffle(cross)
            chosen=cross+[(s,s["source_question"],"identity_source_question",
                           {"reasons":[],"identity":True}) for s in identity[:args.identity_cap]]
            for s,rw,tp,vi in chosen:
                by_target[target].append(row(split,target,s,rw,tp,vi))
        rows=[x for t in LEVELS for x in by_target[t]]
        all_rows[split]=rows
        write_jsonl(out/f"{split}.jsonl",rows)
        summary[split]={
          "total":len(rows),
          "by_target":{t:len(by_target[t]) for t in LEVELS},
          "cross_level":sum(r["source_detected_level"]!=r["target_bloom_level"] for r in rows),
          "identity":sum(r["source_detected_level"]==r["target_bloom_level"] for r in rows),
          "by_source_target":dict(Counter(f"{r['source_detected_level']}->{r['target_bloom_level']}" for r in rows))
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
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":sh,
      "eligible_source_count":len(sources),"rejected_source_count":len(rejected),
      "rejected_sources_by_reason":dict(Counter(r["reason"] for r in rejected)),
      "source_split_sizes":{s:len(v) for s,v in splits.items()},
      "counts":summary,"source_leakage_check":leakage,
      "identity_cap":args.identity_cap,
      "notes":[
        "Synthetic supervision; not human gold.",
        "Built directly from data/figshare_bloom_v1.csv.",
        "Cross-level rows use explicit whole-question transformation patterns.",
        "Unsupported transformations are excluded rather than invented.",
        "Same-detected-target source questions are retained as identity supervision, capped per target/split."
      ]
    }
    manifest={"dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
              "source_figshare":SOURCE_FILE,"source_figshare_sha256":sh,
              "counts":{s:len(v) for s,v in all_rows.items()},"source_leakage_check":leakage}
    (out/"dataset_statistics.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    write_jsonl(out/"rejected_sources.jsonl",rejected)
    (out/"README.md").write_text(
      "# bloom_rewrite_synth_v16\n\n"
      "High-fidelity deterministic Bloom-target supervision built directly from "
      "data/figshare_bloom_v1.csv. Cross-level rows are generated only by explicit "
      "whole-question transformation patterns. No LLM generation or judging is used. "
      "Unsupported transformations are excluded.\\n",encoding="utf-8")
    print("ELIGIBLE SOURCES:",len(sources))
    print("SOURCE SPLITS:",{s:len(v) for s,v in splits.items()})
    print("COUNTS:",json.dumps(summary,indent=2))
    print("REJECTED:",stats["rejected_sources_by_reason"])
    print("LEAKAGE:",leakage)
    print("OUTPUT:",out)

if __name__=="__main__":
    main()
