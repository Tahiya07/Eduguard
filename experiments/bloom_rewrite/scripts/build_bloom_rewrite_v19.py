#!/usr/bin/env python3
"""Quality-first Bloom rewrite supervision.

v19:
- authoritative source: data/figshare_bloom_v1.csv
- clean one-task source questions only
- source task detected from actual wording
- same-task rows retain the source unchanged
- cross-level rows use only explicit safe patterns
- no invented subject matter
- no LLM generation/judging
"""

from __future__ import annotations
import argparse, csv, hashlib, json, random, re
from collections import Counter
from pathlib import Path

LEVELS=["Remember","Understand","Apply","Analyze","Evaluate","Create"]
MAP={"knowledge":"Remember","remembering":"Remember","recall":"Remember",
"comprehension":"Understand","understanding":"Understand","application":"Apply",
"applying":"Apply","analysis":"Analyze","analysing":"Analyze","analyzing":"Analyze",
"evaluation":"Evaluate","evaluating":"Evaluate","synthesis":"Create","creating":"Create"}

DATASET_VERSION="bloom_rewrite_synth_v19"
POLICY_VERSION="bloom_target_policy_v19_clean_single_task_safe_patterns"
SOURCE_FILE="data/figshare_bloom_v1.csv"
SEED=42
IDENTITY_CAP=220

STOP=set("""a an the and or but if then than that this these those it its of in
on at to for from with by as is are was were be been being do does did can could
may might will would should must into over under after before during through about
against between among within without using used your our their you we someone somebody
one""".split())

OPS=set("""define identify name list state recite label recognize recognise explain
describe summarize summarise interpret classify illustrate discuss retell apply use
calculate compute determine find solve implement demonstrate modify estimate measure
analyze analyse compare contrast differentiate distinguish examine evaluate assess
appraise judge justify critique criticize criticise defend design develop construct
formulate propose create devise produce build compose write draw sketch select choose
suggest recommend predict outline support relate rank""".split())

NUM_RE=re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE=re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])")
FILE_RE=re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])")
ACRO_RE=re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")
PLACEHOLDER=re.compile(r"\.{2,}|\[[^\]]*\.{2,}[^\]]*\]|<[^>]*\.{2,}[^>]*>")

MISSING=("above data","above graph","above diagram","following code","following segment",
"following passage","given below","this article","the article","this magazine article",
"the figure above","the table above")

RESIDUE_RE=re.compile(
 r"(?:\bwith\s+(?:an?\s+)?(?:appropriate|relevant|suitable)\s+"
 r"(?:example|examples|diagram|diagrams|drawing|drawings)\b)|"
 r"(?:\b(?:provide|give|support|justify|elaborate|show)\b)|"
 r"(?:\bfor\s+each\b)|"
 r"(?:\bin\s+(?:a|the)\s+table\b)|"
 r"(?:\busing\s+(?:an?\s+)?appropriate\s+(?:diagram|example)\b)",
 re.I
)

MULTI_RE=re.compile(
 r"\b(?:and|or|then|also|followed\s+by)\s+(?:"
 r"define|identify|name|list|state|recite|label|recognize|recognise|explain|describe|"
 r"summarize|summarise|interpret|classify|illustrate|discuss|retell|apply|use|calculate|"
 r"compute|determine|find|solve|implement|demonstrate|modify|estimate|measure|analyze|"
 r"analyse|compare|contrast|differentiate|distinguish|examine|evaluate|assess|appraise|"
 r"judge|justify|critique|criticize|criticise|defend|design|develop|construct|formulate|"
 r"propose|create|devise|produce|build|compose|write|draw|sketch|select|choose|suggest|"
 r"recommend|predict|recite|outline|support|relate|rank)\b",re.I
)

# -------- source operation detection --------
def norm(x):
    return re.sub(r"\s+"," ",(x or "").replace("\u00a0"," ")).strip()


def canon(x):
    x = str(x or "").strip()
    return x if x in LEVELS else MAP.get(x.lower())


def detect_level(q):
    x=q.strip()
    # "How do you <verb> ..." must use the cognitive verb inside the question.
    m=re.match(r"^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+(?:you|we|someone|somebody|one)\s+([A-Za-z]+)\b",x,re.I)
    if m:
        v=m.group(1).lower()
        if v in {"calculate","compute","determine","find","solve","apply","use","implement","demonstrate","modify","estimate","measure"}:
            return "Apply"
        if v in {"compare","contrast","differentiate","distinguish","analyze","analyse","examine"}:
            return "Analyze"
        if v in {"evaluate","assess","appraise","judge","justify","critique","criticize","criticise","defend","recommend"}:
            return "Evaluate"
        if v in {"design","develop","construct","formulate","propose","create","devise","produce","build","compose","write","draw","sketch"}:
            return "Create"
        return "Understand"

    if re.match(r"^\s*how\s+(?:does|do|did)\s+.+\s+differ\s+from\s+.+$",x,re.I):
        return "Analyze"
    if re.match(r"^\s*what\s+(?:is|are|was|were)\b",x,re.I):
        return "Remember"
    if re.match(r"^\s*why\b",x,re.I):
        return "Understand"

    for level,p in (
        ("Evaluate",re.compile(r"^\s*(?:briefly\s+|critically\s+|carefully\s+)?(?:evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|recommend)\b",re.I)),
        ("Analyze",re.compile(r"^\s*(?:briefly\s+)?(?:analyze|analyse|examine|compare|contrast|differentiate|distinguish)\b",re.I)),
        ("Apply",re.compile(r"^\s*(?:calculate|compute|determine|find|solve|apply|use|implement|demonstrate|modify|estimate|measure)\b",re.I)),
        ("Create",re.compile(r"^\s*(?:design|develop|construct|formulate|propose|create|devise|produce|build|compose|write|draw|sketch)\b",re.I)),
        ("Remember",re.compile(r"^\s*(?:define|identify|name|list|state|recite|label|recognize|recognise)\b",re.I)),
        ("Understand",re.compile(r"^\s*(?:explain|describe|summarize|summarise|interpret|classify|illustrate|discuss|retell)\b",re.I)),
    ):
        if p.search(x): return level
    return "Unknown"

# -------- source topic extraction --------
def topic(q):
    x=q.strip().strip('"').rstrip("?.").strip()

    m=re.match(r"^\s*how\s+(?:does|do|did)\s+(.+?)\s+differ\s+from\s+(.+?)\s*$",x,re.I)
    if m: return f"the difference between {m.group(1).strip()} and {m.group(2).strip()}"

    m=re.match(r"^\s*(?:differentiate|distinguish)\s+between\s+(.+?)\s*$",x,re.I)
    if m: return f"the differences between {m.group(1).strip()}"

    m=re.match(r"^\s*(?:compare|contrast)\s+(.+?)\s*$",x,re.I)
    if m: return f"the similarities and differences between {m.group(1).strip()}"

    m=re.match(r"^\s*what\s+(?:is|are|was|were)\s+(.+?)\s*$",x,re.I)
    if m: return m.group(1).strip()

    m=re.match(r"^\s*how\s+(?:do|does|did|can|could|would|should|will)\s+"
               r"(?:you|we|someone|somebody|one)\s+([A-Za-z]+)\s+(.+?)\s*$",x,re.I)
    if m: return m.group(2).strip()

    # Strip one leading command only.
    pats=(
        r"^\s*(?:briefly\s+|critically\s+|carefully\s+)?(?:evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|recommend)\s+",
        r"^\s*(?:briefly\s+)?(?:analyze|analyse|examine|compare|contrast|differentiate|distinguish)\s+",
        r"^\s*(?:calculate|compute|determine|find|solve|apply|use|implement|demonstrate|modify|estimate|measure)\s+",
        r"^\s*(?:design|develop|construct|formulate|propose|create|devise|produce|build|compose|write|draw|sketch)\s+",
        r"^\s*(?:explain|describe|summarize|summarise|interpret|classify|illustrate|discuss|retell)\s+",
        r"^\s*(?:define|identify|name|list|state|recite|label|recognize|recognise)\s+",
        r"^\s*why\s+",
    )
    for p in pats:
        y=re.sub(p,"",x,count=1,flags=re.I)
        if y!=x:
            return y.strip(" .;:,")
    return None

def source_ok(q):
    q=norm(q).strip(' "')
    if not (25<=len(q)<=850): return False,"length"
    if PLACEHOLDER.search(q): return False,"placeholder"
    low=q.lower()
    if any(x in low for x in MISSING): return False,"missing_context"
    if q.count("?")>1: return False,"multiple_questions"
    # Treat a second cognitive command, explicit answer-format instruction, or
    # alternative action as evidence that the source is not a clean one-task item.
    if RESIDUE_RE.search(q): return False,"instruction_residue"
    if MULTI_RE.search(q): return False,"multiple_actions"
    if len(re.split(r"(?<=[.!?])\s+(?=[A-Z])",q))>1: return False,"multiple_sentences"
    if len(content(q))<5: return False,"thin_content"
    t=topic(q)
    if not t or len(content(t))<3: return False,"topic_parse"
    if RESIDUE_RE.search(t) or MULTI_RE.search(t): return False,"topic_residue"
    return True,""

def content(x):
    return {t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*",(x or "").lower())
            if len(t)>2 and t not in STOP and t not in OPS}

def protected(x):
    out={m.group(0).lower() for m in NUM_RE.finditer(x or "")}
    out.update(m.group(0).lower() for m in TECH_RE.finditer(x or ""))
    out.update(m.group(0).lower() for m in FILE_RE.finditer(x or ""))
    out.update(m.group(1).lower() for m in ACRO_RE.finditer(x or ""))
    return out

# -------- exact safe cross-level transformations --------
SIMPLE_UNDERSTAND=re.compile(
 r"^\s*(?:explain|describe|discuss|interpret)\s+(.+?)\s*\.?$",re.I)
SIMPLE_REMEMBER=re.compile(
 r"^\s*(?:define|identify|name|list|state|recite|label|recognize|recognise)\s+(.+?)\s*\.?$",re.I)

def safe_cross(source,src_level,target):
    # Remember <- simple Understand
    if target=="Remember" and src_level=="Understand":
        m=SIMPLE_UNDERSTAND.match(source)
        if m:
            t=m.group(1).strip()
            if not re.match(r"(?i)^(?:why|how|whether|if)\b",t):
                return f"State {t}.","understand_to_remember"
        return None,None

    # Understand <- simple Remember
    if target=="Understand" and src_level=="Remember":
        m=SIMPLE_REMEMBER.match(source)
        if m:
            return f"Explain {m.group(1).strip()}.","remember_to_understand"
        return None,None

    # Analyze <- explicit explanatory relationship/causal/process forms.
    if target=="Analyze" and src_level=="Understand":
        m=re.match(r"^\s*(?:explain|describe)\s+why\s+(.+?)\s*\.?$",source,re.I)
        if m:
            return f"Analyze why {m.group(1).strip()}.","why_to_analyze"
        m=re.match(r"^\s*(?:explain|describe)\s+how\s+(.+?)\s*\.?$",source,re.I)
        if m:
            return f"Analyze how {m.group(1).strip()}.","how_to_analyze"
        m=re.match(
            r"^\s*(?:explain|describe|discuss)\s+(?:the\s+)?"
            r"(relationship|relationships|interaction|interactions)\s+between\s+(.+?)\s*\.?$",
            source,re.I
        )
        if m:
            return f"Analyze the {m.group(1).lower()} between {m.group(2).strip()}.","relationship_to_analyze"
        m=re.match(
            r"^\s*(?:explain|describe|discuss)\s+(?:the\s+)?differences?\s+between\s+(.+?)\s*\.?$",
            source,re.I
        )
        if m:
            return f"Analyze the differences between {m.group(1).strip()}.","difference_to_analyze"
        return None,None

    # Evaluate <- explicit evaluative dimensions only.
    if target=="Evaluate" and src_level=="Understand":
        m=SIMPLE_UNDERSTAND.match(source)
        if m:
            t=m.group(1).strip()
            if re.search(
                r"(?i)\b(?:advantages|disadvantages|effectiveness|suitability|"
                r"appropriateness|strengths|limitations)\b",t
            ):
                return f"Evaluate {t} and justify your judgment.","explicit_dimension_to_evaluate"
        return None,None

    # Apply <- exact method/procedure use form.
    if target=="Apply" and src_level=="Understand":
        m=re.match(
            r"^\s*(?:explain|describe)\s+the\s+(.+?\b(?:method|procedure|algorithm|formula|equation|technique)\b)"
            r"\s+(?:can be|could be|is|are)\s+(?:used|applied|implemented)\s+to\s+(.+?)\s*\.?$",
            source,re.I
        )
        if m:
            return f"Apply the {m.group(1).strip()} to {m.group(2).strip()}.","method_to_apply"
        return None,None

    return None,None

def validate(source,target,rewrite):
    if protected(source)-protected(rewrite):
        return False
    starts={
      "Remember":r"^(state|identify|list)\b",
      "Understand":r"^explain\b",
      "Apply":r"^apply\b",
      "Analyze":r"^analyze\b",
      "Evaluate":r"^evaluate\b",
      "Create":r"^(design|develop|construct|formulate|create)\b",
    }
    return bool(re.search(starts[target],rewrite,re.I))

def make_row(split,target,s,rewrite,template,identity):
    system=("Rewrite an academic question to the requested Bloom level. Preserve "
            "the source content, technical concepts, quantities, named entities, "
            "and constraints. Do not invent subject matter. Output only one "
            "student-facing exam question.")
    user=f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"
    msgs=[{"role":"system","content":system},{"role":"user","content":user},
          {"role":"assistant","content":rewrite}]
    packed=("<|im_start|>system\n"+system+"<|im_end|>\n"
            "<|im_start|>user\n"+user+"<|im_end|>\n"
            "<|im_start|>assistant\n"+rewrite+"<|im_end|>")
    return {
      "example_id":hashlib.sha256(
        f"{DATASET_VERSION}|{s['source_id']}|{target}|{rewrite}".encode()
      ).hexdigest()[:16],
      "source_id":s["source_id"],"group_id":s["group_id"],"split":split,
      "source_question":s["source_question"],
      "source_bloom_level":s["source_bloom_level"],
      "source_detected_level":s["source_detected_level"],
      "source_task_form":s["source_task_form"],
      "target_bloom_level":target,"target_rewrite":rewrite,
      "transformation_type":f"{s['source_bloom_level']}->{target}",
      "synthetic_or_original":"original_identity" if identity else "synthetic",
      "synthetic":not identity,
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,
      "quality_status":"pass",
      "construction_method":"deterministic_explicit_safe_pattern",
      "construction_template":template,
      "source_file":SOURCE_FILE,
      "generator_inputs":["source_question","target_bloom_level"],
      "validation":{
        "protected_spans":sorted(protected(s["source_question"])),
        "missing_protected":[],
        "identity":identity
      },
      "messages":msgs,"text":packed
    }

def write_jsonl(path,rows):
    with path.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

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
            label=canon(raw.get("bloom_level",""))
            if not q or not label: continue
            ok,reason=source_ok(q)
            if not ok:
                rejected.append({"row_index":idx,"source_question":q,
                                 "source_bloom_level":label,"reason":reason})
                continue
            detected=detect_level(q)
            t=topic(q)
            sid="src_"+hashlib.sha256(q.encode()).hexdigest()[:16]
            if sid in seen: continue
            seen.add(sid)
            sources.append({
              "source_id":sid,"group_id":int(hashlib.sha256(sid.encode()).hexdigest()[:8],16),
              "source_question":q,"source_bloom_level":label,
              "source_detected_level":detected,"source_task_form":detected,"topic":t
            })

    splits={"train":[],"validation":[],"test":[]}
    for s in sources:
        u=int(hashlib.sha256(f"{args.seed}|split|{s['source_id']}".encode()).hexdigest()[:8],16)/0xFFFFFFFF
        splits["train" if u<.70 else "validation" if u<.85 else "test"].append(s)

    all_rows={}; info={}
    for split in ("train","validation","test"):
        by_target={t:[] for t in LEVELS}

        # Identity: a clean source whose actual wording already matches target.
        for target in LEVELS:
            ident=[s for s in splits[split] if s["source_detected_level"]==target]
            rng=random.Random(args.seed+len(split)+sum(map(ord,target)))
            rng.shuffle(ident)
            for s in ident[:args.identity_cap]:
                by_target[target].append(
                    make_row(split,target,s,s["source_question"],
                             "identity_source_question",True)
                )

        # Cross-level: only the explicit patterns above.
        for target in LEVELS:
            for s in splits[split]:
                if s["source_detected_level"]==target:
                    continue
                rw,tp=safe_cross(s["source_question"],s["source_detected_level"],target)
                if not rw: continue
                if not validate(s["source_question"],target,rw): continue
                by_target[target].append(
                    make_row(split,target,s,rw,tp,False)
                )

        rows=[r for t in LEVELS for r in by_target[t]]
        all_rows[split]=rows
        write_jsonl(out/f"{split}.jsonl",rows)
        info[split]={
          "total":len(rows),
          "by_target":{t:len(by_target[t]) for t in LEVELS},
          "cross_level_rows":sum(not x["synthetic"] if False else x["synthetic"] for x in rows),
          "identity_rows":sum(x["synthetic"]==False for x in rows),
          "by_source_target":dict(Counter(
              f"{x['source_detected_level']}->{x['target_bloom_level']}" for x in rows
          ))
        }

    sets={s:{r["source_id"] for r in rows} for s,rows in all_rows.items()}
    leakage={
      "train_validation":len(sets["train"]&sets["validation"]),
      "train_test":len(sets["train"]&sets["test"]),
      "validation_test":len(sets["validation"]&sets["test"])
    }
    if any(leakage.values()): raise RuntimeError(f"Source leakage: {leakage}")

    sh=hashlib.sha256(src.read_bytes()).hexdigest()
    stats={
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":sh,
      "eligible_source_count":len(sources),"rejected_source_count":len(rejected),
      "rejected_sources_by_reason":dict(Counter(x["reason"] for x in rejected)),
      "source_split_sizes":{s:len(v) for s,v in splits.items()},
      "counts":info,"identity_cap":args.identity_cap,
      "source_leakage_check":leakage,
      "notes":[
        "Synthetic supervision; not human gold.",
        "Built directly from data/figshare_bloom_v1.csv.",
        "Source task form is detected from wording, independently of the CSV label.",
        "Clean same-task source questions are retained unchanged.",
        "Cross-level transformations are limited to explicit safe patterns.",
        "Unsupported transformations are excluded rather than invented.",
        "No LLM generation or LLM judging is used."
      ]
    }
    (out/"dataset_statistics.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"dataset_manifest.json").write_text(json.dumps({
      "dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
      "source_figshare":SOURCE_FILE,"source_figshare_sha256":sh,
      "counts":{s:len(v) for s,v in all_rows.items()},"source_leakage_check":leakage
    },indent=2,ensure_ascii=False),encoding="utf-8")
    write_jsonl(out/"rejected_sources.jsonl",rejected)
    (out/"README.md").write_text(
      "# bloom_rewrite_synth_v19\n\n"
      "Quality-first Bloom-target supervision built directly from the authoritative "
      "Figshare corpus. Only clean one-task source questions are retained; cross-level "
      "pairs are produced by explicit safe patterns. No LLM is used.\\n",encoding="utf-8")
    print("ELIGIBLE SOURCES:",len(sources))
    print("SOURCE SPLITS:",{s:len(v) for s,v in splits.items()})
    print("COUNTS:",json.dumps(info,indent=2))
    print("REJECTED:",stats["rejected_sources_by_reason"])
    print("LEAKAGE:",leakage)
    print("OUTPUT:",out)

if __name__=="__main__":
    main()
