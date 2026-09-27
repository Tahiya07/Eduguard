#!/usr/bin/env python3
"""Final conservative Bloom rewrite dataset builder.

Quality-first rules:
- Use only coherent one-task source questions from the raw Figshare CSV.
- Detect the source's actual task form from its leading operation.
- Preserve the source topic verbatim whenever possible.
- Cross-level rows come only from explicit safe whole-question patterns.
- Unsupported transformations are omitted.
- No LLM generation or judging.
"""

from __future__ import annotations
import argparse,csv,hashlib,json,random,re
from collections import Counter
from pathlib import Path

LEVELS=["Remember","Understand","Apply","Analyze","Evaluate","Create"]
MAP={
 "knowledge":"Remember","remembering":"Remember","recall":"Remember",
 "comprehension":"Understand","understanding":"Understand",
 "application":"Apply","applying":"Apply","analysis":"Analyze",
 "analysing":"Analyze","analyzing":"Analyze","evaluation":"Evaluate",
 "evaluating":"Evaluate","synthesis":"Create","creating":"Create"
}
DATASET_VERSION="bloom_rewrite_synth_v18"
POLICY_VERSION="bloom_target_policy_v18_explicit_safe"
SOURCE_FILE="data/figshare_bloom_v1.csv"
SEED=42
IDENTITY_CAP=250

STOP=set("""a an the and or but if then than that this these those it its of in on
at to for from with by as is are was were be been being do does did can could may
might will would should must into over under after before during through about
against between among within without using used your our their you we one someone
somebody""".split())

OPS=set("""define identify name list state recite label recognize recognise explain
describe summarize summarise interpret classify illustrate discuss retell apply use
calculate compute determine find solve implement demonstrate modify estimate measure
analyze analyse compare contrast differentiate distinguish examine evaluate assess
appraise judge justify critique criticize criticise defend recommend design develop
construct formulate propose create devise produce build compose write draw sketch
select choose suggest predict outline support relate rank""".split())

NUM_RE=re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")
TECH_RE=re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]*(?:\+\+|#|\d)[A-Za-z0-9_.-]*(?![A-Za-z0-9])")
FILE_RE=re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+\.(?:c|cc|cpp|h|hpp|py|java|js|ts|tsx|jsx|html|css|sql|json|csv|xml|md)|0x[0-9A-Fa-f]+)(?![A-Za-z0-9])")
ACRO_RE=re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")
PLACEHOLDER=re.compile(r"\.{2,}|\[[^\]]*\.\.\.[^\]]*\]|<[^>]*\.\.\.[^>]*>")

FORM_PATTERNS=[
 ("Evaluate",re.compile(r"^(?:briefly\s+|critically\s+|carefully\s+)?(?:evaluate|assess|appraise|judge|justify|critique|criticize|criticise|defend|recommend)\b",re.I)),
 ("Analyze",re.compile(r"^(?:briefly\s+)?(?:analyze|analyse|examine|compare|contrast|differentiate|distinguish)\b",re.I)),
 ("Apply",re.compile(r"^(?:calculate|compute|determine|find|solve|apply|use|implement|demonstrate|modify|estimate|measure)\b",re.I)),
 ("Create",re.compile(r"^(?:design|develop|construct|formulate|propose|create|devise|produce|build|compose|write|draw|sketch)\b",re.I)),
 ("Remember",re.compile(r"^(?:define|identify|name|list|state|recite|label|recognize|recognise)\b",re.I)),
 ("Understand",re.compile(r"^(?:explain|describe|summarize|summarise|interpret|classify|illustrate|discuss|retell)\b",re.I)),
]

SIMPLE_U=re.compile(r"^(?:explain|describe|discuss|illustrate|interpret)\s+(.+?)\s*\.?$",re.I)
SIMPLE_R=re.compile(r"^(?:define|identify|name|list|state|recite|label|recognize|recognise)\s+(.+?)\s*\.?$",re.I)
WHY_U=re.compile(r"^(?:explain|describe)\s+why\s+(.+?)\s*\.?$",re.I)
HOW_U=re.compile(r"^(?:explain|describe)\s+how\s+(.+?)\s*\.?$",re.I)
REL_U=re.compile(r"^(?:explain|describe|discuss)\s+(?:the\s+)?(?:relationship|relationships|interaction|interactions)\s+between\s+(.+?)\s*\.?$",re.I)
DIFF_U=re.compile(r"^(?:explain|describe|discuss)\s+(?:the\s+)?differences?\s+between\s+(.+?)\s*\.?$",re.I)
COMPARE_A=re.compile(r"^(?:compare|contrast)\s+(.+?)\s*\.?$",re.I)
DIFF_A=re.compile(r"^(?:differentiate|distinguish)\s+between\s+(.+?)\s*\.?$",re.I)
METHOD_APPLY=re.compile(
 r"^(?:explain|describe)\s+(?:the\s+)?(.+?\b(?:method|procedure|algorithm|formula|equation|technique)\b)"
 r"\s+(?:can be|could be|is|are)\s+(?:used|applied|implemented)\s+to\s+(.+?)\s*\.?$",re.I
)
HOW_METHOD_APPLY=re.compile(
 r"^how\s+(?:can|could|would|should)\s+(.+?\b(?:method|procedure|algorithm|formula|equation|technique)\b)"
 r"\s+(?:be\s+)?(?:used|applied|implemented)\s+to\s+(.+?)\s*\.?$",re.I
)

BAD_RESIDUE=re.compile(
 r"(?:\b(?:provide|provided|give|giving|support|justify|show|include|including|"
 r"example|examples|answer|answers|elaborate|discuss|explain|describe|evaluate|"
 r"assess|calculate|compute|determine|solve|draw|sketch|write|create|design|"
 r"develop|construct|propose|list|name|state)\b.*\b(?:each|answer|examples?|"
 r"working|calculation|response|discussion|justification)\b)|"
 r"(?:\bwith\s+(?:one|two|three|four|five|six|an?|the)\s+(?:relevant\s+)?examples?\b)|"
 r"(?:\bfor each\b)",re.I
)

def norm(x): return re.sub(r"\s+"," ",(x or "").replace("\u00a0"," ")).strip()
def canon(x):
 x=str(x or "").strip()
 return x if x in LEVELS else MAP.get(x.lower())
def content(x):
 return {t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*",(x or "").lower())
         if len(t)>2 and t not in STOP and t not in OPS}
def protected(x):
 out={m.group(0).lower() for m in NUM_RE.finditer(x or "")}
 out.update(m.group(0).lower() for m in TECH_RE.finditer(x or ""))
 out.update(m.group(0).lower() for m in FILE_RE.finditer(x or ""))
 out.update(m.group(1).lower() for m in ACRO_RE.finditer(x or ""))
 return out

def detect_level(q):
 for level,p in FORM_PATTERNS:
  if p.search(q): return level
 if re.match(r"^\s*what\s+(?:is|are|was|were)\b",q,re.I): return "Remember"
 if re.match(r"^\s*why\b",q,re.I) or re.match(r"^\s*how\b",q,re.I): return "Understand"
 return "Unknown"

def extract_topic(q):
 q=norm(q).strip(' "').rstrip("?.").strip()
 m=COMPARE_A.match(q)
 if m: return f"the similarities and differences between {m.group(1).strip()}", "compare"
 m=DIFF_A.match(q)
 if m: return f"the differences between {m.group(1).strip()}", "differentiate"
 m=REL_U.match(q)
 if m: return f"the relationship between {m.group(1).strip()}", "relationship"
 m=DIFF_U.match(q)
 if m: return f"the differences between {m.group(1).strip()}", "difference"
 m=WHY_U.match(q)
 if m: return f"why {m.group(1).strip()}", "why"
 m=HOW_U.match(q)
 if m: return f"how {m.group(1).strip()}", "how"
 m=HOW_METHOD_APPLY.match(q)
 if m: return f"{m.group(1).strip()} to {m.group(2).strip()}", "method_apply"
 m=METHOD_APPLY.match(q)
 if m: return f"{m.group(1).strip()} to {m.group(2).strip()}", "method_apply"
 m=SIMPLE_U.match(q)
 if m: return m.group(1).strip(), "simple"
 m=SIMPLE_R.match(q)
 if m: return m.group(1).strip(), "simple"
 for _,p in FORM_PATTERNS:
  m=p.match(q)
  if m: return q[m.end():].strip(" .;:,"), "simple"
 return None,""

def source_ok(q):
 q=norm(q).strip(' "')
 if not (25<=len(q)<=850): return False,"length"
 if PLACEHOLDER.search(q): return False,"placeholder"
 if any(x in q.lower() for x in (
  "above data","above graph","above diagram","following code","following segment",
  "following passage","given below","this article","the article","this magazine article"
 )): return False,"missing_context"
 if q.count("?")>1: return False,"multiple_questions"
 level=detect_level(q)
 if level=="Unknown": return False,"unknown_operation"
 # Only one explicit terminal item. Multiple sentence prompts are too hard to
 # normalize safely and are a major source of v3 failures.
 if re.search(r"(?<=[.!?])\s+[A-Z]",q.rstrip(".")):
  return False,"multiple_sentences"
 topic,form=extract_topic(q)
 if not topic or len(content(topic))<3: return False,"topic_parse"
 if BAD_RESIDUE.search(topic): return False,"instruction_residue"
 if re.search(r"\b(?:and|or)\s+(?:give|provide|support|justify|explain|describe|evaluate|assess|calculate|compute|determine|solve|draw|sketch|write|create|design|develop|construct|propose|list|name|state)\b",topic,re.I):
  return False,"multiple_actions"
 return True,""

def supported_cross(source,src_level,topic,target):
 low=(source+" "+topic).lower()

 if target=="Remember" and src_level=="Understand":
  # Only simple explanatory noun/topic questions.
  return SIMPLE_U.match(source) is not None and not re.match(r"(?i)^(why|how)\b",topic)

 if target=="Understand" and src_level=="Remember":
  return SIMPLE_R.match(source) is not None

 if target=="Analyze" and src_level=="Understand":
  return bool(WHY_U.match(source) or HOW_U.match(source) or REL_U.match(source) or DIFF_U.match(source))

 if target=="Evaluate" and src_level=="Understand":
  # Explicit evaluative dimensions only.
  m=SIMPLE_U.match(source)
  if not m: return False
  t=m.group(1).lower()
  return any(x in t for x in (
   "advantages and disadvantages","advantages","disadvantages","effectiveness",
   "suitability","appropriateness","appropriate","strengths","limitations"
  ))

 if target=="Apply" and src_level=="Understand":
  return bool(METHOD_APPLY.match(source) or HOW_METHOD_APPLY.match(source))

 return False

def transform(source,topic,src_level,target):
 if target=="Remember" and src_level=="Understand":
  return f"State {topic}.","understand_to_remember"
 if target=="Understand" and src_level=="Remember":
  return f"Explain {topic}.","remember_to_understand"
 if target=="Analyze" and src_level=="Understand":
  if WHY_U.match(source): return f"Analyze {topic}.","why_to_analyze"
  if HOW_U.match(source): return f"Analyze {topic}.","how_to_analyze"
  if REL_U.match(source): return f"Analyze {topic}.","relationship_to_analyze"
  if DIFF_U.match(source): return f"Analyze {topic}.","difference_to_analyze"
 if target=="Evaluate" and src_level=="Understand":
  return f"Evaluate {topic} and justify your judgment.","explicit_dimension_to_evaluate"
 if target=="Apply" and src_level=="Understand":
  if METHOD_APPLY.match(source) or HOW_METHOD_APPLY.match(source):
   return f"Apply {topic}.","method_to_apply"
 return None,None

def validate(source,topic,target,rewrite):
 if protected(source)-protected(rewrite):
  return False,["protected_span_loss"]
 if not re.search({
  "Remember":r"^(state|identify|list)\b",
  "Understand":r"^explain\b",
  "Apply":r"^apply\b",
  "Analyze":r"^analyze\b",
  "Evaluate":r"^evaluate\b",
  "Create":r"^(design|develop|construct|formulate|create)\b"
 }[target],rewrite,re.I):
  return False,["target_operation"]
 # The actual topic content must survive.
 src=content(topic); out=content(rewrite)
 if len(src & out)/max(1,len(src))<0.90:
  return False,["topic_content_loss"]
 return True,[]

def make_row(split,target,s,rewrite,template,validation):
 system=("Rewrite an academic question to the requested Bloom level. Preserve the "
         "source content, technical concepts, quantities, named entities, and "
         "constraints. Do not invent subject matter. Output only one student-facing "
         "exam question.")
 user=f"Original question:\n{s['source_question']}\n\nTarget Bloom level:\n{target}"
 msgs=[{"role":"system","content":system},{"role":"user","content":user},{"role":"assistant","content":rewrite}]
 packed=("<|im_start|>system\n"+system+"<|im_end|>\n"
         "<|im_start|>user\n"+user+"<|im_end|>\n"
         "<|im_start|>assistant\n"+rewrite+"<|im_end|>")
 synthetic=rewrite!=s["source_question"]
 return {
  "example_id":hashlib.sha256(f"{DATASET_VERSION}|{s['source_id']}|{target}|{rewrite}".encode()).hexdigest()[:16],
  "source_id":s["source_id"],"group_id":s["group_id"],"split":split,
  "source_question":s["source_question"],"source_bloom_level":s["source_bloom_level"],
  "source_detected_level":s["source_detected_level"],"source_task_form":s["source_task_form"],
  "target_bloom_level":target,"target_rewrite":rewrite,
  "transformation_type":f"{s['source_bloom_level']}->{target}",
  "synthetic_or_original":"synthetic" if synthetic else "original_identity",
  "synthetic":synthetic,"dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,
  "quality_status":"pass","construction_method":"deterministic_explicit_safe_patterns",
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
   ok,why=source_ok(q)
   if not ok:
    rejected.append({"row_index":idx,"source_question":q,"source_bloom_level":label,"reason":why})
    continue
   detected=detect_level(q)
   topic,form=extract_topic(q)
   sid="src_"+hashlib.sha256(q.encode()).hexdigest()[:16]
   if sid in seen: continue
   seen.add(sid)
   sources.append({
    "source_id":sid,"group_id":int(hashlib.sha256(sid.encode()).hexdigest()[:8],16),
    "source_question":q,"source_bloom_level":label,
    "source_detected_level":detected,"source_task_form":form,"topic":topic
   })

 splits={"train":[],"validation":[],"test":[]}
 for s in sources:
  u=int(hashlib.sha256(f"{args.seed}|split|{s['source_id']}".encode()).hexdigest()[:8],16)/0xFFFFFFFF
  splits["train" if u<.70 else "validation" if u<.85 else "test"].append(s)

 all_rows={}; count_info={}
 for split in ("train","validation","test"):
  by_target={t:[] for t in LEVELS}
  for target in LEVELS:
   # Identity examples: use source's detected operation, not blindly the CSV label.
   identity=[s for s in splits[split] if s["source_detected_level"]==target]
   rng=random.Random(args.seed+len(split)+sum(map(ord,target)))
   rng.shuffle(identity)
   for s in identity[:args.identity_cap]:
    by_target[target].append(make_row(
     split,target,s,s["source_question"],"identity_source_question",
     {"reasons":[],"topic_content_recall":1.0,"protected_spans":sorted(protected(s["source_question"])),
      "missing_protected":[],"identity":True}
    ))

   # Explicit cross-level transformations.
   for s in splits[split]:
    if s["source_detected_level"]==target: continue
    if not supported_cross(s["source_question"],s["source_detected_level"],s["topic"],target):
     continue
    rewrite,template=transform(s["source_question"],s["topic"],s["source_detected_level"],target)
    ok,reasons=validate(s["source_question"],s["topic"],target,rewrite)
    if not ok: continue
    by_target[target].append(make_row(
     split,target,s,rewrite,template,
     {"reasons":[],"topic_content_recall":1.0,
      "protected_spans":sorted(protected(s["source_question"])),
      "missing_protected":[],"identity":False}
    ))

  rows=[r for t in LEVELS for r in by_target[t]]
  all_rows[split]=rows
  write_jsonl(out/f"{split}.jsonl",rows)
  count_info[split]={
   "total":len(rows),
   "by_target":{t:len(by_target[t]) for t in LEVELS},
   "cross_level_rows":sum(r["source_detected_level"]!=r["target_bloom_level"] for r in rows),
   "identity_rows":sum(r["source_detected_level"]==r["target_bloom_level"] for r in rows),
   "by_source_target":dict(Counter(f"{r['source_detected_level']}->{r['target_bloom_level']}" for r in rows))
  }

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
  "counts":count_info,"identity_cap":args.identity_cap,
  "source_leakage_check":leakage,
  "notes":[
   "Synthetic supervision; not human gold.",
   "Built directly from data/figshare_bloom_v1.csv.",
   "Source task form is detected independently of the CSV source label.",
   "Same-task rows retain the original question.",
   "Cross-level rows use only explicit, whole-question safe patterns.",
   "Unsupported transformations are excluded rather than invented.",
   "No LLM generation or LLM judging is used."
  ]
 }
 manifest={"dataset_version":DATASET_VERSION,"policy_version":POLICY_VERSION,"seed":args.seed,
           "source_figshare":SOURCE_FILE,"source_figshare_sha256":source_hash,
           "counts":{s:len(v) for s,v in all_rows.items()},
           "source_leakage_check":leakage}
 (out/"dataset_statistics.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
 (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
 write_jsonl(out/"rejected_sources.jsonl",rejected)
 (out/"README.md").write_text(
  "# bloom_rewrite_synth_v18\n\n"
  "Quality-first deterministic Bloom-target rewrite supervision built directly "
  "from data/figshare_bloom_v1.csv. Same-task questions are retained; cross-level "
  "rows use explicit safe patterns only. No LLM is used. Unsupported transformations "
  "are omitted rather than invented.\\n",encoding="utf-8")
 print("ELIGIBLE SOURCES:",len(sources))
 print("SOURCE SPLITS:",{s:len(v) for s,v in splits.items()})
 print("COUNTS:",json.dumps(count_info,indent=2))
 print("REJECTED:",summary["rejected_sources_by_reason"])
 print("LEAKAGE:",leakage)
 print("OUTPUT:",out)

if __name__=="__main__":
 main()
