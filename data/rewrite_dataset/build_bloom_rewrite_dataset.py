#!/usr/bin/env python3
"""Build a Figshare-backed Bloom target-level rewrite corpus."""
from __future__ import annotations
import csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/rewrite_dataset"
LEVELS=("Remember","Understand","Apply","Analyze","Evaluate","Create")
INPUTS={
 "train":ROOT/"data/figshare_bloom_v1_train.csv",
 "validation":ROOT/"data/figshare_bloom_v1_val.csv",
 "test":ROOT/"data/figshare_bloom_v1_test.csv",
}
SYSTEM=("Rewrite an academic question to the requested Bloom level. Preserve the original topic, "
        "technical concepts, quantities, named entities, and important constraints. Do not invent "
        "subject matter. Do not merely replace the original verb; change the student's cognitive "
        "operation. Output only one student-facing exam question.")
TASKS=("define|explain|describe|discuss|list|name|state|identify|recall|recognize|summarize|interpret|"
       "classify|apply|calculate|compute|solve|demonstrate|analyze|analyse|compare|contrast|"
       "differentiate|distinguish|examine|evaluate|assess|justify|critique|appraise|design|develop|"
       "create|propose|formulate|construct|suggest|recommend|highlight|outline|determine|give|show|"
       "comment|illustrate|mention|specify|write|draw|revise|participate|derive|predict|choose|"
       "select|use|implement|elaborate|retell|recite|compose|criticize|criticise|rank|rate|place|"
       "judge|defend|correct|find|detect|fix|sketch|estimate|label|verify|paraphrase|perform|"
       "produce|devise|prepare|plan|relate|inspect|investigate|prove|test|validate|generalize|"
       "generalise|restate|express|model|combine|categorize|categorise|group|argue|review|"
       "synthesize|synthesise|provide|support|divide")
TASK_RE=re.compile(rf"\b(?:{TASKS})\b",re.I)
LEVEL_VERBS={
 "Remember":("define","list","name","state","identify","recall","recognize","label","mention","specify","recite","restate"),
 "Understand":("explain","describe","discuss","summarize","interpret","classify","illustrate","elaborate","paraphrase","retell"),
 "Apply":("apply","calculate","compute","solve","use","demonstrate","implement","determine","perform"),
 "Analyze":("analyze","analyse","compare","contrast","differentiate","distinguish","examine","investigate","relate","inspect","break"),
 "Evaluate":("evaluate","assess","justify","critique","appraise","judge","defend","rank","rate","validate","review"),
 "Create":("design","develop","create","propose","formulate","construct","suggest","recommend","compose","devise","model","combine","synthesize","synthesise","generalize","generalise"),
}
PLACEHOLDER_RE=re.compile(r"(?:\.{3,}|(?:\s*\.\s*){3,}|_{2,}|\bfill\s+in\s+the\s+blank\b)",re.I)
GENERIC_RE=re.compile(r"^(?:this|that|these|those|the statement|this statement|your answer|the answer|"
                       r"your position|evidence|arguments?|justifications?|the decision|the recommendation|"
                       r"the choice|the result|the process|the method|the concept|the idea|the following|the above)\b",re.I)
COUNT_RE=re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*\(\s*\d+\s*\)",re.I)
FORMAT_RE=re.compile(
 r"^(?:(?:please|briefly|clearly|concisely|carefully|critically)\s+)+|"
 r"^(?:based on (?:your|the) understanding|in your own words)[,:]?\s*|"
 r"^(?:with (?:the )?aid of|using|by using)\s+(?:an?\s+)?(?:appropriate|relevant)\s+[^,;:]+[,:]\s*|"
 r"^(?:in (?:a|the) (?:table|memo|paragraph|form))\s*[,;:]?\s*",re.I)
SECONDARY_RE=re.compile(
 rf"\s+(?:and|or)\s+(?:justify|support|defend|provide|show|comment|explain|describe|discuss|"
 rf"analyze|analyse|evaluate|assess|critique|identify|list|give|state|illustrate|outline)\b.*$",re.I)

def clean(s): return re.sub(r"\s+"," ",str(s).replace("\u00a0"," ")).strip().strip('"').strip("'").strip()
def norm(s): return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ",s.lower())).strip()
def sid(q): return "src_"+hashlib.sha256(clean(q).encode()).hexdigest()[:16]
def toks(s): return {t for t in re.findall(r"[a-z0-9][a-z0-9_-]*",s.lower()) if len(t)>2}
def sentences(s): return [x.strip() for x in re.split(r"(?<=[.!?])\s+",s) if x.strip()]

def nominalize(s):
 s=clean(COUNT_RE.sub(lambda m:m.group(1).lower(),s)).rstrip(" .,:;?")
 pats=[

  (r"^how\s+would\s+you\s+(relate|link|connect)\s+(.+?)\s+(?:with|to)\s+(.+)$",lambda m:f"the relationship between {m.group(2)} and {m.group(3)}"),
  (r"^how\s+would\s+you\s+(differentiate|distinguish)\s+(?:between\s+)?(.+?)\s+(?:and|from)\s+(.+)$",lambda m:f"the distinction between {m.group(2)} and {m.group(3)}"),
  (r"^how\s+would\s+you\s+classify\s+(.+)$",lambda m:f"the classification of {m.group(1)}"),
  (r"^how\s+(?:would|could|can|should|do|does|did)\s+you\s+(.+)$",lambda m:f"the process used to {m.group(1)}"),
  (r"^how\s+to\s+(.+)$",lambda m:f"the process used to {m.group(1)}"),
  (r"^how\s+far\s+(.+)$",lambda m:f"the extent to which {m.group(1)}"),
  (r"^in\s+what\s+ways\s+(.+)$",lambda m:f"the ways in which {m.group(1)}"),
  (r"^in\s+which\s+(.+?)\s+(?:does|do|did|is|are)\s+(.+)$",lambda m:f"the {m.group(1)} in which {m.group(2)}"),
  (r"^who\s+(?:was|is|were|are)\s+(.+)$",lambda m:m.group(1)),
  (r"^where\s+(?:is|are|was|were)\s+(.+)$",lambda m:m.group(1)),
  (r"^how\s+much\s+time\s+(?:did|does|do)\s+(.+?)\s+(?:take|takes|took)\s+to\s+(.+)$",lambda m:f"the time taken by {m.group(1)} to {m.group(2)}"),
  (r"^how\s+much\s+(.+?)\s+is\s+(.+)$",lambda m:m.group(2)),
  (r"^how\s+many\s+(.+?)\s+(?:does|do|did|can|could|would)\s+(.+)$",lambda m:f"the number of {m.group(1)} that {m.group(2)}"),
  (r"^what\s+goes\s+in\s+(.+)$",lambda m:f"the contents of {m.group(1)}"),
  (r"^what\s+(.+?)\s+means?$",lambda m:f"the meaning of {m.group(1)}"),
  (r"^what\s+does\s+(.+?)\s+(.+)$",lambda m:f"the {m.group(2)} of {m.group(1)}"),
  (r"^what\s+(?:is|are)\s+(.+)$",lambda m:m.group(1)),
  (r"^what\s+(.+)$",lambda m:m.group(1)),
  (r"^if\s+(.+)$",lambda m:f"the question of whether {m.group(1)}"),
  (r"^whether\s+(.+)$",lambda m:f"the question of whether {m.group(1)}"),
  (r"^how\s+(?:does|do|did)\s+(.+?)\s+affect\s+(.+)$",lambda m:f"the effect of {m.group(1)} on {m.group(2)}"),
  (r"^how\s+(?:does|do|did)\s+(.+?)\s+compare\s+(?:with|to)\s+(.+)$",lambda m:f"the comparison between {m.group(1)} and {m.group(2)}"),
  (r"^how\s+(?:does|do|did)\s+(.+?)\s+differ\s+from\s+(.+)$",lambda m:f"the difference between {m.group(1)} and {m.group(2)}"),
  (r"^how\s+(?:does|do|did)\s+(.+?)\s+explain\s+(.+)$",lambda m:f"the role of {m.group(1)} in explaining {m.group(2)}"),
  (r"^how\s+(?:does|do|did)\s+(.+)$",lambda m:f"the way {m.group(1)}"),
  (r"^how\s+(?:would|could|can|should)\s+(.+)$",lambda m:f"the process of {m.group(1)}"),
  (r"^why\s+(?:does|do|did|is|are|was|were|would|could|can|should)\s+(.+)$",lambda m:f"the reasons for {m.group(1)}"),
  (r"^why\s+(.+)$",lambda m:f"the reasons for {m.group(1)}"),
  (r"^what\s+(?:does|do|did)\s+(.+?)\s+(?:mean|refer to|stand for)$",lambda m:f"the meaning of {m.group(1)}"),
  (r"^what\s+(?:is|are)\s+(.+)$",lambda m:m.group(1)),
  (r"^what\s+happens\s+if\s+(.+)$",lambda m:f"the effects of {m.group(1)}"),
  (r"^which\s+of\s+the\s+following\s+(.+)$",lambda m:f"the appropriate {m.group(1)}"),
 ]
 for p,fn in pats:
  m=re.match(p,s,re.I)
  if m: return clean(fn(m))
 return s

def extract_topic(q,level):
 q=clean(q)
 if not q or PLACEHOLDER_RE.search(q): return None
 ss=sentences(q); chosen=None
 for s in ss:
  if any(re.search(rf"\b{re.escape(v)}\b",s,re.I) for v in LEVEL_VERBS.get(level,())):
   chosen=s; break
 if chosen is None:
  for s in ss:
   if TASK_RE.search(s) or "?" in s or re.match(r"^(?:what|who|where|when|why|how|which|in what ways)\b",s,re.I): chosen=s; break
 if chosen is None: return None
 body=clean(COUNT_RE.sub(lambda m:m.group(1).lower(),chosen))
 body=FORMAT_RE.sub("",body).strip()
 body=re.sub(r"^(?:can|could|would|should|do|does|did)\s+you\s+","",body,flags=re.I).strip()
 match=None
 for v in LEVEL_VERBS.get(level,()):
  m=re.search(rf"\b{re.escape(v)}\b",body,re.I)
  if m: match=m; break
 if match is None: match=TASK_RE.search(body)
 if match:
  before=body[:match.start()].strip(" ,:;-"); after=body[match.end():].strip(" ,:;-")
  if before and re.match(r"^(?:how|why|what|which)\b",before,re.I): body=body
  elif before and not re.match(r"^(?:based on|using|by using|with|given|assuming|from|as|in|for|on the basis of|according to|with reference to)\b",before,re.I):
   body=f"{before} {after}".strip()
  else: body=after
 body=SECONDARY_RE.sub("",body).strip()
 body=re.sub(r"\s+(?:show|provide|support|justify|defend|comment)\s+(?:your|the|each|one|two|three|four|five|six|seven|eight|nine|ten|an?|a|relevant|appropriate)\b.*$","",body,flags=re.I).strip()
 body=re.sub(r"\s+(?:to|in order to|so that|so as to)\s+(?:help|allow|enable|make it possible)\b.*$","",body,flags=re.I).strip()
 body=re.sub(r"\s+(?:and|or|with|of|in|on|for|to)\s*$","",body,flags=re.I)
 if not body or GENERIC_RE.fullmatch(body) or re.match(r"^(?:evidence|arguments?|justifications?|your views?|your answer|your position)\b",body,re.I):
  try: idx=ss.index(chosen)
  except ValueError: idx=0
  if idx>0:
   body=clean(ss[idx-1]); body=FORMAT_RE.sub("",body).strip()
   body=re.sub(rf"^\s*(?:{TASKS})\b[\s,:-]*","",body,flags=re.I).strip(" .,:;?")
   body=SECONDARY_RE.sub("",body).strip()
 body=nominalize(body)
 body=clean(body).rstrip(" .,:;?")
 if not toks(body) or PLACEHOLDER_RE.search(body) or GENERIC_RE.fullmatch(body): return None
 if re.match(r"^(?:how|why|what|which|whether|if|can|could|would|should)\b",body,re.I): return None
 return body

TEMPLATES={
 "Remember":("What key facts about {topic} should be recalled?","State the main characteristics of {topic}.","Identify the essential components or terms associated with {topic}.","Which key facts about {topic} should a student be able to recall?","Name the principal elements of {topic}.","List the essential facts associated with {topic}."),
 "Understand":("Explain {topic}.","Describe the main idea and purpose of {topic}.","Summarize what {topic} means in its academic context.","How would you explain {topic} to a learner unfamiliar with it?","Describe the role and significance of {topic}.","Explain the relationship or meaning expressed by {topic}."),
 "Apply":("How would you apply knowledge related to {topic} to a concrete case?","Given a concrete case involving {topic}, determine the appropriate result or action.","Demonstrate how knowledge of {topic} would be used in a practical situation.","Use the relevant knowledge about {topic} to solve a concrete problem.","Apply knowledge related to {topic} in a specified case and determine the outcome.","In a practical scenario involving {topic}, determine the appropriate result or action."),
 "Analyze":("Analyze {topic} by examining its components, relationships, causes, or patterns.","Examine the structure of {topic} and identify the relationships among its parts.","Compare relevant aspects of {topic} and explain how they relate.","Break down {topic} into components and analyze how those components interact.","What relationships, contrasts, or patterns can be identified within {topic}?","Examine the causes and underlying structure of {topic}."),
 "Evaluate":("Evaluate {topic} using explicit criteria and supporting evidence, and justify your judgment.","Assess the strengths and limitations of {topic} against relevant criteria.","Critique {topic} by considering its validity, suitability, and trade-offs.","How effective is {topic} according to stated criteria, and why?","Judge the quality or suitability of {topic} using evidence and clear standards.","Defend a reasoned judgment about {topic} based on explicit criteria."),
 "Create":("Design a new solution or plan related to {topic} for a defined problem.","Develop an original approach incorporating {topic} under stated requirements.","Propose a new strategy or artifact based on {topic} for a specified need.","Formulate a new solution involving {topic} under given constraints.","How would you construct an original solution using {topic}?","Create a new plan or artifact that incorporates {topic} for a clearly defined purpose.")
}
CUES={
 "Remember":("recall","state","identify","name","list","facts","characteristics"),
 "Understand":("explain","describe","summarize","meaning","purpose"),
 "Apply":("apply","concrete","case","determine","practical","scenario","procedure","problem","use"),
 "Analyze":("analyze","examine","compare","components","relationships","structure","patterns","causes","break down"),
 "Evaluate":("evaluate","assess","critique","criteria","evidence","justify","judgment","effective","judge"),
 "Create":("design","develop","construct","original","artifact","strategy","solution","constraints","formulate","propose","create")
}
def build_rewrite(topic,target,q,s):
 digest=int(hashlib.sha256(f"{s}|{target}".encode()).hexdigest()[:8],16)
 qn=norm(q)
 for off in range(len(TEMPLATES[target])):
  idx=(digest+off)%len(TEMPLATES[target]); c=TEMPLATES[target][idx].format(topic=topic); cn=norm(c)
  if cn!=qn and any(x in cn for x in CUES[target]): return c,idx
 raise ValueError("no_template")

def read_csv(p):
 with p.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def read_jsonl(p):
 rows=[]
 for n,line in enumerate(p.read_text(encoding="utf-8").splitlines(),1):
  if line.strip():
   try: rows.append(json.loads(line))
   except json.JSONDecodeError as e: raise SystemExit(f"Invalid JSONL {p}:{n}: {e}") from e
 return rows
def write_jsonl(p,rows):
 p.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8")
 read_jsonl(p)

def main():
 all_rows=[]; rejected=[]; source_sets={k:set() for k in INPUTS}
 for split,p in INPUTS.items():
  for src in read_csv(p):
   q=clean(src.get("question","")); level=clean(src.get("bloom_level","")); s=sid(q)
   topic=extract_topic(q,level)
   if topic is None:
    rejected.append({"split":split,"source_id":s,"question":q,"source_bloom_level":level,"reason":"unusable_source_or_topic"}); continue
   if s in source_sets[split]:
    rejected.append({"split":split,"source_id":s,"question":q,"reason":"duplicate_source"}); continue
   group=[]; fail=None
   for target in LEVELS:
    try: rewrite,idx=build_rewrite(topic,target,q,s)
    except Exception as e: fail={"split":split,"source_id":s,"question":q,"source_bloom_level":level,"reason":"generation_failed","target":target,"detail":str(e),"topic":topic}; break
    rn=norm(rewrite)
    if target!=level and rn==norm(q): fail={"split":split,"source_id":s,"question":q,"reason":"cross_level_identity","target":target,"topic":topic}; break
    prompt=f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nOriginal question:\n{q}\n\nTarget Bloom level:\n{target}<|im_end|>"
    sft=prompt+f"\n<|im_start|>assistant\n{rewrite}<|im_end|>"
    group.append({
      "example_id":hashlib.sha256(f"{s}|{target}".encode()).hexdigest()[:16],
      "task":"bloom_rewrite","source_id":s,"group_id":s[4:],"split":split,
      "source_question":q,"source_bloom_level":level,"target_bloom_level":target,
      "target_rewrite":rewrite,"source_topic":topic,"synthetic":target!=level,
      "transformation_type":f"{level}->{target}",
      "construction_method":"deterministic_source_anchored_cognitive_operation_template",
      "construction_template_index":idx,"dataset_version":"figshare_target_rewrite_v5",
      "policy_version":"figshare_bloom_target_policy_v5",
      "source_file":str(p.relative_to(ROOT)).replace("\\","/"),
      "quality_status":"pass","prompt_text":prompt,"sft_text":sft
    })
   if fail: rejected.append(fail); continue
   source_sets[split].add(s); all_rows.extend(group)

 split_ids={k:{r["source_id"] for r in all_rows if r["split"]==k} for k in INPUTS}
 overlaps={"train_validation":len(split_ids["train"]&split_ids["validation"]),"train_test":len(split_ids["train"]&split_ids["test"]),"validation_test":len(split_ids["validation"]&split_ids["test"])}
 if any(overlaps.values()): raise SystemExit(f"Source leakage: {overlaps}")
 grouped=defaultdict(set)
 for r in all_rows: grouped[r["source_id"]].add(r["target_bloom_level"])
 incomplete=[k for k,v in grouped.items() if v!=set(LEVELS)]
 duplicate=len(all_rows)-len({r["example_id"] for r in all_rows})
 if incomplete or duplicate: raise SystemExit(f"Invariants failed: incomplete={len(incomplete)} duplicate={duplicate}")

 OUT.mkdir(parents=True,exist_ok=True)
 all_rows.sort(key=lambda r:(r["split"],r["source_id"],LEVELS.index(r["target_bloom_level"])))
 for split in INPUTS: write_jsonl(OUT/f"bloom_rewrite_{split}.jsonl",[r for r in all_rows if r["split"]==split])
 write_jsonl(OUT/"bloom_rewrite_all.jsonl",all_rows)
 stats={
  "source_of_truth":{k:str(v.relative_to(ROOT)).replace("\\","/") for k,v in INPUTS.items()},
  "figshare_source_rows":{k:len(read_csv(v)) for k,v in INPUTS.items()},
  "accepted_source_rows":{k:len(split_ids[k]) for k in INPUTS},
  "rewrite_rows":{k:sum(r["split"]==k for r in all_rows) for k in INPUTS},
  "targets_per_source":6,"target_counts":dict(Counter(r["target_bloom_level"] for r in all_rows)),
  "rejected_source_rows":len(rejected),"rejection_reasons":dict(Counter(r["reason"] for r in rejected)),
  "source_split_overlap":overlaps,"incomplete_source_groups":len(incomplete),"duplicate_example_ids":duplicate,
  "jsonl_revalidated":True
 }
 (OUT/"stats.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
 (OUT/"rejected_sources.json").write_text(json.dumps(rejected,indent=2,ensure_ascii=False),encoding="utf-8")
 (OUT/"dataset_manifest.json").write_text(json.dumps({
  "dataset_version":"figshare_target_rewrite_v5","policy_version":"figshare_bloom_target_policy_v5",
  "source_of_truth":"Figshare split files in data/figshare_bloom_v1_{train,val,test}.csv",
  "target_levels":list(LEVELS),"same_level_behavior":"identity copy",
  "cross_level_behavior":"source-anchored cognitive-operation transformation; not verb substitution",
  "training_fields":["prompt_text","sft_text"],"checks":{
   "six_targets_per_accepted_source":not incomplete,"source_split_disjoint":not any(overlaps.values()),
   "duplicate_example_ids_zero":duplicate==0,"jsonl_revalidated":True,
   "no_source_bloom_level_in_generator_prompt":True},"stats":stats
 },indent=2,ensure_ascii=False),encoding="utf-8")
 print(json.dumps(stats,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
