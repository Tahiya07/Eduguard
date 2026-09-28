#!/usr/bin/env python3
"""Build corrected multitask SFT data: Bloom rewrite + QA + summarization."""
from __future__ import annotations
import hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/"data/rewrite_dataset"; BASE=ROOT/"data/multitask_bloom_rewrite"; OUT=ROOT/"data/multitask_corrected"
LEVELS={"Remember","Understand","Apply","Analyze","Evaluate","Create"}; SEED=42
def read(p):
 rows=[]
 for n,line in enumerate(p.read_text(encoding="utf-8").splitlines(),1):
  if line.strip():
   try: rows.append(json.loads(line))
   except json.JSONDecodeError as e: raise SystemExit(f"Invalid JSONL {p}:{n}: {e}") from e
 return rows
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8"); read(p)
def key(s): return hashlib.sha256(s.encode()).hexdigest()
def main():
 bt=read(B/"bloom_rewrite_train.jsonl"); bv=read(B/"bloom_rewrite_validation.jsonl"); bx=read(B/"bloom_rewrite_test.jsonl")
 t=read(BASE/"train.jsonl"); v=read(BASE/"validation.jsonl"); x=read(BASE/"test.jsonl")
 only=lambda rows,task:[r for r in rows if r.get("task")==task]
 qa_t,sm_t=only(t,"qa"),only(t,"summarization"); qa_v,sm_v=only(v,"qa"),only(v,"summarization"); qa_x,sm_x=only(x,"qa"),only(x,"summarization")
 g=defaultdict(list)
 for r in bt: g[r["source_id"]].append(r)
 groups=[rows for rows in g.values() if len(rows)==6 and {r["target_bloom_level"] for r in rows}==LEVELS]
 if not groups: raise SystemExit("No complete Bloom source groups.")
 desired=min(round((len(qa_t)+len(sm_t))*0.4/0.6),len(groups)*6); desired-=desired%6; ng=desired//6
 groups.sort(key=lambda z:key(z[0]["source_id"])); by=defaultdict(list)
 for gr in groups: by[gr[0]["source_bloom_level"]].append(gr)
 labels=sorted(by); alloc={z:int(ng*len(by[z])/len(groups)) for z in labels}
 while sum(alloc.values())<ng:
  z=max(labels,key=lambda a:len(by[a])-alloc[a]); alloc[z]+=1
 while sum(alloc.values())>ng:
  z=max((a for a in labels if alloc[a]>0),key=lambda a:alloc[a]); alloc[z]-=1
 selected=[gr for z in labels for gr in by[z][:alloc[z]]]; bloom_sel=[r for gr in selected for r in gr]
 rng=random.Random(SEED); rng.shuffle(bloom_sel)
 train=bloom_sel+qa_t+sm_t; validation=bv+qa_v+sm_v; test=bx+qa_x+sm_x
 rng.shuffle(train); rng.shuffle(validation); rng.shuffle(test)
 OUT.mkdir(parents=True,exist_ok=True); write(OUT/"train.jsonl",train); write(OUT/"validation.jsonl",validation); write(OUT/"test.jsonl",test)
 def summary(rows):
  c=Counter(r.get("task") for r in rows); n=len(rows); return {"total":n,"by_task":dict(c),"proportions":{k:round(v/n,6) for k,v in c.items()}}
 manifest={"dataset_version":"multitask_corrected_v5","seed":SEED,"base_model_target":"Qwen2.5-1.5B-Instruct",
 "requested_train_mix":{"bloom_rewrite":0.4,"qa":0.3,"summarization":0.3},
 "sources":{"bloom_rewrite":"data/rewrite_dataset/bloom_rewrite_{train,validation,test}.jsonl","qa_summarization":"data/multitask_bloom_rewrite/{train,validation,test}.jsonl"},
 "train":summary(train),"validation":summary(validation),"test":summary(test),
 "bloom_train_source_groups_selected":len(selected),"bloom_train_rows_selected":len(bloom_sel),
 "bloom_train_complete_groups":True,"old_bloom_rows_excluded":True,"qa_summarization_unchanged":True,"test_not_tuned":True}
 (OUT/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8"); print(json.dumps(manifest,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
