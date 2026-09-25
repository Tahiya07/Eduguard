#!/usr/bin/env python
"""Build Mix-A multitask v4 from corrected Bloom supervision.

QA/summarization are reused from the locked multitask corpus. The 8,321-row
baseline test is copied byte-for-byte and is never regenerated or tuned on.
"""
from __future__ import annotations
import argparse,hashlib,json,random,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

SCRIPT_DIR=Path(__file__).resolve().parent; EXP=SCRIPT_DIR.parent; ROOT=EXP.parents[1]
if str(EXP) not in sys.path: sys.path.insert(0,str(EXP))
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from prompts import build_sft_text,build_prompt_only_text,build_generation_prompt

LOCKED=ROOT/"data/multitask_bloom_rewrite"
V4=ROOT/"data/bloom_rewrite_versions/bloom_rewrite_synth_v4_1"
OUT=ROOT/"data/multitask_bloom_rewrite_v4_1"

def read(p):
    with open(p,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def write(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
def hfile(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()
def enrich(r):
    x=dict(r); x["task"]="bloom_rewrite"; x["sft_text"]=build_sft_text("bloom_rewrite",x); x["prompt_text"]=build_prompt_only_text("bloom_rewrite",x); x["generation_prompt"]=build_generation_prompt("bloom_rewrite",x); return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--mix-bloom",type=float,default=.40); ap.add_argument("--mix-qa",type=float,default=.30); ap.add_argument("--mix-sum",type=float,default=.30); args=ap.parse_args()
    locked_train=read(LOCKED/"train.jsonl"); locked_val=read(LOCKED/"validation.jsonl"); locked_test_path=LOCKED/"test.jsonl"; locked_test=read(locked_test_path)
    if len(locked_test)!=8321: raise SystemExit(f"Frozen test must contain 8321 rows; got {len(locked_test)}")
    bloom_test_keys={str(r.get("group_id") or r.get("source_id") or r.get("source_question","")).lower() for r in locked_test if r.get("task")=="bloom_rewrite"}
    bloom_train=read(V4/"train.jsonl") if (V4/"train.jsonl").exists() else []
    bloom_val=read(V4/"validation.jsonl") if (V4/"validation.jsonl").exists() else []
    if not bloom_train or not bloom_val: raise SystemExit("Run prepare_bloom_rewrite_v4.py for train and validation first")
    def leak_filter(rows): return [enrich(r) for r in rows if str(r.get("group_id") or r.get("source_id") or r.get("source_question","")).lower() not in bloom_test_keys]
    bloom_train=leak_filter(bloom_train); bloom_val=leak_filter(bloom_val)
    qa_train=[r for r in locked_train if r.get("task")=="qa"]; sum_train=[r for r in locked_train if r.get("task")=="summarization"]
    qa_val=[r for r in locked_val if r.get("task")=="qa"]; sum_val=[r for r in locked_val if r.get("task")=="summarization"]
    rng=random.Random(args.seed); n_b=len(bloom_train); total=round(n_b/args.mix_bloom); n_q=min(len(qa_train),round(total*args.mix_qa)); n_s=min(len(sum_train),round(total*args.mix_sum)); rng.shuffle(qa_train); rng.shuffle(sum_train)
    train=bloom_train+qa_train[:n_q]+sum_train[:n_s]; rng.shuffle(train); val=bloom_val+qa_val+sum_val; rng.shuffle(val)
    write(OUT/"train.jsonl",train); write(OUT/"validation.jsonl",val); write(OUT/"test.jsonl",locked_test)
    old_hash=hfile(locked_test_path); new_hash=hfile(OUT/"test.jsonl")
    if old_hash!=new_hash: raise SystemExit("FATAL: frozen test bytes changed")
    counts={s:{"total":len(x),"by_task":dict(Counter(r["task"] for r in x))} for s,x in (("train",train),("validation",val),("test",locked_test))}
    manifest={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"dataset_version":"multitask_bloom_rewrite_v4_1","seed":args.seed,"mix":{"bloom_rewrite":args.mix_bloom,"qa":args.mix_qa,"summarization":args.mix_sum},"bloom_dataset_version":"bloom_rewrite_synth_v4_1","locked_baseline_test":"data/multitask_bloom_rewrite/test.jsonl","locked_test_sha256":old_hash,"test_frozen":True,"test_identical_to_baseline":True,"counts":counts,"train_bloom_rows":len(bloom_train),"validation_bloom_rows":len(bloom_val),"notes":["Bloom v4 targets are teacher-generated and strictly validated.","QA and summarization reuse the locked training/validation pools.","The baseline 8,321-row test is copied byte-for-byte.","Do not tune on test."]}
    (OUT/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
