#!/usr/bin/env python
"""Generate Bloom rewrite supervision v4 with a stronger teacher and strict QC.

Important:
- Uses only source questions + target Bloom level as teacher inputs.
- Does not use source Bloom level in the generation prompt.
- Reuses the existing v3 TRAIN/VALIDATION source split only to preserve the
  experiment's leakage grouping. Existing v3 targets are never used as labels.
- Does not modify the locked multitask test set.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR=Path(__file__).resolve().parent
EXP=SCRIPT_DIR.parent
ROOT=EXP.parents[1]
if str(EXP) not in sys.path: sys.path.insert(0,str(EXP))
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from bloom_target_policy_v4 import BLOOM_LEVELS, POLICY_VERSION, build_teacher_messages, validate_candidate, clean_output, canonical_level


def read_jsonl(p):
    rows=[]
    with open(p,encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def write_jsonl(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()

def render(messages,tokenizer):
    if getattr(tokenizer,"chat_template",None):
        return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    parts=[]
    for m in messages: parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(parts)+"\n<|im_start|>assistant\n"

def load_teacher(model_id,device):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    dtype=torch.float16 if device=="cuda" else torch.float32
    model=AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=True,torch_dtype=dtype)
    model.eval().to(device)
    return tok,model

def make_generator(tok,model,device,max_input=512,max_new=128,temperature=.35,top_p=.9):
    import torch
    def generate(source,target,retry=False):
        prompt=render(build_teacher_messages(source,target,retry=retry),tok)
        x=tok(prompt,return_tensors="pt",truncation=True,max_length=max_input)
        x={k:v.to(device) for k,v in x.items()}
        kwargs=dict(max_new_tokens=max_new,do_sample=True,temperature=temperature,top_p=top_p,
                    pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        with torch.no_grad(): y=model.generate(**x,**kwargs)
        return clean_output(tok.decode(y[0][x["input_ids"].shape[1]:],skip_special_tokens=True))
    return generate

def load_semantic():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model=SentenceTransformer("BAAI/bge-small-en-v1.5")
        def sim(a,b):
            e=model.encode([a,b],normalize_embeddings=True)
            return float(np.dot(e[0],e[1]))
        return sim,"BAAI/bge-small-en-v1.5"
    except Exception as e:
        return None,f"unavailable: {e}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher-model",required=True)
    ap.add_argument("--input-v3",default="data/bloom_rewrite_versions/bloom_rewrite_synth_v3")
    ap.add_argument("--output-dir",default="data/bloom_rewrite_versions/bloom_rewrite_synth_v4")
    ap.add_argument("--split",choices=["train","validation"],default="train")
    ap.add_argument("--attempts",type=int,default=3)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--no-semantic",action="store_true")
    ap.add_argument("--min-semantic",type=float,default=.55)
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args()
    random.seed(args.seed)
    inp=Path(args.input_v3); out=Path(args.output_dir)
    source_rows=read_jsonl(inp/f"{args.split}.jsonl")
    pairs={}
    for r in source_rows:
        q=r.get("source_question"); t=canonical_level(r.get("target_bloom_level"))
        if q and t: pairs[(str(r.get("source_id") or sha(q)[:16]),q,t)]=r
    items=list(pairs.values()); random.shuffle(items)
    if args.limit: items=items[:args.limit]
    tok,model=load_teacher(args.teacher_model,args.device)
    generate=make_generator(tok,model,args.device)
    sim_fn,sim_name=(None,"disabled") if args.no_semantic else load_semantic()
    accepted=[]; failures=Counter(); attempts_used=[]
    for i,row in enumerate(items,1):
        source=row["source_question"]; target=row["target_bloom_level"]
        best=None
        for attempt in range(args.attempts):
            candidate=generate(source,target,retry=attempt>0)
            semantic=sim_fn(source,candidate) if sim_fn and candidate else None
            v=validate_candidate(source,target,candidate,semantic_similarity=semantic,min_semantic_similarity=args.min_semantic)
            if v.ok:
                best=(candidate,v,attempt+1); break
            failures[v.failure_category or "QUALITY_REJECTION"]+=1
        if best is None:
            attempts_used.append(args.attempts)
            continue
        candidate,v,ntry=best; attempts_used.append(ntry)
        src_id=str(row.get("source_id") or sha(source)[:16])
        rec={
            "example_id":sha(f"v4|{src_id}|{target}|{candidate}")[:16],
            "source_id":src_id,
            "group_id":row.get("group_id"),
            "split":args.split,
            "source_question":source,
            "source_bloom_level":row.get("source_bloom_level"),
            "target_bloom_level":target,
            "target_rewrite":candidate,
            "transformation_type":f"{row.get('source_bloom_level')}->{target}",
            "synthetic_or_original":"synthetic_teacher_generated",
            "synthetic":True,
            "dataset_version":"bloom_rewrite_synth_v4",
            "policy_version":POLICY_VERSION,
            "quality_status":"pass",
            "validation":v.__dict__,
            "teacher_model":args.teacher_model,
            "generator_inputs":["source_question","target_bloom_level"],
            "teacher_attempts":ntry,
        }
        accepted.append(rec)
        if i%25==0: print(f"{i}/{len(items)} accepted={len(accepted)}")
    write_jsonl(out/f"{args.split}.jsonl",accepted)
    report={
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),"dataset_version":"bloom_rewrite_synth_v4",
        "policy_version":POLICY_VERSION,"teacher_model":args.teacher_model,"source_dataset":str(inp),
        "split":args.split,"source_pairs":len(items),"accepted":len(accepted),"rejected":len(items)-len(accepted),
        "acceptance_rate":len(accepted)/len(items) if items else 0,"failure_counts":dict(failures),
        "semantic_model":sim_name,"min_semantic":args.min_semantic,"attempts":args.attempts,
        "transformation_counts":dict(Counter(x["transformation_type"] for x in accepted)),
    }
    (out/f"{args.split}_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
