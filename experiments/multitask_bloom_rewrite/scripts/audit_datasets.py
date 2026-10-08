#!/usr/bin/env python
"""Audit the finalized generator inputs and prepared corpus without training."""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BLOOM=ROOT/"data/rewrite dataset/bloomshift_final_candidate"
FINAL=ROOT/"data/multitask_bloom_rewrite_final"

def sha256(p):
    h=hashlib.sha256();
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def load_jsonl(p): return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]
def text_stats(values):
    lens=[len(re.findall(r"\b\w+\b",str(x))) for x in values]
    return {"n":len(lens),"min":min(lens) if lens else 0,"median":sorted(lens)[len(lens)//2] if lens else 0,"max":max(lens) if lens else 0}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--skip-final",action="store_true"); a=ap.parse_args()
    report={"generator_inputs":{"bloomshift":{"path":str(BLOOM.relative_to(ROOT)),"version":"bloomshift_final_candidate"},"qa":"rajpurkar/squad","summarization":"FiscalNote/billsum"},"final_corpus":{},"errors":[]}
    for split in ("train","validation","test"):
        p=BLOOM/(split+".json")
        if not p.is_file(): report["errors"].append(f"missing BloomShift {p}"); continue
        rows=json.loads(p.read_text(encoding="utf-8")); report["generator_inputs"]["bloomshift"][split]={"rows":len(rows),"sha256":sha256(p),"groups":len({r.get("group_id") for r in rows})}
    if not a.skip_final:
        for split in ("train","validation","test"):
            p=FINAL/(split+".jsonl")
            if not p.is_file(): report["errors"].append(f"missing final corpus {p}"); continue
            rows=load_jsonl(p); counts=Counter(r.get("task") for r in rows); report["final_corpus"][split]={"rows":len(rows),"task_counts":dict(counts),"sha256":sha256(p),"missing_sft":sum(not r.get("sft_text") for r in rows),"missing_prompt":sum(not r.get("prompt_text") for r in rows)}
        train=report["final_corpus"].get("train",{}).get("task_counts",{}); total=sum(train.values()); report["final_corpus"]["train_mix"]={k:round(v/total,6) for k,v in train.items()} if total else {}
    report["status"]="PASS" if not report["errors"] else "FAIL"
    out=ROOT/"experiments/multitask_bloom_rewrite/reports/dataset_audit.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); raise SystemExit(0 if not report["errors"] else 2)
if __name__=="__main__": main()
