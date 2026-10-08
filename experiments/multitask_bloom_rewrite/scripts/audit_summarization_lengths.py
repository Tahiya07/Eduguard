#!/usr/bin/env python
"""Audit final BillSum rows for source/target length risk; no model inference."""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path
SCRIPT_DIR=Path(__file__).resolve().parent
EXPERIMENT_DIR=SCRIPT_DIR.parent
REPO_ROOT=EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path: sys.path.insert(0,str(EXPERIMENT_DIR))
from paths import MULTITASK_FINAL_DATA_DIR, TASK_SUMMARIZATION

def read_jsonl(path):
    with path.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def stats(values):
    if not values: return {"n":0}
    s=sorted(values)
    return {"n":len(s),"mean":round(sum(s)/len(s),2),"median":round(statistics.median(s),2),"p95":s[min(len(s)-1,round(.95*(len(s)-1)))],"min":s[0],"max":s[-1]}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default=str(MULTITASK_FINAL_DATA_DIR/"train.jsonl"))
    ap.add_argument("--max-seq-length",type=int,default=8192)
    ap.add_argument("--max-new-tokens",type=int,default=128)
    ap.add_argument("--approx-chars-per-token",type=float,default=4.0)
    ap.add_argument("--output",default=str(REPO_ROOT/"experiments/multitask_bloom_rewrite/reports/summarization_length_audit.json"))
    a=ap.parse_args(); p=Path(a.dataset); p= p if p.is_absolute() else REPO_ROOT/p
    rows=[r for r in read_jsonl(p) if r.get("task")==TASK_SUMMARIZATION]
    src=[str(r.get("article") or "") for r in rows]; tgt=[str(r.get("abstract") or "") for r in rows]
    sc=[len(x) for x in src]; tc=[len(x) for x in tgt]; st=[int(x/a.approx_chars_per_token) for x in sc]; tt=[int(x/a.approx_chars_per_token) for x in tc]
    report={"dataset":str(p),"source_dataset":"FiscalNote/billsum","task":"summarization","n":len(rows),"source_chars":stats(sc),"target_chars":stats(tc),"source_tokens_est":stats(st),"target_tokens_est":stats(tt),"max_seq_length":a.max_seq_length,"max_new_tokens":a.max_new_tokens,"source_over_budget_est":sum(x+80>a.max_seq_length for x in st),"target_exceeds_generation_cap_est":sum(x>a.max_new_tokens for x in tt),"policy":"Final preparation excludes BillSum examples whose complete SFT sequence exceeds 8192; the article is not silently truncated."}
    out=Path(a.output); out=out if out.is_absolute() else REPO_ROOT/out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
