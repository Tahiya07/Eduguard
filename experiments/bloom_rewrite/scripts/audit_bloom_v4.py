#!/usr/bin/env python
"""Audit Bloom v4 without using the held-out multitask test as a tuning source."""
from __future__ import annotations
import argparse,json,re
from collections import Counter
from pathlib import Path
from bloom_target_policy_v4 import validate_candidate

def read(p):
    with open(p,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-dir",default="data/bloom_rewrite_versions/bloom_rewrite_synth_v4"); ap.add_argument("--split",choices=["train","validation","all"],default="all"); args=ap.parse_args()
    d=Path(args.dataset_dir); splits=["train","validation"] if args.split=="all" else [args.split]
    rows=[]
    for s in splits:
        p=d/f"{s}.jsonl"
        if p.exists(): rows += read(p)
    failures=Counter(); trans=Counter(); exact=0; generic=0
    for r in rows:
        v=validate_candidate(r["source_question"],r["target_bloom_level"],r["target_rewrite"])
        if not v.ok: failures[v.failure_category]+=1
        trans[r["transformation_type"]]+=1
        if r["source_question"].strip().lower().rstrip("?.")==r["target_rewrite"].strip().lower().rstrip("?."): exact+=1
        if any(x in r["target_rewrite"].lower() for x in ("compare the main components of","assess how well","stated academic criteria","analyze how the parts of")): generic+=1
    report={"n":len(rows),"failures_on_revalidation":dict(failures),"exact_source_copies":exact,"known_v3_template_phrases":generic,"transformation_counts":dict(trans),"clean":not failures and not exact and not generic}
    print(json.dumps(report,indent=2))
    (d/"audit_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if failures or exact or generic: raise SystemExit(2)
if __name__=="__main__":main()
