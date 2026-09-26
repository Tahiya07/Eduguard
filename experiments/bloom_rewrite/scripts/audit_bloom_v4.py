#!/usr/bin/env python
"""Audit Bloom v4 without using the held-out multitask test as a tuning source."""
from __future__ import annotations
import argparse,json,re
from collections import Counter
from pathlib import Path
import sys

SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0,str(SCRIPT_DIR.parent))

from bloom_target_policy_v4 import validate_candidate

def read(p):
    with open(p,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-dir",default="data/bloom_rewrite_versions/bloom_rewrite_synth_v4_2"); ap.add_argument("--split",choices=["train","validation","all"],default="all"); ap.add_argument("--min-semantic",type=float,default=.68); args=ap.parse_args()
    d=Path(args.dataset_dir); splits=["train","validation"] if args.split=="all" else [args.split]
    rows=[]
    for s in splits:
        p=d/f"{s}.jsonl"
        if p.exists(): rows += read(p)
    failures=Counter(); trans=Counter(); exact=0; generic=0
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        semantic_model=SentenceTransformer("BAAI/bge-small-en-v1.5")
        def sim(a,b):
            e=semantic_model.encode([a,b],normalize_embeddings=True)
            return float(np.dot(e[0],e[1]))
    except Exception as exc:
        raise SystemExit(f"Semantic audit model unavailable: {exc}")
    hard_prefixes=(
        "EMPTY_OUTPUT","META_OR_ANSWER_LANGUAGE","INVALID_EXAM_QUESTION_FORM",
        "INVENTED_ARTIFACT_REFERENCE","SCOPE_EXPANSION","SCOPE_CONTENT_ADDITION",
        "PROTECTED_SPAN_LOSS","LOW_SEMANTIC_SIMILARITY","CLASSIFIER_TARGET_MISMATCH",
        "CLASSIFIER_LOW_CONFIDENCE","MULTIPLE_QUESTIONS","TOO_LONG",
        "NEAR_SOURCE_COPY","EXACT_SOURCE_COPY"
    )
    for r in rows:
        semantic=sim(r["source_question"],r["target_rewrite"])
        v=validate_candidate(
            r["source_question"],
            r["target_bloom_level"],
            r["target_rewrite"],
            semantic_similarity=semantic,
            min_semantic_similarity=args.min_semantic,
        )
        judge=r.get("teacher_judge") or {}
        if judge.get("pass") is True:
            hard=[x for x in v.reasons if x.upper().startswith(hard_prefixes)]
            if hard:
                failures["HARD_FAILURE_WITH_JUDGE_PASS"] += len(hard)
        elif not v.ok:
            failures[v.failure_category]+=1
        trans[r["transformation_type"]]+=1
        if r["source_question"].strip().lower().rstrip("?.")==r["target_rewrite"].strip().lower().rstrip("?."): exact+=1
        if any(x in r["target_rewrite"].lower() for x in ("compare the main components of","assess how well","stated academic criteria","analyze how the parts of")): generic+=1
    report={"n":len(rows),"failures_on_revalidation":dict(failures),"exact_source_copies":exact,"known_v3_template_phrases":generic,"teacher_judged_rows":sum(1 for r in rows if (r.get("teacher_judge") or {}).get("pass") is True),"transformation_counts":dict(trans),"semantic_model":"BAAI/bge-small-en-v1.5","min_semantic":args.min_semantic,"clean":not failures and not exact and not generic}
    print(json.dumps(report,indent=2))
    (d/"audit_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if failures or exact or generic: raise SystemExit(2)
if __name__=="__main__":main()
