#!/usr/bin/env python3
"""Audit an expansion source pool against supplied inventories and flag near matches.

Exact normalization: Unicode NFKC + lowercase + alphanumeric tokens.
Near-match heuristic: token-set Jaccard >= --threshold, with a length prefilter.
This is a triage aid, not a semantic deduplication or Bloom-validity certification.
"""
from __future__ import annotations
import argparse, csv, json, re, unicodedata
from pathlib import Path

TEXT_KEYS = ("source_question", "question", "original_question", "question_text", "prompt")
TOKEN = re.compile(r"[a-z0-9]+", re.I)

def normalize(s: str) -> str:
    return " ".join(TOKEN.findall(unicodedata.normalize("NFKC", s or "").casefold()))

def read_rows(path: Path):
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, dict): data = data.get("data", data.get("records", []))
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def question(row):
    return next((str(row.get(k, "")).strip() for k in TEXT_KEYS if row.get(k)), "")

def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="append", required=True, help="Pool CSV; repeat for each part")
    ap.add_argument("--existing", action="append", default=[], help="Existing JSON/CSV inventory; repeat for every historical inventory")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()
    if not 0 < args.threshold <= 1: ap.error("--threshold must be in (0,1]")
    pool=[]
    for name in args.pool:
        for row in read_rows(Path(name)):
            q=question(row); n=normalize(q)
            if n: pool.append({"id": row.get("source_id", ""), "question": q, "norm": n, "tokens": set(n.split())})
    existing=[]
    for name in args.existing:
        for row in read_rows(Path(name)):
            q=question(row); n=normalize(q)
            if n: existing.append({"path": name, "question": q, "norm": n, "tokens": set(n.split())})
    exact_existing={r["norm"] for r in existing}
    internal=[]; cross=[]
    for i,a in enumerate(pool):
        for b in pool[i+1:]:
            if abs(len(a["norm"])-len(b["norm"])) > max(len(a["norm"]),len(b["norm"]))*0.25: continue
            score=jaccard(a["tokens"],b["tokens"])
            if score >= args.threshold:
                internal.append({"source_id_a":a["id"],"source_id_b":b["id"],"token_jaccard":round(score,4),"question_a":a["question"],"question_b":b["question"]})
        for b in existing:
            if a["norm"] == b["norm"]: continue
            if abs(len(a["norm"])-len(b["norm"])) > max(len(a["norm"]),len(b["norm"]))*0.35: continue
            score=jaccard(a["tokens"],b["tokens"])
            if score >= args.threshold:
                cross.append({"source_id":a["id"],"existing_path":b["path"],"token_jaccard":round(score,4),"source_question":a["question"],"existing_question":b["question"]})
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("near_duplicate_pool_pairs.csv",internal),("near_duplicate_existing_matches.csv",cross)):
        fields=list(rows[0]) if rows else (["source_id_a","source_id_b","token_jaccard","question_a","question_b"] if "pool_pairs" in name else ["source_id","existing_path","token_jaccard","source_question","existing_question"])
        with (out/name).open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={"pool_rows":len(pool),"pool_unique_normalized":len({r["norm"] for r in pool}),"existing_rows":len(existing),"exact_pool_matches_to_existing":sum(r["norm"] in exact_existing for r in pool),"internal_near_duplicate_pairs":len(internal),"near_matches_to_existing":len(cross),"threshold":args.threshold,"limitation":"near-match flags require human adjudication; all historical inventories must be supplied"}
    (out/"audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
