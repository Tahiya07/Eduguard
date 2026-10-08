#!/usr/bin/env python3
"""Deterministic BloomShift candidate audit.

This audit flags structural/provenance/shortcut risks; it does not assign
human semantic or Bloom judgments.
"""

from __future__ import annotations
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

LEVELS = ["Remember","Understand","Apply","Analyze","Evaluate","Create"]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()

def load_dataset(root: Path):
    rows=[]
    for split in ("train","validation","test"):
        with (root/f"{split}.json").open(encoding="utf-8") as f:
            data=json.load(f)
        rows.extend(data)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="data/rewrite dataset/bloomshift_final_candidate")
    ap.add_argument("--output",default=None)
    args=ap.parse_args()
    root=Path(args.root)
    rows=load_dataset(root)

    ids=[r["example_id"] for r in rows]
    groups=defaultdict(list)
    for r in rows: groups[r["source_id"]].append(r)

    split_groups={s:{str(r["group_id"]) for r in rows if r["split"]==s}
                  for s in ("train","validation","test")}
    split_overlap={}
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        split_overlap[f"{a}_{b}"]=len(split_groups[a]&split_groups[b])

    transitions={f"{s}->{t}" for s in LEVELS for t in LEVELS if s!=t}
    present={r["source_bloom_level"]+"->"+r["target_bloom_level"] for r in rows}
    missing=sorted(transitions-present)

    templates=defaultdict(list)
    for r in rows:
        target=norm(r["target_rewrite"])
        anchor=norm(r.get("source_anchor",""))
        if anchor and anchor in target:
            target=target.replace(anchor,"§SOURCE§")
        templates[target].append(r)
    cross_split=[v for v in templates.values() if len({r["split"] for r in v})>1]
    single_target_cross=[
        v for v in cross_split if len({r["target_bloom_level"] for r in v})==1
    ]

    quality_flags=[]
    checks=[
        ("comma_and_meets", re.compile(r"whether[^?]{0,250},\s*and\s+[^?]{0,150}\s+meets\b",re.I)),
        ("double_space", re.compile(r"\s{2,}")),
        ("space_before_punctuation", re.compile(r"\s+[,.?]")),
        ("double_terminal_question", re.compile(r"\?\s*[.?]$")),
    ]
    for r in rows:
        for name,pat in checks:
            if pat.search(r["target_rewrite"]):
                quality_flags.append({"example_id":r["example_id"],"check":name,
                                      "target_rewrite":r["target_rewrite"]})

    report={
      "rows":len(rows),
      "unique_example_ids":len(set(ids)),
      "duplicate_example_ids":sum(n-1 for n in Counter(ids).values() if n>1),
      "split_rows":dict(Counter(r["split"] for r in rows)),
      "unique_source_ids":len(groups),
      "source_group_size_distribution":dict(Counter(len(v) for v in groups.values())),
      "source_groups_with_nonuniform_target_count":sum(1 for v in groups.values() if len(v) not in (1,2,3,4,5)),
      "split_group_overlap":split_overlap,
      "identity_rows":sum(r["source_bloom_level"]==r["target_bloom_level"] for r in rows),
      "missing_non_identity_transitions":missing,
      "normalized_template_count":len(templates),
      "cross_split_template_count":len(cross_split),
      "cross_split_single_target_template_count":len(single_target_cross),
      "quality_flags":quality_flags,
      "near_duplicate_source_review_note":"Use a separate similarity audit before claiming source-text uniqueness; exact source repetition is expected because each source may yield multiple target transformations.",
    }
    out=json.dumps(report,indent=2,ensure_ascii=False)
    if args.output: Path(args.output).write_text(out+"\n",encoding="utf-8")
    else: print(out)

if __name__=="__main__": main()
