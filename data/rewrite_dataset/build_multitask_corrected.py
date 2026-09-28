#!/usr/bin/env python3
"""Build the corrected multitask SFT corpus from the corrected Bloom dataset."""
from __future__ import annotations
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BLOOM = ROOT / "data" / "rewrite_dataset"
BASE = ROOT / "data" / "multitask_bloom_rewrite"
OUT = ROOT / "data" / "multitask_corrected"
SEED = 42
LEVELS = {"Remember","Understand","Apply","Analyze","Evaluate","Create"}

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def key(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()

def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\\n" for r in rows), encoding="utf-8")

def main() -> None:
    bloom_train = read_jsonl(BLOOM/"bloom_rewrite_train.jsonl")
    bloom_val = read_jsonl(BLOOM/"bloom_rewrite_validation.jsonl")
    bloom_test = read_jsonl(BLOOM/"bloom_rewrite_test.jsonl")
    base_train = read_jsonl(BASE/"train.jsonl")
    base_val = read_jsonl(BASE/"validation.jsonl")
    base_test = read_jsonl(BASE/"test.jsonl")

    def task(rows, name):
        return [r for r in rows if r.get("task") == name]

    qa_train, sum_train = task(base_train,"qa"), task(base_train,"summarization")
    qa_val, sum_val = task(base_val,"qa"), task(base_val,"summarization")
    qa_test, sum_test = task(base_test,"qa"), task(base_test,"summarization")

    # Fixed QA/Sum counts from the locked baseline imply 6972 Bloom rows
    # (1162 complete source groups) for the closest integer 40/30/30 train mix.
    target_bloom_rows = 6972
    grouped = defaultdict(list)
    for r in bloom_train:
        grouped[r["source_id"]].append(r)
    groups = [rows for rows in grouped.values() if len(rows)==6 and {r["target_bloom_level"] for r in rows}==LEVELS]
    if len(groups)*6 < target_bloom_rows:
        target_bloom_rows = (len(groups))*6
    n_groups = target_bloom_rows // 6
    groups.sort(key=lambda g:key(g[0]["source_id"]))
    # Stable stratified allocation by original source Bloom level.
    by_level = defaultdict(list)
    for g in groups:
        by_level[g[0]["source_bloom_level"]].append(g)
    level_names = sorted(by_level)
    allocation = {lvl:int(n_groups*len(by_level[lvl])/len(groups)) for lvl in level_names}
    while sum(allocation.values()) < n_groups:
        lvl=max(level_names,key=lambda x:len(by_level[x])-allocation[x])
        allocation[lvl]+=1
    while sum(allocation.values()) > n_groups:
        lvl=max((x for x in level_names if allocation[x]>0),key=lambda x:allocation[x])
        allocation[lvl]-=1

    selected_groups=[]
    for lvl in level_names:
        selected_groups.extend(by_level[lvl][:allocation[lvl]])
    bloom_train_sel=[r for g in selected_groups for r in g]

    rng=random.Random(SEED)
    rng.shuffle(bloom_train_sel)
    train=bloom_train_sel+qa_train+sum_train
    val=bloom_val+qa_val+sum_val
    test=bloom_test+qa_test+sum_test
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)

    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/"train.jsonl",train); write(OUT/"validation.jsonl",val); write(OUT/"test.jsonl",test)

    def stats(rows):
        c=Counter(r.get("task") for r in rows)
        total=len(rows)
        return {"total":total,"by_task":dict(c),"proportions":{k:round(v/total,6) for k,v in c.items()}}

    manifest={
        "dataset_version":"multitask_corrected_v2",
        "seed":SEED,
        "base_model_target":"Qwen2.5-1.5B-Instruct",
        "mix_preset":"A",
        "requested_mix":{"bloom_rewrite":0.4,"qa":0.3,"summarization":0.3},
        "sources":{
            "bloom_rewrite":"data/rewrite_dataset/bloom_rewrite_*.jsonl",
            "qa_and_summarization":"data/multitask_bloom_rewrite/{train,validation,test}.jsonl"
        },
        "train":stats(train),
        "validation":stats(val),
        "test":stats(test),
        "bloom_train_source_groups_selected":len(selected_groups),
        "bloom_train_rows_selected":len(bloom_train_sel),
        "bloom_train_each_source_has_all_six_targets":True,
        "qa_and_summarization_unchanged":True,
        "old_bloom_rows_excluded":True,
        "test_not_resampled_or_tuned":True,
    }
    (OUT/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(manifest,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
