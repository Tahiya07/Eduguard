"""Build a task-separated SFT dataset whose bloom_rewrite task comes from the EduGuard v3.1 Bloom splits.

Inputs (read only, verified before use):
  data/bloom_rewrite_eduguard_v3_1/                 v3.1 Bloom splits (train/validation/test.jsonl + manifest)
  data/multitask_bloom_rewrite_separate_sumclean_v1/  existing task-separated dataset; its qa/ and summarization/
                                                    files are copied byte-for-byte (summarization eligibility kept)
Output (refuses to overwrite):
  data/multitask_bloom_rewrite_separate_eduguard_v3_1/

Bloom rows keep their source IDs, target labels and split assignment; sft_text, prompt_text and generation_prompt
are rendered with the unchanged prompts.py builders. All rows flagged gold_label_contested are excluded and recorded
in the manifest; dataset_io.validate_separate_dataset re-derives and enforces that exclusion on every load.

Optional: --check-tokens counts every Bloom SFT sequence with the locally cached Qwen tokenizer (CPU only, offline).
Run from the repository root:  python -I experiments/multitask_bloom_rewrite/scripts/build_eduguard_v3_1_dataset.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPTS_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for item in (EXPERIMENT_DIR, SCRIPTS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import dataset_io  # noqa: E402
from prompts import build_generation_prompt, build_prompt_only_text, build_sft_text  # noqa: E402

SPLITS = dataset_io.SPLITS
BLOOM_SRC = REPO_ROOT / "data" / "bloom_rewrite_eduguard_v3_1"
BASE = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate_sumclean_v1"
OUT = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate_eduguard_v3_1"
# Generator for this dataset: Qwen2.5-1.5B-Instruct (not the 3B model). Its tokenizer is used for the length check.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
KEEP_FIELDS = (
    "task", "split", "id", "source_id", "group_id", "discipline", "topic", "source_question", "source_bloom_level",
    "target_bloom_level", "target_rewrite", "transformation_type", "eligibility", "answer_source_available",
    "content_provenance", "deliberately_false_claim", "claim_under_evaluation", "pass3_rewritten",
    "gold_label_contested", "review_status", "independent_review_verified",
)


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check-tokens", action="store_true")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    args = ap.parse_args()
    if OUT.exists():
        sys.exit(f"{OUT} already exists; refusing to overwrite.")

    # 1. verify inputs
    base_integrity = dataset_io.validate_separate_dataset(BASE)
    src_manifest_path = BLOOM_SRC / "dataset_manifest.json"
    src_manifest = json.loads(src_manifest_path.read_text(encoding="utf-8"))
    for name, digest in src_manifest["files_sha256"].items():
        if dataset_io.sha256_file(BLOOM_SRC / name) != digest:
            sys.exit(f"{BLOOM_SRC / name} does not match its manifest SHA-256")
    base_manifest = json.loads((BASE / "dataset_manifest.json").read_text(encoding="utf-8"))
    if base_manifest["tokenizer"]["model_id"] != MODEL_ID:
        sys.exit(f"base dataset tokenizer is {base_manifest['tokenizer']['model_id']}, expected {MODEL_ID}")

    # 2. bloom rows: exclude contested, render SFT text
    OUT.mkdir(parents=True)
    hashes, counts, excluded, bloom_stats = {}, {}, [], {}
    for split in SPLITS:
        src_rows = dataset_io.read_jsonl(BLOOM_SRC / f"{split}.jsonl")
        out_rows = []
        for row in src_rows:
            if row.get("split") != split or row.get("task") != "bloom_rewrite":
                sys.exit(f"source row {row.get('id')} has task/split {row.get('task')}/{row.get('split')}")
            if row["gold_label_contested"]:
                excluded.append({"id": row["id"], "split": split, "source_id": row["source_id"],
                                 "target_bloom_level": row["target_bloom_level"]})
                continue
            record = {k: row[k] for k in KEEP_FIELDS if k in row}
            record["sft_text"] = build_sft_text("bloom_rewrite", record)
            record["prompt_text"] = build_prompt_only_text("bloom_rewrite", record)
            record["generation_prompt"] = build_generation_prompt("bloom_rewrite", record)
            if not record["sft_text"].startswith(record["prompt_text"]):
                sys.exit(f"{record['id']}: prompt_text is not a prefix of sft_text")
            if not record["sft_text"].rstrip().endswith(record["target_rewrite"].strip() + "<|im_end|>"):
                sys.exit(f"{record['id']}: sft_text does not end with the target rewrite")
            out_rows.append(record)
        path = OUT / "bloom_rewrite" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in out_rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        hashes[f"bloom_rewrite/{split}.jsonl"] = dataset_io.sha256_file(path)
        counts[split] = {"bloom_rewrite": len(out_rows)}
        bloom_stats[split] = {
            "source_rows": len(src_rows), "excluded_contested": len(src_rows) - len(out_rows), "rows": len(out_rows),
            "source_questions": len({r["source_id"] for r in out_rows}),
            "usable_rows": sum(r["eligibility"] == "usable" for r in out_rows),
            "target_level_counts": dict(sorted(Counter(r["target_bloom_level"] for r in out_rows).items())),
        }

    # 3. qa and summarization: byte-for-byte copies of the verified base dataset
    for task in ("qa", "summarization"):
        (OUT / task).mkdir()
        for split in SPLITS:
            shutil.copy2(BASE / task / f"{split}.jsonl", OUT / task / f"{split}.jsonl")
            name = f"{task}/{split}.jsonl"
            hashes[name] = dataset_io.sha256_file(OUT / name)
            if hashes[name] != base_manifest["generated_files_sha256"][name]:
                sys.exit(f"copied {name} differs from the base dataset")
            for s2 in SPLITS:
                counts[s2][task] = base_manifest["counts"][s2][task]
    for extra in ("summarization/summarization_eligibility.json", "summarization/summarization_exclusions.csv"):
        if (BASE / extra).is_file():
            shutil.copy2(BASE / extra, OUT / extra)

    # 4. optional tokenizer length check (CPU, offline; no model weights are loaded)
    token_check = {"performed": False}
    if args.check_tokens:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        token_check = {"performed": True, "model_id": MODEL_ID,
                       "max_seq_length": args.max_seq_length, "bloom_rewrite": {}}
        for split in SPLITS:
            lens = [len(tok(r["sft_text"], add_special_tokens=False)["input_ids"])
                    for r in dataset_io.read_jsonl(OUT / "bloom_rewrite" / f"{split}.jsonl")]
            token_check["bloom_rewrite"][split] = {"rows": len(lens), "max_tokens": max(lens), "min_tokens": min(lens),
                                                   "overlength": sum(n > args.max_seq_length for n in lens)}
            if token_check["bloom_rewrite"][split]["overlength"]:
                sys.exit(f"bloom_rewrite/{split}: sequences exceed {args.max_seq_length} tokens")

    # 5. manifest
    manifest = {
        "version": dataset_io.MANIFEST_VERSION,
        "storage_layout": "one_directory_per_task",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": base_manifest.get("seed"),
        "generator_model_role": base_manifest.get("generator_model_role"),
        "classifier_dataset_excluded": True,
        "tasks": list(dataset_io.TASKS),
        "persisted_merged_jsonl": False,
        "bloom_source": {
            "kind": dataset_io.EDUGUARD_BLOOM_SOURCE_KIND,
            "path": rel(BLOOM_SRC),
            "version": src_manifest["version"],
            "files_sha256": {**src_manifest["files_sha256"],
                             "dataset_manifest.json": dataset_io.sha256_file(src_manifest_path)},
            "upstream": src_manifest["source"],
            "heldout_annotation_provenance": {
                "inclusion_basis": "eduguard_v3_1_model_assisted_repair",
                "independent_review_verified": False,
                "warnings": ["Bloom rewrites are model-assisted (Claude) and not human-verified; checks are automated.",
                             "Splits are pinned to final_corrected_dataset/training_ready/tests/reference_split_ids.json."],
            },
        },
        "bloom_exclusions": {
            "policy": "exclude rows whose gold label is contested (gold_label_contested = true) from training and evaluation",
            "reason": "Examples 213 and 570: source labelled Analyze but the task executes a known rule (Apply); "
                      "awaiting subject-matter-expert decision. See final_corrected_dataset/repaired_v3_20261125/work/challenge_review.json.",
            "excluded_rows": excluded,
            "excluded_count": len(excluded),
        },
        "bloom_rewrite_stats": bloom_stats,
        "qa_source": base_manifest.get("qa_source"),
        "summarization_source": base_manifest.get("summarization_source"),
        "upstream_datasets": base_manifest.get("upstream_datasets"),
        "summarization_quality_filter": base_manifest.get("summarization_quality_filter"),
        "tokenizer": {"model_id": MODEL_ID, "max_seq_length": args.max_seq_length,
                      "bloom_rewrite_check": token_check,
                      "qa_summarization_check": "inherited from the base dataset (files copied byte-for-byte)"},
        "derived_from": {"dataset_dir": rel(BASE), "manifest_sha256": base_integrity["manifest_sha256"],
                         "copied_unchanged": ["qa/*", "summarization/*"]},
        "generated_files_sha256": dict(sorted(hashes.items())),
        "counts": counts,
        "notes": [
            "No combined train.jsonl, validation.jsonl, or test.jsonl is persisted.",
            "bloom_rewrite is the EduGuard v3.1 split; qa and summarization are unchanged from the base dataset.",
            "Source IDs, target labels and split assignments of the Bloom rows are preserved; only contested rows are dropped.",
        ],
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 6. verify with the loader that training uses
    integrity = dataset_io.validate_separate_dataset(OUT)
    print(json.dumps({"out": rel(OUT), "counts": integrity["counts"], "excluded_contested": len(excluded),
                      "bloom_source_check": integrity.get("bloom_source_check"), "token_check": token_check}, indent=1))


if __name__ == "__main__":
    main()
