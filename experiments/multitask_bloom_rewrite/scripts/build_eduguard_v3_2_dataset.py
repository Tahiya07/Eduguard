"""Build the task-separated SFT dataset v3.2: EduGuard v3.1 Bloom + closed-book academic QA (qa_eduguard_v2) + summarization.

Inputs (read only, verified before use):
  data/multitask_bloom_rewrite_separate_eduguard_v3_1/  bloom_rewrite/ and summarization/ are copied byte-for-byte
  data/qa_eduguard_v2/                                 1,172 / 146 / 147 closed-book QA rows (reserve.jsonl unused)
Output (refuses to overwrite):
  data/multitask_bloom_rewrite_separate_eduguard_v3_2/

QA rows replace the SQuAD-style extractive QA of v3.1. Each row keeps its qa_eduguard_v2 fields and gains
task="qa", qa_format="closed_book", answer (= reference_answer, the training target) and the sft_text /
prompt_text / generation_prompt rendered by prompts.py. dataset_io.validate_separate_dataset re-verifies every row
against the source on each load.

Optional: --check-tokens counts every QA SFT sequence and answer with the cached Qwen2.5-1.5B tokenizer (CPU, offline).
Run from the repository root:  python -I experiments/multitask_bloom_rewrite/scripts/build_eduguard_v3_2_dataset.py
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
BASE = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate_eduguard_v3_1"
QA_SRC = REPO_ROOT / "data" / "qa_eduguard_v2"
OUT = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate_eduguard_v3_2"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
QA_KEEP_FIELDS = (
    "id", "split", "question", "reference_answer", "key_points", "discipline", "discipline_canonical",
    "discipline_group", "course", "topic", "qa_type", "qa_type_canonical", "difficulty", "difficulty_canonical",
    "verification", "provenance", "answer_review", "answer_review_note",
)
COPIED = ("bloom_rewrite", "summarization")
EXTRA_FILES = ("summarization/summarization_eligibility.json", "summarization/summarization_exclusions.csv")


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check-tokens", action="store_true")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    args = ap.parse_args()
    if OUT.exists():
        sys.exit(f"{OUT} already exists; refusing to overwrite.")

    # 1. verify inputs
    base_integrity = dataset_io.validate_separate_dataset(BASE)
    base_manifest = dataset_io.load_manifest(BASE)
    if base_manifest["tokenizer"]["model_id"] != MODEL_ID:
        sys.exit(f"base tokenizer is {base_manifest['tokenizer']['model_id']}, expected {MODEL_ID}")
    qa_manifest_path = QA_SRC / "dataset_manifest.json"
    qa_manifest = json.loads(qa_manifest_path.read_text(encoding="utf-8"))
    for name, digest in qa_manifest["files_sha256"].items():
        if dataset_io.sha256_file(QA_SRC / name) != digest:
            sys.exit(f"{QA_SRC / name} does not match its manifest SHA-256")

    # 2. qa rows: closed-book SFT format
    OUT.mkdir(parents=True)
    (OUT / "qa").mkdir()
    hashes, counts, qa_stats = {}, {s: {} for s in SPLITS}, {}
    for split in SPLITS:
        out_rows = []
        for row in dataset_io.read_jsonl(QA_SRC / f"{split}.jsonl"):
            if row["split"] != split:
                sys.exit(f"{row['id']}: source split {row['split']!r} in {split}.jsonl")
            if split != "train" and row["answer_review"] != "correct":
                sys.exit(f"{row['id']}: held-out answer is not reviewed as correct")
            record = {"task": "qa", "qa_format": dataset_io.QA_CLOSED_BOOK}
            record.update({k: row[k] for k in QA_KEEP_FIELDS if k in row})
            record["answer"] = row["reference_answer"]
            record["sft_text"] = build_sft_text("qa", record)
            record["prompt_text"] = build_prompt_only_text("qa", record)
            record["generation_prompt"] = build_generation_prompt("qa", record)
            if not record["sft_text"].startswith(record["prompt_text"]):
                sys.exit(f"{record['id']}: prompt_text is not a prefix of sft_text")
            if not record["sft_text"].endswith(record["answer"].strip() + "<|im_end|>"):
                sys.exit(f"{record['id']}: sft_text does not end with the reference answer")
            if not record["generation_prompt"].endswith("<|im_start|>assistant\n"):
                sys.exit(f"{record['id']}: generation_prompt does not open an assistant turn")
            out_rows.append(record)
        path = OUT / "qa" / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in out_rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        hashes[f"qa/{split}.jsonl"] = dataset_io.sha256_file(path)
        counts[split]["qa"] = len(out_rows)
        qa_stats[split] = {
            "rows": len(out_rows),
            "discipline_group": dict(Counter(r["discipline_group"] for r in out_rows).most_common()),
            "answer_review": dict(Counter(r["answer_review"] for r in out_rows).most_common()),
        }

    # 3. bloom_rewrite and summarization: byte-for-byte copies of the verified v3.1 dataset
    for task in COPIED:
        (OUT / task).mkdir()
        for split in SPLITS:
            name = f"{task}/{split}.jsonl"
            shutil.copy2(BASE / name, OUT / name)
            hashes[name] = dataset_io.sha256_file(OUT / name)
            if hashes[name] != base_manifest["generated_files_sha256"][name]:
                sys.exit(f"copied {name} differs from the v3.1 dataset")
            counts[split][task] = base_manifest["counts"][split][task]
    for extra in EXTRA_FILES:
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
        token_check = {"performed": True, "model_id": MODEL_ID, "max_seq_length": args.max_seq_length, "qa": {}}
        for split in SPLITS:
            rows = dataset_io.read_jsonl(OUT / "qa" / f"{split}.jsonl")
            seq = [len(tok(r["sft_text"], add_special_tokens=False)["input_ids"]) for r in rows]
            ans = [len(tok(r["answer"], add_special_tokens=False)["input_ids"]) for r in rows]
            token_check["qa"][split] = {"rows": len(rows), "max_sequence_tokens": max(seq),
                                        "max_answer_tokens": max(ans), "overlength": sum(n > args.max_seq_length for n in seq)}
            if token_check["qa"][split]["overlength"]:
                sys.exit(f"qa/{split}: sequences exceed {args.max_seq_length} tokens")

    # 5. manifest (bloom and summarization records carried over from v3.1)
    manifest = {k: v for k, v in base_manifest.items()
                if k not in ("timestamp_utc", "generated_files_sha256", "counts", "notes", "derived_from", "tokenizer")}
    manifest.update({
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "qa_source": {
            "kind": dataset_io.QA_EDUGUARD_SOURCE_KIND,
            "path": rel(QA_SRC),
            "format": "closed_book: system + 'Question:' user turn -> reference_answer (no context passage)",
            "files_sha256": {**qa_manifest["files_sha256"],
                             "dataset_manifest.json": dataset_io.sha256_file(qa_manifest_path)},
            "source": qa_manifest["source"],
            "answer_review": qa_manifest["answer_review"],
            "replaces": "SQuAD-style extractive QA used in eduguard_v3_1 (context/answer/answer_start)",
            "stats": qa_stats,
        },
        "tokenizer": {"model_id": MODEL_ID, "max_seq_length": args.max_seq_length,
                      "qa_check": token_check,
                      "bloom_rewrite_check": base_manifest["tokenizer"].get("bloom_rewrite_check"),
                      "summarization_check": "inherited (files copied byte-for-byte)"},
        "derived_from": {"dataset_dir": rel(BASE), "manifest_sha256": base_integrity["manifest_sha256"],
                         "copied_unchanged": ["bloom_rewrite/*", "summarization/*"]},
        "generated_files_sha256": dict(sorted(hashes.items())),
        "counts": counts,
        "notes": [
            "No combined train.jsonl, validation.jsonl, or test.jsonl is persisted.",
            "qa is closed-book academic QA from data/qa_eduguard_v2; bloom_rewrite and summarization are unchanged from v3.1.",
            "QA answers are not human-verified; held-out QA answers passed a model-assisted review (see qa_source.answer_review).",
        ],
    })
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 6. verify with the loader that training uses
    integrity = dataset_io.validate_separate_dataset(OUT)
    print(json.dumps({"out": rel(OUT), "counts": integrity["counts"], "qa_source_check": integrity["qa_source_check"],
                      "bloom_source_check": integrity["bloom_source_check"], "token_check": token_check}, indent=1))


if __name__ == "__main__":
    main()
