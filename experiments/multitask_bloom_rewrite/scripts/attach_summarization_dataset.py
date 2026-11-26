"""Attach an owner-built lecture summarization dataset: replace the summarization task of the active SFT dataset.

Input  --source DIR   train.jsonl, validation.jsonl, test.jsonl (read only, never modified). One JSON object per line:
                      required: id, source_text (lecture PDF/slide/notes text), summary (the target study summary)
                      optional, preserved: department, course, topic, source_type, review_status, any other field
       --base DIR     current task-separated dataset (default: paths.TASK_SEPARATED_DATA_DIR); its bloom_rewrite/
                      and qa/ files are copied byte-for-byte
Output --out DIR      new task-separated dataset (refuses to overwrite); summarization rows are rendered with the
                      lecture prompt (prompts.LECTURE_SUMMARIZATION_SYSTEM); no arXiv rows are kept
       --write-config also writes configs/<config-name>.json from the active config with dataset_dir, fresh
                      output/results dirs, validation/test limits and the summarization generation length updated

Checks before anything is written (any failure stops the build):
  * required fields are non-empty strings; IDs are unique across all three splits; a "split" field, if present,
    matches its file; rows have review_status == --require-review-status (default "verified"; "" disables)
  * no exact duplicate source_text or summary within or across splits (whitespace/case-normalised)
  * no cross-split near-duplicate source_text (char 3-5-gram TF-IDF cosine >= --near-dup-threshold, needs sklearn)
  * every summary is shorter than its source text
  * with --check-tokens: every SFT sequence fits --max-seq-length with the Qwen2.5-1.5B tokenizer (CPU, offline)
The loader then re-verifies every row against the source files on each load.

Example (repository root, project venv):
  .venv/Scripts/python.exe -I experiments/multitask_bloom_rewrite/scripts/attach_summarization_dataset.py \
      --source data/summarization_eduguard_v1 --out data/multitask_bloom_rewrite_separate_eduguard_v3_3 \
      --check-tokens --write-config --config-name qwen15b_multitask_eduguard_v3_3_YYYYMMDD
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
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
import paths  # noqa: E402
from prompts import build_generation_prompt, build_prompt_only_text, build_sft_text  # noqa: E402

SPLITS = dataset_io.SPLITS
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
RESERVED = {"task", "split", "summarization_format", "article", "abstract", "sft_text", "prompt_text",
            "generation_prompt"}
COPIED = ("bloom_rewrite", "qa")


def resolve(p: Path) -> Path:
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


def rel(p: Path) -> str:
    return p.resolve().relative_to(REPO_ROOT).as_posix()


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def check_source(source: Path, require_status: str, near_dup: float) -> tuple[dict[str, list[dict]], dict]:
    rows = {s: dataset_io.read_jsonl(source / f"{s}.jsonl") for s in SPLITS}
    errors: list[str] = []
    seen_ids: dict[str, str] = {}
    for split, split_rows in rows.items():
        if not split_rows:
            errors.append(f"{split}.jsonl is empty")
        for i, r in enumerate(split_rows, 1):
            where = f"{split}.jsonl:{i}"
            for f in dataset_io.LECTURE_REQUIRED_FIELDS:
                if not isinstance(r.get(f), str) or not r[f].strip():
                    errors.append(f"{where}: missing or empty {f!r}")
            clash = RESERVED & (set(r) - {"split"})
            if clash:
                errors.append(f"{where}: uses reserved field(s) {sorted(clash)}")
            if "split" in r and r["split"] != split:
                errors.append(f"{where}: split field {r['split']!r} does not match the file")
            rid = str(r.get("id", ""))
            if rid in seen_ids:
                errors.append(f"{where}: id {rid!r} already used in {seen_ids[rid]}")
            seen_ids[rid] = split
            if require_status and r.get("review_status") != require_status:
                errors.append(f"{where}: review_status {r.get('review_status')!r} is not {require_status!r}")
            if isinstance(r.get("summary"), str) and isinstance(r.get("source_text"), str) \
                    and len(r["summary"].split()) >= len(r["source_text"].split()):
                errors.append(f"{where}: summary is not shorter than its source_text")
    for field in ("source_text", "summary"):
        where_seen: dict[str, str] = {}
        for split, split_rows in rows.items():
            for r in split_rows:
                key = norm(r.get(field, ""))
                if key and key in where_seen:
                    errors.append(f"duplicate {field} in {where_seen[key]} and {split} ({r.get('id')})")
                where_seen.setdefault(key, f"{split} ({r.get('id')})")
    near = {"performed": False}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        flat = [(s, r) for s in SPLITS for r in rows[s]]
        if len(flat) > 1:
            sim = cosine_similarity(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
                                    .fit_transform([r["source_text"] for _, r in flat]))
            worst = 0.0
            for i in range(len(flat)):
                for j in range(i + 1, len(flat)):
                    if flat[i][0] != flat[j][0]:
                        worst = max(worst, float(sim[i, j]))
                        if sim[i, j] >= near_dup:
                            errors.append(f"near-duplicate source_text across splits: {flat[i][1]['id']} "
                                          f"({flat[i][0]}) / {flat[j][1]['id']} ({flat[j][0]}) cosine {sim[i, j]:.2f}")
            near = {"performed": True, "threshold": near_dup, "max_cross_split_similarity": round(worst, 4)}
    except ImportError:
        near = {"performed": False, "reason": "scikit-learn not installed"}
    if errors:
        sys.exit("Summarization source rejected:\n  " + "\n  ".join(errors[:50])
                 + (f"\n  ... {len(errors) - 50} more" if len(errors) > 50 else ""))
    return rows, near


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=paths.TASK_SEPARATED_DATA_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-review-status", default="verified")
    ap.add_argument("--near-dup-threshold", type=float, default=0.80)
    ap.add_argument("--check-tokens", action="store_true")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--config-name", default=None, help="new config file name (without .json)")
    args = ap.parse_args()
    source, base, out = resolve(args.source), resolve(args.base), resolve(args.out)
    for label, path in (("--source", source), ("--out", out), ("--base", base)):
        if not path.is_relative_to(REPO_ROOT):
            sys.exit(f"{label} must be inside the repository ({REPO_ROOT}); the loader resolves it from there.")
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite.")
    if args.write_config and not args.config_name:
        sys.exit("--write-config needs --config-name")
    new_cfg_path = paths.CONFIG_DIR / f"{args.config_name}.json" if args.config_name else None
    if args.write_config and new_cfg_path.exists():
        sys.exit(f"{new_cfg_path} already exists; refusing to overwrite.")

    # 1. verify inputs
    # the base contributes only bloom_rewrite/ and qa/ (byte-for-byte, hash-checked below); its old summarization
    # source is replaced, so that source's fingerprint is not required to match
    base_integrity = dataset_io.validate_separate_dataset(base, check_summarization_source=False)
    base_manifest = dataset_io.load_manifest(base)
    source_rows, near = check_source(source, args.require_review_status, args.near_dup_threshold)
    source_hashes = {name: dataset_io.sha256_file(source / name) for name in dataset_io.LECTURE_SOURCE_FILES}
    out.mkdir(parents=True)
    try:
        cfg_note = build(args, source, base, out, base_manifest, base_integrity, source_rows, near, source_hashes,
                         new_cfg_path)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)  # never leave a half-built dataset behind
        raise
    print("\nTo make this the active dataset, set in experiments/multitask_bloom_rewrite/paths.py:")
    print(f'  TASK_SEPARATED_DATA_DIR = DATA_DIR / "{out.name}"')
    if cfg_note:
        print(f'  ACTIVE_TRAINING_CONFIG = "{new_cfg_path.name}"')


def build(args, source, base, out, base_manifest, base_integrity, source_rows, near, source_hashes, new_cfg_path):
    # 2. summarization rows (lecture format)
    hashes, counts, row_hashes, stats = {}, {s: {} for s in SPLITS}, {}, {}
    (out / "summarization").mkdir()
    for split in SPLITS:
        records = []
        for r in source_rows[split]:
            rec = {"task": "summarization", "split": split, "summarization_format": dataset_io.SUM_LECTURE}
            rec.update({k: v for k, v in r.items() if k != "split"})
            rec["article"], rec["abstract"] = r["source_text"], r["summary"]
            rec["sft_text"] = build_sft_text("summarization", rec)
            rec["prompt_text"] = build_prompt_only_text("summarization", rec)
            rec["generation_prompt"] = build_generation_prompt("summarization", rec)
            assert rec["sft_text"].startswith(rec["prompt_text"])
            assert rec["sft_text"].endswith(r["summary"].strip() + "<|im_end|>")
            records.append(rec)
        path = out / "summarization" / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        hashes[f"summarization/{split}.jsonl"] = dataset_io.sha256_file(path)
        counts[split]["summarization"] = len(records)
        row_hashes[split] = {rec["id"]: dataset_io._row_sha256(rec) for rec in records}
        words = [len(r["summary"].split()) / max(1, len(r["source_text"].split())) for r in source_rows[split]]
        stats[split] = {
            "rows": len(records),
            "department": dict(Counter(str(r.get("department", "unspecified")) for r in records).most_common()),
            "source_type": dict(Counter(str(r.get("source_type", "unspecified")) for r in records).most_common()),
            "review_status": dict(Counter(str(r.get("review_status", "unspecified")) for r in records).most_common()),
            "summary_to_source_word_ratio": {"min": round(min(words), 3), "max": round(max(words), 3),
                                             "mean": round(sum(words) / len(words), 3)},
        }

    # 3. bloom_rewrite and qa: byte-for-byte copies of the verified base dataset
    for task in COPIED:
        (out / task).mkdir()
        for split in SPLITS:
            name = f"{task}/{split}.jsonl"
            shutil.copy2(base / name, out / name)
            hashes[name] = dataset_io.sha256_file(out / name)
            if hashes[name] != base_manifest["generated_files_sha256"][name]:
                sys.exit(f"copied {name} differs from the base dataset")
            counts[split][task] = base_manifest["counts"][split][task]

    # 4. optional tokenizer length check (CPU, offline)
    token_check = {"performed": False}
    if args.check_tokens:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        token_check = {"performed": True, "model_id": MODEL_ID, "max_seq_length": args.max_seq_length,
                       "summarization": {}}
        for split in SPLITS:
            rows = dataset_io.read_jsonl(out / "summarization" / f"{split}.jsonl")
            seq = [len(tok(r["sft_text"], add_special_tokens=False)["input_ids"]) for r in rows]
            tgt = [len(tok(r["abstract"], add_special_tokens=False)["input_ids"]) for r in rows]
            over = [r["id"] for r, n in zip(rows, seq) if n > args.max_seq_length]
            token_check["summarization"][split] = {"rows": len(rows), "max_sequence_tokens": max(seq),
                                                   "max_summary_tokens": max(tgt), "overlength": len(over)}
            if over:
                sys.exit(f"summarization/{split}: {len(over)} sequences exceed {args.max_seq_length} tokens, "
                         f"e.g. {over[:5]}; nothing was written.")

    # 5. manifest: arXiv summarization records are dropped entirely
    manifest = {k: v for k, v in base_manifest.items() if k not in (
        "timestamp_utc", "generated_files_sha256", "counts", "notes", "derived_from", "summarization_source",
        "summarization_quality_filter", "upstream_datasets")}
    manifest.update({
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "summarization_source": {
            "kind": dataset_io.SUM_LECTURE_SOURCE_KIND,
            "path": rel(source),
            "format": "lecture: system + 'Lecture material:' user turn -> study summary",
            "files_sha256": source_hashes,
            "required_review_status": args.require_review_status or None,
            "near_duplicate_check": near,
            "row_sha256": row_hashes,
            "stats": stats,
            "replaces": "all previous summarization rows (no arXiv data is used)",
        },
        "tokenizer": {**base_manifest.get("tokenizer", {}), "summarization_check": token_check},
        "derived_from": {"dataset_dir": rel(base), "manifest_sha256": base_integrity["manifest_sha256"],
                         "copied_unchanged": [f"{t}/*" for t in COPIED]},
        "generated_files_sha256": dict(sorted(hashes.items())),
        "counts": counts,
        "notes": [
            "No combined train.jsonl, validation.jsonl, or test.jsonl is persisted.",
            "summarization is the owner-built lecture dataset; bloom_rewrite and qa are unchanged from the base.",
        ],
    })
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    integrity = dataset_io.validate_separate_dataset(out)

    # 6. optional config
    cfg_note = None
    if args.write_config:
        cfg = json.loads((paths.CONFIG_DIR / paths.ACTIVE_TRAINING_CONFIG).read_text(encoding="utf-8"))
        cfg["experiment_name"] = f"multitask_{args.config_name}"
        cfg["dataset_dir"] = rel(out)
        cfg["output_dir"] = f"experiments/multitask_bloom_rewrite/models/{args.config_name}"
        cfg["results_dir"] = f"experiments/multitask_bloom_rewrite/results/{args.config_name}"
        cfg["validation_task_limits"] = dict(counts["validation"])
        cfg["test_task_limits"] = dict(counts["test"])
        if token_check["performed"]:
            longest = max(v["max_summary_tokens"] for v in token_check["summarization"].values())
            gen = cfg["generation"]["max_new_tokens_by_task"]
            gen["summarization"] = max(int(gen.get("summarization", 256)), int(math.ceil(longest * 1.25 / 16) * 16))
        cfg["notes"] = [
            f"Generated by attach_summarization_dataset.py from {paths.ACTIVE_TRAINING_CONFIG}.",
            f"dataset_dir -> {rel(out)}: owner-built lecture summarization replaces the previous summarization task.",
            "validation/test limits equal the dataset counts; summarization max_new_tokens covers the longest summary "
            "with 25% headroom (when --check-tokens ran). Model, LoRA, optimizer, epochs and task weights unchanged.",
        ]
        new_cfg_path.write_text(json.dumps(cfg, indent="\t", ensure_ascii=False) + "\n", encoding="utf-8")
        cfg_note = rel(new_cfg_path)

    print(json.dumps({"out": rel(out), "counts": integrity["counts"],
                      "summarization_source_check": integrity["summarization_source_check"],
                      "near_duplicate_check": near, "token_check": token_check, "config": cfg_note}, indent=1))
    return cfg_note


if __name__ == "__main__":
    main()
