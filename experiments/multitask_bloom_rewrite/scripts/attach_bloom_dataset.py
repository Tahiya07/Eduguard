"""Attach an EduGuard Bloom rewrite source (e.g. data/bloom_rewrite_eduguard_v3_2) to a task-separated SFT dataset.

  python -I experiments/multitask_bloom_rewrite/scripts/attach_bloom_dataset.py \
      --source data/bloom_rewrite_eduguard_v3_2 --base data/multitask_bloom_rewrite_separate_eduguard_v3_6 \
      --out data/multitask_bloom_rewrite_separate_eduguard_v3_7 --check-tokens \
      --write-config --config-name qwen15b_multitask_eduguard_v3_7_20261126

The base's qa/ and summarization/ files (and summarization eligibility files, if any) are copied byte-for-byte; its
bloom_rewrite/ is replaced by the source rows (contested rows excluded and listed in bloom_exclusions), each rendered
with prompts.build_sft_text / build_prompt_only_text / build_generation_prompt. The result is re-validated with
dataset_io.validate_separate_dataset, which re-checks every Bloom row against the source files. With --write-config
the active config is copied with dataset_dir, output/results dirs, validation/test limits and (after --check-tokens)
the bloom_rewrite generation length updated; base_results_dir is dropped (the Bloom held-out set changed).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
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


def rel(p: Path) -> str:
    return p.resolve().relative_to(REPO_ROOT).as_posix()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check-tokens", action="store_true")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--config-name")
    args = ap.parse_args()
    source, base, out = (REPO_ROOT / p for p in (args.source, args.base, args.out))
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    new_cfg = paths.CONFIG_DIR / f"{args.config_name}.json" if args.config_name else None
    if args.write_config and (not new_cfg or new_cfg.exists()):
        sys.exit("--write-config needs a new --config-name")
    dataset_io.validate_separate_dataset(base)
    base_manifest = dataset_io.load_manifest(base)
    src_manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))

    out.mkdir(parents=True)
    try:
        hashes, counts, stats, excluded = {}, {s: {} for s in SPLITS}, {}, []
        (out / "bloom_rewrite").mkdir()
        for s in SPLITS:
            rows = dataset_io.read_jsonl(source / f"{s}.jsonl")
            kept = []
            for r in rows:
                if r.get("gold_label_contested") is True:
                    excluded.append({"id": r["id"], "split": s, "source_id": r["source_id"],
                                     "target_bloom_level": r["target_bloom_level"]})
                    continue
                rec = dict(r)
                rec["sft_text"] = build_sft_text("bloom_rewrite", rec)
                rec["prompt_text"] = build_prompt_only_text("bloom_rewrite", rec)
                rec["generation_prompt"] = build_generation_prompt("bloom_rewrite", rec)
                assert rec["sft_text"].startswith(rec["prompt_text"])
                kept.append(rec)
            path = out / "bloom_rewrite" / f"{s}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as fh:
                for rec in kept:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            hashes[f"bloom_rewrite/{s}.jsonl"] = dataset_io.sha256_file(path)
            counts[s]["bloom_rewrite"] = len(kept)
            stats[s] = {"source_rows": len(rows), "excluded_contested": len(rows) - len(kept), "rows": len(kept),
                        "direct_rows": sum(r.get("pair_type") == "direct" for r in kept),
                        "pivot_rows": sum(r.get("pair_type") == "pivot" for r in kept),
                        "source_questions": len({r["source_id"] for r in kept}),
                        "source_level_counts": dict(Counter(r["source_bloom_level"] for r in kept)),
                        "target_level_counts": dict(Counter(r["target_bloom_level"] for r in kept))}
        for task in ("qa", "summarization"):
            for f in sorted((base / task).iterdir()):
                (out / task).mkdir(exist_ok=True)
                shutil.copy2(f, out / task / f.name)
                name = f"{task}/{f.name}"
                if name in base_manifest["generated_files_sha256"]:
                    hashes[name] = dataset_io.sha256_file(out / name)
                    if hashes[name] != base_manifest["generated_files_sha256"][name]:
                        sys.exit(f"copied {name} differs from the base")
            for s in SPLITS:
                counts[s][task] = base_manifest["counts"][s][task]

        token_check = {"performed": False}
        if args.check_tokens:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            token_check = {"performed": True, "model_id": MODEL_ID, "max_seq_length": args.max_seq_length,
                           "bloom_rewrite": {}}
            for s in SPLITS:
                rows = dataset_io.read_jsonl(out / "bloom_rewrite" / f"{s}.jsonl")
                seq = [len(tok(r["sft_text"], add_special_tokens=False)["input_ids"]) for r in rows]
                tgt = [len(tok(r["target_rewrite"], add_special_tokens=False)["input_ids"]) for r in rows]
                token_check["bloom_rewrite"][s] = {"rows": len(rows), "max_sequence_tokens": max(seq),
                                                   "max_target_tokens": max(tgt),
                                                   "overlength": sum(n > args.max_seq_length for n in seq)}
                if token_check["bloom_rewrite"][s]["overlength"]:
                    sys.exit(f"bloom_rewrite/{s}: sequences over {args.max_seq_length} tokens")

        bs = dict(base_manifest["bloom_source"])
        bs.update({"path": rel(source), "version": src_manifest.get("version"),
                   "files_sha256": {n: dataset_io.sha256_file(source / n) for n in dataset_io.EDUGUARD_SOURCE_FILES},
                   "derived_from": src_manifest.get("derived_from"), "changes": src_manifest.get("changes")})
        manifest = dict(base_manifest)
        manifest.update({
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "bloom_source": bs,
            "bloom_exclusions": {**base_manifest.get("bloom_exclusions", {}), "excluded_rows": excluded},
            "bloom_rewrite_stats": stats,
            "tokenizer": {**base_manifest.get("tokenizer", {}), "bloom_rewrite_check": token_check}
            if token_check["performed"] else base_manifest.get("tokenizer", {}),
            "derived_from": {"dataset_dir": rel(base), "copied_unchanged": ["qa/*", "summarization/*"],
                             "bloom_rewrite": f"rebuilt from {rel(source)}"},
            "generated_files_sha256": dict(sorted(hashes.items())),
            "counts": counts,
        })
        (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
        integrity = dataset_io.validate_separate_dataset(out)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise

    if args.write_config:
        cfg = json.loads((paths.CONFIG_DIR / paths.ACTIVE_TRAINING_CONFIG).read_text(encoding="utf-8"))
        cfg["experiment_name"] = f"multitask_{args.config_name}"
        cfg["dataset_dir"] = rel(out)
        cfg["output_dir"] = f"experiments/multitask_bloom_rewrite/models/{args.config_name}"
        cfg["results_dir"] = f"experiments/multitask_bloom_rewrite/results/{args.config_name}"
        cfg["validation_task_limits"] = dict(counts["validation"])
        cfg["test_task_limits"] = dict(counts["test"])
        cfg.pop("base_results_dir", None)
        if token_check["performed"]:
            longest = max(token_check["bloom_rewrite"][s]["max_target_tokens"] for s in ("train", "validation"))
            gen = cfg["generation"]["max_new_tokens_by_task"]
            gen["bloom_rewrite"] = max(int(gen.get("bloom_rewrite", 128)), int(math.ceil(longest * 1.25 / 16) * 16))
        cfg["notes"] = list(cfg.get("notes", [])) + [
            f"Generated by attach_bloom_dataset.py from {paths.ACTIVE_TRAINING_CONFIG}: bloom_rewrite rebuilt from "
            f"{rel(source)}; qa and summarization unchanged. No base_results_dir: base is re-evaluated.",
        ]
        new_cfg.write_text(json.dumps(cfg, indent="\t", ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": rel(out), "counts": integrity["counts"], "bloom_source_check": integrity["bloom_source_check"],
                      "token_check": token_check, "config": rel(new_cfg) if args.write_config else None}, indent=1))


if __name__ == "__main__":
    main()
