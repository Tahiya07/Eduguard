#!/usr/bin/env python
"""Build a leakage-controlled Bloom target-rewrite dataset.

Figshare Bloom data is a CLASSIFICATION corpus. It is used only as a
source-question pool. Target rewrites are produced by a deterministic
transformation framework, never by Qwen2.5-0.5B or Qwen2.5-1.5B.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bloom_target_policy import (  # noqa: E402
    BLOOM_LEVELS,
    all_source_target_pairs,
    canonical_level,
    source_is_usable,
    synthesize_rewrite,
    validate_rewrite,
)
from grouping import GroupingThresholds, group_questions, normalize_question  # noqa: E402
from paths import (  # noqa: E402
    DATASET_VERSION,
    FIGSHARE_COMBINED,
    FIGSHARE_V1,
    HUMAN_EVAL_DIR,
    REWRITE_DATA_DIR,
    SEED,
)
from prompt_format import build_messages, build_sft_text  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def inspect_figshare(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        "filename": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "has_rewrite_pairs": False,
        "rewrite_pair_columns": [
            col
            for col in df.columns
            if str(col).lower() in {
                "target_rewrite",
                "rewritten_question",
                "target_question",
                "rewrite",
            }
        ],
        "dataset_type": "classification_only",
        "label_columns": [col for col in df.columns if "bloom" in str(col).lower() or "label" in str(col).lower() or str(col).upper() == "BT LEVEL"],
        "cannot_supervise_target_rewriting_by_itself": True,
    }


def load_source_pool(figshare_path: Path) -> pd.DataFrame:
    df = pd.read_csv(figshare_path)
    required = {"question", "bloom_level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{figshare_path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["source_question"] = df["question"].astype(str).str.strip()
    df["source_bloom_level"] = df["bloom_level"].map(canonical_level)
    df["origin_row"] = df.index.astype(int)
    df["source_file"] = str(figshare_path.relative_to(REPO_ROOT)).replace("\\", "/")
    return df[["source_question", "source_bloom_level", "origin_row", "source_file", "original_label"] if "original_label" in df.columns else ["source_question", "source_bloom_level", "origin_row", "source_file"]]


def filter_sources(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keep_flags = []
    reasons = Counter()
    for _, row in df.iterrows():
        ok, reason = source_is_usable(row["source_question"], row["source_bloom_level"])
        keep_flags.append(ok)
        reasons[reason] += 1
    filtered = df.loc[keep_flags].reset_index(drop=True)
    return filtered, {
        "input_rows": int(len(df)),
        "kept_rows": int(len(filtered)),
        "dropped_rows": int(len(df) - len(filtered)),
        "drop_reasons": dict(reasons),
    }


def assign_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    thresholds = GroupingThresholds()
    group_ids = group_questions(df["source_question"].tolist(), thresholds)
    out = df.copy()
    out["group_id"] = group_ids
    sizes = Counter(group_ids)
    return out, {
        "n_questions": int(len(out)),
        "n_groups": int(len(sizes)),
        "largest_group": int(max(sizes.values())),
        "singleton_groups": int(sum(1 for size in sizes.values() if size == 1)),
        "thresholds": thresholds.__dict__,
    }


def split_groups(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = random.Random(seed)
    groups = sorted(df["group_id"].unique().tolist())
    by_level: dict[str, list[int]] = defaultdict(list)
    for group_id, group_df in df.groupby("group_id"):
        majority = group_df["source_bloom_level"].value_counts().idxmax()
        by_level[str(majority)].append(int(group_id))

    train, val, test = set(), set(), set()
    notes = []
    for level in BLOOM_LEVELS:
        ids = by_level.get(level, [])
        rng.shuffle(ids)
        if len(ids) < 6:
            notes.append(
                f"Source level {level} has only {len(ids)} groups; split is still "
                "group-held-out but stratification is coarse."
            )
        n = len(ids)
        n_test = max(1, int(round(n * 0.15))) if n >= 3 else (1 if n == 2 else 0)
        n_val = max(1, int(round(n * 0.15))) if n - n_test >= 3 else (1 if n - n_test >= 2 else 0)
        test.update(ids[:n_test])
        val.update(ids[n_test:n_test + n_val])
        train.update(ids[n_test + n_val:])

    leftover = [gid for gid in groups if gid not in train | val | test]
    rng.shuffle(leftover)
    for i, gid in enumerate(leftover):
        (train, val, test)[i % 3].add(gid)

    mapping = {}
    for gid in train:
        mapping[gid] = "train"
    for gid in val:
        mapping[gid] = "validation"
    for gid in test:
        mapping[gid] = "test"

    out = df.copy()
    out["split"] = out["group_id"].map(mapping)
    if out["split"].isna().any():
        raise RuntimeError("Some groups were not assigned a split.")

    # Leakage assertion: no normalized question shared across splits.
    split_norms = {name: set() for name in ("train", "validation", "test")}
    for _, row in out.iterrows():
        split_norms[row["split"]].add(normalize_question(row["source_question"]))
    overlaps = {
        "train_validation": len(split_norms["train"] & split_norms["validation"]),
        "train_test": len(split_norms["train"] & split_norms["test"]),
        "validation_test": len(split_norms["validation"] & split_norms["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Normalized source leakage across splits: {overlaps}")

    group_splits = out.groupby("group_id")["split"].nunique()
    if int(group_splits.max()) != 1:
        raise RuntimeError("A question family was split across partitions.")

    return out, {
        "groups_per_split": {k: int(v) for k, v in Counter(mapping.values()).items()},
        "sources_per_split": out["split"].value_counts().to_dict(),
        "normalized_overlaps": overlaps,
        "notes": notes,
        "seed": seed,
        "ratios": "approximately 70/15/15 by group, stratified on majority source Bloom level",
    }


def build_examples(df: pd.DataFrame) -> tuple[list[dict], dict]:
    pairs = all_source_target_pairs()
    examples: list[dict] = []
    rejected = Counter()
    matrix = Counter()
    split_matrix = defaultdict(Counter)
    for _, row in df.iterrows():
        source_level = row["source_bloom_level"]
        source_q = row["source_question"]
        source_id = f"src_{int(row['origin_row']):05d}_{sha256_text(source_q)[:10]}"
        for target_level in BLOOM_LEVELS:
            if target_level == source_level:
                continue
            transformation = f"{source_level}->{target_level}"
            rewrite = synthesize_rewrite(source_q, source_level, target_level, source_id)
            validation = validate_rewrite(source_q, rewrite, target_level)
            if not validation.ok:
                rejected[validation.failure_category or "OTHER"] += 1
                continue
            example_id = sha256_text(f"{source_id}|{transformation}|{rewrite}")[:16]
            messages = build_messages(source_q, target_level, rewrite)
            text = build_sft_text(source_q, target_level, rewrite)
            if "Original Bloom level:" in text or "Source Bloom level:" in text:
                raise RuntimeError("source Bloom level leaked into SFT text")
            record = {
                "example_id": example_id,
                "source_id": source_id,
                "group_id": int(row["group_id"]),
                "split": row["split"],
                "source_question": source_q,
                "source_bloom_level": source_level,  # metadata only — not in generator prompt
                "target_bloom_level": target_level,
                "target_rewrite": rewrite,
                "transformation_type": transformation,
                "synthetic_or_original": "synthetic",
                "synthetic": True,
                "quality_status": validation.quality_status,
                "topic_overlap": round(validation.topic_overlap, 4),
                "source_file": row["source_file"],
                "generator_inputs": ["source_question", "target_bloom_level"],
                "messages": messages,
                "text": text,
            }
            examples.append(record)
            matrix[transformation] += 1
            split_matrix[row["split"]][transformation] += 1

    coverage = {
        f"{src}->{tgt}": int(matrix.get(f"{src}->{tgt}", 0))
        for src, tgt in pairs
    }
    missing = [key for key, count in coverage.items() if count == 0]
    return examples, {
        "n_pass_examples": len(examples),
        "n_rejected_syntheses": int(sum(rejected.values())),
        "rejection_categories": dict(rejected),
        "transformation_coverage": coverage,
        "missing_transformations": missing,
        "split_transformation_coverage": {
            split: dict(counts) for split, counts in split_matrix.items()
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_hash(examples: list[dict]) -> str:
    # Include SFT text so prompt-format changes (v1 3-input vs v2 2-input) change the hash.
    lines = sorted(f"{ex['example_id']}|{ex.get('text', '')}" for ex in examples)
    return sha256_text("\n".join(lines))


def sample_human_eval(examples: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    test = [ex for ex in examples if ex["split"] == "test"]
    wanted = [
        "Remember->Understand",
        "Understand->Apply",
        "Apply->Analyze",
        "Analyze->Evaluate",
        "Evaluate->Create",
        "Remember->Create",
        "Remember->Analyze",
        "Understand->Evaluate",
        "Apply->Create",
        "Create->Remember",
        "Create->Apply",
        "Evaluate->Understand",
    ]
    sampled: list[dict] = []
    for trans in wanted:
        pool = [ex for ex in test if ex["transformation_type"] == trans]
        rng.shuffle(pool)
        for ex in pool[:3]:
            sampled.append(
                {
                    "item_id": f"he_{len(sampled)+1:03d}",
                    "source_question": ex["source_question"],
                    "target_bloom_level": ex["target_bloom_level"],
                    "system_output": "",
                    "hidden_system_id": "",
                    "rater_target_bloom_alignment_1to5": "",
                    "rater_topic_preservation_1to5": "",
                    "rater_cognitive_demand_1to5": "",
                    "rater_question_quality_1to5": "",
                    "rater_grammatical_quality_1to5": "",
                    "rater_non_triviality_1to5": "",
                    "notes": "",
                    # Blind metadata — withhold from raters:
                    "example_id": ex["example_id"],
                    "transformation_type": ex["transformation_type"],
                    "source_bloom_level_withheld": ex["source_bloom_level"],
                }
            )
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Bloom target-rewrite dataset")
    parser.add_argument("--source-csv", type=str, default=str(FIGSHARE_V1))
    parser.add_argument("--output-dir", type=str, default=str(REWRITE_DATA_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild in place. Default refuses if dataset_manifest.json already exists.",
    )
    args = parser.parse_args()

    source_csv = Path(args.source_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "dataset_manifest.json").is_file() and not args.overwrite:
        raise SystemExit(
            "Dataset already exists. Refusing silent overwrite. "
            "Use prepare_dataset.py --validate-only, or pass --overwrite / --new-version."
        )

    inspection = inspect_figshare(source_csv)
    combined_inspection = inspect_figshare(FIGSHARE_COMBINED) if FIGSHARE_COMBINED.is_file() else None
    print("PHASE: source inspection")
    print(json.dumps(inspection, indent=2))
    if inspection["has_rewrite_pairs"]:
        raise SystemExit("Unexpected rewrite-pair columns found; inspect before synthesizing.")

    raw = load_source_pool(source_csv)
    filtered, filter_stats = filter_sources(raw)
    print("PHASE: source quality filter", json.dumps(filter_stats))
    grouped, group_stats = assign_groups(filtered)
    print("PHASE: grouping", json.dumps({k: v for k, v in group_stats.items() if k != "thresholds"}))
    split_df, split_stats = split_groups(grouped, args.seed)
    print("PHASE: split", json.dumps({k: v for k, v in split_stats.items() if k != "notes"}))
    examples, synth_stats = build_examples(split_df)
    print("PHASE: synthesis", json.dumps({k: v for k, v in synth_stats.items() if k != "split_transformation_coverage"}))

    by_split = defaultdict(list)
    for ex in examples:
        by_split[ex["split"]].append(ex)
    for split_name in ("train", "validation", "test"):
        write_jsonl(output_dir / f"{split_name}.jsonl", by_split[split_name])

    source_level_counts = split_df["source_bloom_level"].value_counts().to_dict()
    target_level_counts = Counter(ex["target_bloom_level"] for ex in examples)
    ds_hash = dataset_hash(examples)
    statistics = {
        "dataset_version": DATASET_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source_inspection": inspection,
        "combined_inspection": combined_inspection,
        "filter_stats": filter_stats,
        "grouping": group_stats,
        "split": split_stats,
        "synthesis": {
            "n_pass_examples": synth_stats["n_pass_examples"],
            "n_rejected_syntheses": synth_stats["n_rejected_syntheses"],
            "rejection_categories": synth_stats["rejection_categories"],
            "missing_transformations": synth_stats["missing_transformations"],
        },
        "usable_examples": synth_stats["n_pass_examples"],
        "counts": {
            "train": len(by_split["train"]),
            "validation": len(by_split["validation"]),
            "test": len(by_split["test"]),
        },
        "source_bloom_distribution": source_level_counts,
        "target_bloom_distribution": dict(target_level_counts),
        "transformation_coverage": synth_stats["transformation_coverage"],
        "split_transformation_coverage": synth_stats["split_transformation_coverage"],
        "methodology": {
            "strategy": "B_public_bloom_questions_plus_deterministic_synthetic_transformation",
            "teacher_model": None,
            "candidate_models_used_to_generate_labels": False,
            "self_level_transformations": False,
            "split_unit": "question_family_group",
            "generator_task": "question + target_level → rewrite",
            "source_bloom_level_role": "metadata_only_not_in_generator_prompt",
            "prompt_format_version": "production_aligned_v2",
        },
        "dataset_hash": ds_hash,
    }
    (output_dir / "dataset_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    manifest = {
        "dataset_version": DATASET_VERSION,
        "dataset_hash": ds_hash,
        "seed": args.seed,
        "git_commit": git_commit(),
        "source_files": [
            {
                "path": inspection["filename"],
                "sha256": sha256_file(source_csv),
                "role": "classification_source_pool_not_rewrite_supervision",
            }
        ],
        "outputs": {
            "train": str((output_dir / "train.jsonl").as_posix()),
            "validation": str((output_dir / "validation.jsonl").as_posix()),
            "test": str((output_dir / "test.jsonl").as_posix()),
        },
        "prompt_format": "qwen2.5_instruct_chatml_question_plus_target_only",
        "generator_task": "question + target_level → rewrite",
        "source_bloom_level_role": "metadata_only",
        "notes": [
            "Figshare is a Bloom CLASSIFICATION dataset and does not contain original_question -> target_level -> target_rewrite pairs.",
            "Synthetic targets were generated by a deterministic policy/template framework, not by Qwen2.5-0.5B or Qwen2.5-1.5B.",
            "Existing figshare_bloom_v1_{train,val,test}.csv splits were NOT reused because they are classification splits and have normalized overlaps.",
            "BEFORE: question + source Bloom + target Bloom → rewrite. AFTER: question + target Bloom → rewrite (source Bloom is metadata only).",
            "Generator prompts must never contain 'Original Bloom level:' or 'Source Bloom level:'.",
        ],
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    human_items = sample_human_eval(examples, args.seed)
    human_path = HUMAN_EVAL_DIR / "blinded_eval_items.csv"
    if human_items:
        with human_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(human_items[0].keys()))
            writer.writeheader()
            writer.writerows(human_items)
    (HUMAN_EVAL_DIR / "README.md").write_text(
        "Human scoring remains pending. Raters see ONLY:\n"
        "  - original question\n"
        "  - target Bloom level\n"
        "  - generated rewrite\n"
        "Do NOT show source Bloom level or model identity (0.5B vs 1.5B).\n"
        "Rate on a 1–5 scale: target Bloom alignment, topic preservation, "
        "cognitive-demand appropriateness, question quality, grammatical quality, "
        "non-triviality.\n",
        encoding="utf-8",
    )

    print("Wrote", output_dir)
    print("usable_examples", statistics["usable_examples"])
    print("counts", statistics["counts"])
    print("missing_transformations", statistics["synthesis"]["missing_transformations"])
    print("dataset_hash", ds_hash)


if __name__ == "__main__":
    main()
