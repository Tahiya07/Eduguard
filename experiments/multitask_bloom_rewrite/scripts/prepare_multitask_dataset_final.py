#!/usr/bin/env python
"""Prepare the final 1.5B multitask generator corpus.

Final generator tasks:
  1. Bloom transformation from the canonical BloomShift final candidate corpus.
  2. Extractive/grounded QA from SQuAD 1.1.
  3. Summarization from ccdv/arxiv-summarization.

The 0.5B Bloom-classification dataset is deliberately excluded.

Preparation is online/offline-cache capable: datasets are downloaded only during
dataset preparation. The resulting JSONL corpus is self-contained for training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "multitask_bloom_rewrite"
if str(EXPERIMENT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(EXPERIMENT_DIR))
DEFAULT_BLOOM_DIR = REPO_ROOT / "data" / "rewrite dataset" / "bloomshift_final_candidate"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "multitask_bloom_rewrite_separate"

TASK_BLOOM = "bloom_rewrite"
TASK_QA = "qa"
TASK_SUM = "summarization"
ALLOWED_TASKS = {TASK_BLOOM, TASK_QA, TASK_SUM}
UPSTREAM_DATASET_INFO: dict[str, Any] = {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return data


def validate_annotation_provenance(
    bloom_dir: Path,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the owner batch-attestation scope without inventing reviewer scores.

    The batch attestation permits all source rows to be included in the modeling
    workflow. Independent dual-review evidence remains a separate, explicit status.
    """
    attestation_path = bloom_dir / "batch_review_attestation.json"
    if not attestation_path.is_file():
        raise ValueError(
            "Missing batch_review_attestation.json. Record the dataset owner's "
            "all-row inclusion decision before rebuilding the corpus."
        )
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read batch attestation {attestation_path}: {exc}") from exc

    if attestation.get("processing_policy", {}).get("exclusions") is None:
        raise ValueError("Batch attestation lacks an explicit all-record inclusion policy.")
    scope = attestation.get("scope", {})
    declared_counts = scope.get("split_counts", {})
    actual_counts = {
        "train": len(train_rows),
        "validation": len(validation_rows),
        "test": len(test_rows),
    }
    if not scope.get("include_every_canonical_record") or declared_counts != actual_counts:
        raise ValueError(
            "Canonical BloomShift rows do not match the all-row batch attestation: "
            f"declared={declared_counts}, actual={actual_counts}"
        )
    if int(scope.get("total_rows", -1)) != sum(actual_counts.values()):
        raise ValueError("Batch attestation total_rows does not match the canonical split totals.")

    reviewer_paths = [bloom_dir / "reviewer_1.csv", bloom_dir / "reviewer_2.csv"]
    reviewer_hashes: dict[str, str] = {}
    ratings_by_reviewer: list[dict[str, dict[str, str]]] = []
    for path in reviewer_paths:
        if not path.is_file():
            ratings_by_reviewer.append({})
            continue
        reviewer_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        ratings_by_reviewer.append({
            str(row.get("example_id") or "").strip(): row
            for row in csv_rows
            if str(row.get("example_id") or "").strip()
        })

    rating_fields = (
        "bloom_aligned", "reviewer_bloom_level", "content_preserved",
        "context_preserved", "meaningful_transformation", "pedagogically_valid",
        "clear_and_grammatical", "final_decision",
    )
    heldout_rows = validation_rows + test_rows
    incomplete_annotations = 0
    for row in heldout_rows:
        row_id = str(row.get("example_id") or row.get("id") or "").strip()
        for reviewer_rows in ratings_by_reviewer:
            rating = reviewer_rows.get(row_id)
            if not rating or any(not str(rating.get(field) or "").strip() for field in rating_fields):
                incomplete_annotations += 1

    independent_files = (
        len(reviewer_hashes) == 2
        and reviewer_hashes.get("reviewer_1.csv") != reviewer_hashes.get("reviewer_2.csv")
    )
    independent_verified = independent_files and incomplete_annotations == 0
    warnings = []
    if not independent_files:
        warnings.append(
            "Independent reviewer files are missing or byte-identical; inter-rater reliability is not established."
        )
    if incomplete_annotations:
        warnings.append(
            f"{incomplete_annotations} held-out reviewer records lack all required ratings; "
            "held-out targets are usable for provisional model diagnostics, not independent gold-label claims."
        )

    return {
        "inclusion_basis": "dataset_owner_batch_attestation",
        "owner_reported_all_rows_human_reviewed": bool(
            attestation.get("dataset_owner_report", {}).get("all_records_reported_human_reviewed")
        ),
        "all_source_rows_included": True,
        "source_split_counts": actual_counts,
        "reviewer_file_sha256": reviewer_hashes,
        "independent_reviewer_files": independent_files,
        "heldout_rows": len(heldout_rows),
        "incomplete_reviewer_records": incomplete_annotations,
        "independent_review_verified": independent_verified,
        "row_level_ratings_required_fields": list(rating_fields),
        "warnings": warnings,
        "research_claim_limit": (
            "Batch attestation is not independent dual annotation and does not establish inter-rater reliability."
        ),
    }


def normalize_bloom(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    from prompts import build_generation_prompt, build_prompt_only_text, build_sft_text

    out = []
    for row in rows:
        required = (
            "source_id", "group_id", "source_question",
            "target_bloom_level", "target_rewrite", "example_id"
        )
        missing = [k for k in required if k not in row or not str(row.get(k) or "").strip()]
        if missing:
            raise ValueError(
                f"Bloom row in {split!r} is structurally invalid and cannot be silently dropped; "
                f"missing/empty fields {missing}: {row.get('example_id') or row.get('id') or '<unknown-id>'}"
            )

        # Do not expose source review gaps to the model as target labels. Every
        # canonical row is retained under the explicit owner batch attestation.
        # Preserve legacy fields separately so no evidence is fabricated.
        record = {
            "task": TASK_BLOOM,
            "split": split,
            "id": row["example_id"],
            "source_id": row["source_id"],
            "group_id": row["group_id"],
            "source_question": row["source_question"],
            "source_bloom_level": row.get("source_bloom_level"),
            "target_bloom_level": row["target_bloom_level"],
            "target_rewrite": row["target_rewrite"],
            "question_family": row.get("question_family"),
            "transformation_type": row.get("transformation_type"),
            "transformation_strategy": row.get("transformation_strategy"),
            "source_review_status": row.get("review_status") or "unmarked",
            "source_training_eligible": row.get("training_eligible"),
            "review_status": "owner_batch_attested",
            "training_eligible": True,
            "review_provenance_method": "dataset_owner_batch_attestation",
            "independent_review_verified": False,
            "dataset_inclusion_reason": "all canonical BloomShift rows retained; legacy review metadata is not an exclusion filter",
        }
        record["sft_text"] = build_sft_text(TASK_BLOOM, record)
        record["prompt_text"] = build_prompt_only_text(TASK_BLOOM, record)
        record["generation_prompt"] = build_generation_prompt(TASK_BLOOM, record)
        out.append(record)

    if len(out) != len(rows):
        raise AssertionError(
            f"BloomShift {split}: retained {len(out)} of {len(rows)} source rows; no-drop policy violated."
        )
    print(f"BloomShift {split}: included={len(out)}/{len(rows)} under owner batch attestation", flush=True)
    return out


def bounded_token_count(tokenizer: Any, text: str, max_tokens: int) -> int:
    """Count up to max_tokens + 1 tokens without unbounded overlength tokenization."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    return len(tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens + 1,
    )["input_ids"])


def load_squad(
    seed: int,
    tokenizer_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length: int = 8192,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
    max_test_rows: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare SQuAD with context-group isolation and answer-preserving windows.

    The official SQuAD validation split is held out as the QA test split. A
    validation split is carved out of official train by exact context group.
    Seed-shuffled candidates are converted only until each requested cap is met.
    Windowing preserves the exact answer text and character span.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from prompts import build_sft_text

    print("[SQuAD] Loading rajpurkar/squad...", flush=True)
    ds = load_dataset("rajpurkar/squad")
    UPSTREAM_DATASET_INFO[TASK_QA] = {
        "dataset": "rajpurkar/squad",
        "split_fingerprints": {
            name: getattr(ds[name], "_fingerprint", None)
            for name in ("train", "validation")
        },
        "raw_split_rows": {name: len(ds[name]) for name in ("train", "validation")},
        "revision_pinned": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True)
    official_train = ds["train"]
    official_dev = ds["validation"]

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in official_train:
        context_key = str(row.get("context", ""))
        groups.setdefault(context_key, []).append(row)

    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    target_train = int(len(official_train) * 0.95)
    train_raw: list[dict[str, Any]] = []
    val_raw: list[dict[str, Any]] = []
    for _, group_rows in group_items:
        if len(train_raw) < target_train:
            train_raw.extend(group_rows)
        else:
            val_raw.extend(group_rows)

    # Group isolation is more important than an exact 95/5 row ratio. Record
    # the realized ratio in the manifest rather than claiming it is exact.
    if not train_raw or not val_raw:
        raise ValueError(
            f"SQuAD grouped split failed: train={len(train_raw)}, validation={len(val_raw)}"
        )

    caps = {"train": max_train_rows, "validation": max_validation_rows, "test": max_test_rows}
    split_offsets = {"train": 101, "validation": 303, "test": 505}

    def convert(rows: Any, split: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cap = caps[split]
        total = len(rows)
        indices = list(range(total))
        random.Random(seed + split_offsets[split]).shuffle(indices)
        print(
            f"[SQuAD] Converting {split}: {total:,} candidates; "
            f"target={'all' if cap is None else cap}",
            flush=True,
        )
        for row_index, source_index in enumerate(indices):
            if cap is not None and len(out) >= cap:
                break
            row = rows[int(source_index)]
            if (row_index + 1) % 1000 == 0:
                print(
                    f"[SQuAD] {split}: examined {row_index + 1:,}/{total:,}; "
                    f"retained={len(out):,}/{cap if cap is not None else 'all'}",
                    flush=True,
                )
            answers = row.get("answers", {})
            texts = answers.get("text", [])
            starts = answers.get("answer_start", [])
            context = str(row.get("context", ""))
            question = str(row.get("question", "")).strip()
            answer = str(texts[0]) if texts else ""
            answer_start = int(starts[0]) if starts else None
            if not context.strip() or not question or not answer.strip():
                continue
            if answer_start is None or context[answer_start:answer_start + len(answer)] != answer:
                raise ValueError(
                    f"SQuAD answer span mismatch for {row.get('id')}: "
                    f"start={answer_start}, answer={answer!r}"
                )

            record = {
                "task": TASK_QA,
                "split": split,
                "id": str(row["id"]),
                "context": context,
                "question": question,
                "answer": answer,
                "answer_start": answer_start,
                "title": row.get("title"),
            }
            full_text = build_sft_text(TASK_QA, record)
            if bounded_token_count(tokenizer, full_text, max_seq_length) > max_seq_length:
                # Keep the exact answer substring and progressively shrink only
                # surrounding context until the entire prompt + target fits.
                answer_end = answer_start + len(answer)
                left_radius = answer_start
                right_radius = len(context) - answer_end
                radius = max(left_radius, right_radius)
                fitted = False
                while True:
                    left = max(0, answer_start - radius)
                    right = min(len(context), answer_end + radius)
                    candidate_context = context[left:right]
                    candidate_start = answer_start - left
                    if candidate_context[candidate_start:candidate_start + len(answer)] != answer:
                        raise ValueError(
                            f"Answer text changed during SQuAD windowing for {row['id']}."
                        )
                    record["context"] = candidate_context
                    record["answer_start"] = candidate_start
                    candidate_text = build_sft_text(TASK_QA, record)
                    if bounded_token_count(tokenizer, candidate_text, max_seq_length) <= max_seq_length:
                        fitted = True
                        break
                    if radius == 0:
                        break
                    radius = max(0, radius // 2)

                if not fitted:
                    raise ValueError(
                        f"SQuAD example {row['id']} cannot fit max_seq_length={max_seq_length} "
                        "even with answer-only context. Increase the limit or explicitly revise "
                        "the dataset policy; this example was not silently truncated."
                    )

            final_text = build_sft_text(TASK_QA, record)
            final_len = bounded_token_count(tokenizer, final_text, max_seq_length)
            if final_len > max_seq_length:
                raise AssertionError(f"Internal error: SQuAD row {row['id']} exceeds token budget.")
            if record["context"][record["answer_start"]:record["answer_start"] + len(answer)] != answer:
                raise AssertionError(f"Internal error: SQuAD answer span invalid for {row['id']}.")
            out.append(record)
        return out

    print(f"[SQuAD] Grouped source split: train={len(train_raw):,}, validation={len(val_raw):,}; official validation/test={len(official_dev):,}", flush=True)
    result = (
        convert(train_raw, "train"),
        convert(val_raw, "validation"),
        convert(official_dev, "test"),
    )
    for split, rows in zip(("train", "validation", "test"), result):
        cap = caps[split]
        if cap is not None and len(rows) < cap:
            raise ValueError(
                f"SQuAD {split} produced {len(rows)} valid rows, below requested cap {cap}; "
                "inspect source availability and token-budget filtering."
            )
    print(f"[SQuAD] Conversion complete: train={len(result[0]):,}, validation={len(result[1]):,}, test={len(result[2]):,}", flush=True)
    return result

def load_arxiv_summarization(
    tokenizer_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length: int = 8192,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
    max_test_rows: int | None = None,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Select a deterministic capped subset of valid arXiv rows.

    Seed-shuffle each official split and stop tokenizing once its requested number
    of sequence-budget-compliant examples has been collected.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from prompts import build_sft_text

    print("[arXiv] Loading ccdv/arxiv-summarization...", flush=True)
    ds = load_dataset("ccdv/arxiv-summarization")
    required_splits = ("train", "validation", "test")
    missing_splits = [name for name in required_splits if name not in ds]
    if missing_splits:
        raise ValueError(
            "ccdv/arxiv-summarization is missing required official splits: "
            + ", ".join(missing_splits)
        )

    UPSTREAM_DATASET_INFO[TASK_SUM] = {
        "dataset": "ccdv/arxiv-summarization",
        "split_fingerprints": {
            name: getattr(ds[name], "_fingerprint", None)
            for name in required_splits
        },
        "raw_split_rows": {name: len(ds[name]) for name in required_splits},
        "revision_pinned": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True)
    caps = {
        "train": max_train_rows,
        "validation": max_validation_rows,
        "test": max_test_rows,
    }
    excluded = {"train": 0, "validation": 0, "test": 0}
    available = {"train": 0, "validation": 0, "test": 0}

    def convert(split_rows: Any, split: str) -> list[dict[str, Any]]:
        cap = caps[split]
        split_seed = seed + {"train": 202, "validation": 404, "test": 606}[split]
        # Shuffle the Hugging Face Dataset index mapping, not the article text.
        # This permits early stopping without materializing a Python copy of the corpus.
        candidates = split_rows.shuffle(seed=split_seed) if cap is not None else split_rows
        retained: list[dict[str, Any]] = []
        eligible_seen = 0
        examined = 0
        total = len(split_rows)
        print(
            f"[arXiv] Processing {split}: {total:,} source rows; "
            f"target={'all' if cap is None else cap}; seeded_order={cap is not None}",
            flush=True,
        )
        for i, row in enumerate(candidates):
            if cap is not None and len(retained) >= cap:
                break
            examined += 1
            article = str(row.get("article", "")).strip()
            abstract = str(row.get("abstract", "")).strip()
            if article and abstract:
                record = {
                    "task": TASK_SUM,
                    "split": split,
                    "id": str(row.get("id") or f"arxiv_{split}_{i:07d}"),
                    "article": article,
                    "abstract": abstract,
                    "title": str(row.get("title", "")).strip(),
                }
                full_len = bounded_token_count(
                    tokenizer, build_sft_text(TASK_SUM, record), max_seq_length
                )
                if full_len > max_seq_length:
                    excluded[split] += 1
                else:
                    eligible_seen += 1
                    retained.append(record)
            if examined % 1000 == 0:
                print(
                    f"[arXiv] {split}: examined={examined:,}/{total:,}; "
                    f"eligible={eligible_seen:,}; over_budget={excluded[split]:,}; "
                    f"retained={len(retained):,}/{cap if cap is not None else 'all'}",
                    flush=True,
                )
        available[split] = eligible_seen
        retained.sort(key=lambda item: str(item.get("id", "")))
        print(
            f"[arXiv] {split} complete: examined={examined:,}/{total:,}, "
            f"eligible_retained={eligible_seen:,}, excluded_over_budget={excluded[split]:,}, "
            f"retained={len(retained):,}",
            flush=True,
        )
        if cap is not None and len(retained) < cap:
            raise ValueError(
                f"arXiv {split} produced only {len(retained)} valid rows, below requested cap {cap}. "
                "The source split may not contain enough examples within the sequence budget."
            )
        return retained

    train_rows = convert(ds["train"], "train")
    val_rows = convert(ds["validation"], "validation")
    test_rows = convert(ds["test"], "test")
    return train_rows, val_rows, test_rows, excluded, available

def enrich_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from prompts import build_generation_prompt, build_prompt_only_text, build_sft_text

    for row in rows:
        row["sft_text"] = build_sft_text(row["task"], row)
        row["prompt_text"] = build_prompt_only_text(row["task"], row)
        row["generation_prompt"] = build_generation_prompt(row["task"], row)
    return rows


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row["task"] for row in rows))




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare BloomShift, SQuAD, and arXiv as separate task-specific JSONL files."
    )
    parser.add_argument("--bloom-dir", default=str(DEFAULT_BLOOM_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument(
        "--max-qa-train",
        type=int,
        default=None,
        help="Optional deterministic cap on QA training rows; evaluation QA splits are capped to the corresponding BloomShift split size.",
    )
    parser.add_argument(
        "--max-sum-train",
        type=int,
        default=None,
        help="Optional deterministic cap on summarization training rows; evaluation summarization splits are capped to the corresponding BloomShift split size.",
    )
    parser.add_argument(
        "--max-qa-validation", type=int, default=None,
        help="Optional QA validation cap; never exceeds BloomShift validation size.",
    )
    parser.add_argument(
        "--max-qa-test", type=int, default=None,
        help="Optional QA test cap; never exceeds BloomShift test size.",
    )
    parser.add_argument(
        "--max-sum-validation", type=int, default=None,
        help="Optional summarization validation cap; never exceeds BloomShift validation size.",
    )
    parser.add_argument(
        "--max-sum-test", type=int, default=None,
        help="Optional summarization test cap; never exceeds BloomShift test size.",
    )
    args = parser.parse_args()

    bloom_dir = Path(args.bloom_dir)
    if not bloom_dir.is_absolute():
        bloom_dir = REPO_ROOT / bloom_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    # Refuse a legacy merged corpus before downloading/tokenizing upstream datasets.
    stale_merged = [
        out_dir / name
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
        if (out_dir / name).is_file()
    ]
    if stale_merged:
        raise SystemExit(
            "Refusing to use an output directory containing legacy merged split files: "
            + ", ".join(str(path) for path in stale_merged)
            + ". Choose an empty task-separated output directory or remove those old files explicitly."
        )

    required_bloom = [
        bloom_dir / "train.json",
        bloom_dir / "validation.json",
        bloom_dir / "test.json",
    ]
    missing = [str(path) for path in required_bloom if not path.is_file()]
    if missing:
        raise SystemExit("Missing canonical BloomShift files:\n" + "\n".join(missing))
    if args.max_seq_length < 16:
        raise SystemExit("--max-seq-length must be at least 16.")
    if args.max_qa_train is not None and args.max_qa_train < 1:
        raise SystemExit("--max-qa-train must be positive when supplied.")
    if args.max_sum_train is not None and args.max_sum_train < 1:
        raise SystemExit("--max-sum-train must be positive when supplied.")
    for arg_name in (
        "max_qa_validation", "max_qa_test", "max_sum_validation", "max_sum_test"
    ):
        value = getattr(args, arg_name)
        if value is not None and value < 1:
            raise SystemExit(f"--{arg_name.replace('_', '-')} must be positive when supplied.")

    print("[1/7] Loading and validating canonical BloomShift splits...", flush=True)
    bloom_train_raw = load_json(required_bloom[0])
    bloom_val_raw = load_json(required_bloom[1])
    bloom_test_raw = load_json(required_bloom[2])
    annotation_provenance = validate_annotation_provenance(
        bloom_dir, bloom_train_raw, bloom_val_raw, bloom_test_raw
    )

    bloom_train = normalize_bloom(bloom_train_raw, "train")
    bloom_val = normalize_bloom(bloom_val_raw, "validation")
    bloom_test = normalize_bloom(bloom_test_raw, "test")
    if (len(bloom_train), len(bloom_val), len(bloom_test)) != (
        len(bloom_train_raw), len(bloom_val_raw), len(bloom_test_raw)
    ):
        raise SystemExit("No-drop policy violated: not every canonical BloomShift row was retained.")

    def group_set(rows: list[dict[str, Any]]) -> set[Any]:
        return {row["group_id"] for row in rows}

    bloom_groups = {split: group_set(rows) for split, rows in (
        ("train", bloom_train), ("validation", bloom_val), ("test", bloom_test)
    )}
    if bloom_groups["train"] & bloom_groups["validation"]:
        raise SystemExit("BloomShift train/validation group leakage detected.")
    if bloom_groups["train"] & bloom_groups["test"]:
        raise SystemExit("BloomShift train/test group leakage detected.")
    if bloom_groups["validation"] & bloom_groups["test"]:
        raise SystemExit("BloomShift validation/test group leakage detected.")

    # Evaluation splits follow BloomShift sizes by default. Explicit caps can
    # reduce them, but cannot make public-dataset evaluation splits larger.
    qa_val_target = min(len(bloom_val), args.max_qa_validation or len(bloom_val))
    qa_test_target = min(len(bloom_test), args.max_qa_test or len(bloom_test))
    sum_val_target = min(len(bloom_val), args.max_sum_validation or len(bloom_val))
    sum_test_target = min(len(bloom_test), args.max_sum_test or len(bloom_test))
    print(
        "[EVAL CAPS] "
        f"QA validation={qa_val_target}, QA test={qa_test_target}, "
        f"summarization validation={sum_val_target}, summarization test={sum_test_target}",
        flush=True,
    )

    print("[2/7] Preparing SQuAD QA data...", flush=True)
    qa_train, qa_val, qa_test = load_squad(
        args.seed,
        tokenizer_model_id=args.model_id,
        max_seq_length=args.max_seq_length,
        max_train_rows=args.max_qa_train,
        max_validation_rows=qa_val_target,
        max_test_rows=qa_test_target,
    )
    print("[3/7] Preparing arXiv summarization data with bounded memory...", flush=True)
    arxiv_train, arxiv_val, arxiv_test, arxiv_excluded, arxiv_available = load_arxiv_summarization(
        tokenizer_model_id=args.model_id,
        max_seq_length=args.max_seq_length,
        max_train_rows=args.max_sum_train,
        max_validation_rows=sum_val_target,
        max_test_rows=sum_test_target,
        seed=args.seed,
    )
    if not qa_train or not qa_val or not qa_test:
        raise SystemExit("SQuAD preparation produced an empty split.")
    if not arxiv_train or not arxiv_val or not arxiv_test:
        raise SystemExit("arXiv summarization preparation produced an empty split.")

    # Record rows retained by the early-stopping source scans. These are not
    # estimates of the full source splits' total eligible-example counts.
    qa_train_available = len(qa_train)
    arxiv_train_available = arxiv_available["train"]
    qa_validation_available = len(qa_val)
    arxiv_validation_available = arxiv_available["validation"]
    qa_test_available = len(qa_test)
    arxiv_test_available = arxiv_available["test"]

    print("[4/7] Applying deterministic training caps and evaluation split sizes...", flush=True)
    # Caps are applied during source iteration/tokenization, not after full conversion.
    if args.max_qa_train is not None and len(qa_train) > args.max_qa_train:
        raise AssertionError("QA training cap was not enforced during source selection.")
    if args.max_sum_train is not None and len(arxiv_train) > args.max_sum_train:
        raise AssertionError("Summarization training cap was not enforced during source selection.")

    # Match each auxiliary evaluation task's row count to BloomShift's held-out
    # count. Fixed, task-specific seeds make selection reproducible. Sorting after
    # sampling gives stable file order. Source split boundaries are preserved.
    qa_val_cap = min(len(qa_val), qa_val_target)
    sum_val_cap = min(len(arxiv_val), sum_val_target)
    qa_test_cap = min(len(qa_test), qa_test_target)
    sum_test_cap = min(len(arxiv_test), sum_test_target)
    if len(qa_val) > qa_val_cap:
        qa_val = random.Random(args.seed + 303).sample(qa_val, qa_val_cap)
        qa_val.sort(key=lambda row: str(row.get("id", "")))
    if len(arxiv_val) > sum_val_cap:
        arxiv_val = random.Random(args.seed + 404).sample(arxiv_val, sum_val_cap)
        arxiv_val.sort(key=lambda row: str(row.get("id", "")))
    if len(qa_test) > qa_test_cap:
        qa_test = random.Random(args.seed + 505).sample(qa_test, qa_test_cap)
        qa_test.sort(key=lambda row: str(row.get("id", "")))
    if len(arxiv_test) > sum_test_cap:
        arxiv_test = random.Random(args.seed + 606).sample(arxiv_test, sum_test_cap)
        arxiv_test.sort(key=lambda row: str(row.get("id", "")))

    # Fail closed if a future refactor accidentally allows oversized evaluation splits.
    for task_name, rows, cap in (
        ("QA validation", qa_val, qa_val_target),
        ("QA test", qa_test, qa_test_target),
        ("summarization validation", arxiv_val, sum_val_target),
        ("summarization test", arxiv_test, sum_test_target),
    ):
        if len(rows) > cap:
            raise AssertionError(f"{task_name} has {len(rows)} rows, exceeding cap {cap}.")

    # Task files stay physically separate. Evaluation caps make per-task example
    # counts comparable to BloomShift while retaining independent source splits.
    task_splits: dict[str, dict[str, list[dict[str, Any]]]] = {
        TASK_BLOOM: {
            "train": enrich_prompts(bloom_train),
            "validation": enrich_prompts(bloom_val),
            "test": enrich_prompts(bloom_test),
        },
        TASK_QA: {
            "train": enrich_prompts(qa_train),
            "validation": enrich_prompts(qa_val),
            "test": enrich_prompts(qa_test),
        },
        TASK_SUM: {
            "train": enrich_prompts(arxiv_train),
            "validation": enrich_prompts(arxiv_val),
            "test": enrich_prompts(arxiv_test),
        },
    }
    for task in ALLOWED_TASKS:
        for split in ("train", "validation", "test"):
            rows = task_splits[task][split]
            if not rows:
                raise SystemExit(f"Empty prepared task split: {task}/{split}")
            if any(row.get("task") != task or row.get("split") != split for row in rows):
                raise SystemExit(f"Incorrect task/split tags in {task}/{split}.")
            if not all(row.get("sft_text") and row.get("prompt_text") and row.get("generation_prompt") for row in rows):
                raise SystemExit(f"Missing SFT/prompt content in {task}/{split}.")

    print("[5/7] Enriching prompts and checking token budgets...", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    token_counts: dict[str, dict[str, dict[str, int]]] = {}
    for split in ("train", "validation", "test"):
        token_counts[split] = {}
        for task in (TASK_BLOOM, TASK_QA, TASK_SUM):
            rows = task_splits[task][split]
            overlength: list[str] = []
            prompt_overflow: list[str] = []
            lengths: list[int] = []
            print(f"[tokens] Checking {task}/{split}: {len(rows):,} examples...", flush=True)
            for row_index, row in enumerate(rows):
                if (row_index + 1) % 100 == 0 or row_index + 1 == len(rows):
                    print(f"[tokens] {task}/{split}: {row_index + 1:,}/{len(rows):,}", flush=True)
                full_len = bounded_token_count(tokenizer, row["sft_text"], args.max_seq_length)
                prompt_len = bounded_token_count(tokenizer, row["prompt_text"], args.max_seq_length)
                lengths.append(full_len)
                if full_len > args.max_seq_length:
                    overlength.append(str(row["id"]))
                if prompt_len >= args.max_seq_length:
                    prompt_overflow.append(str(row["id"]))
            if overlength or prompt_overflow:
                raise SystemExit(
                    f"{task}/{split} token budget failure: overlength={len(overlength)}, "
                    f"prompt_without_target={len(prompt_overflow)}; "
                    f"examples={(overlength + prompt_overflow)[:10]}"
                )
            lengths.sort()
            token_counts[split][task] = {
                "rows": len(rows),
                "overlength": 0,
                "min_tokens": min(lengths),
                "p50_tokens": lengths[int((len(lengths) - 1) * 0.50)],
                "p95_tokens": lengths[int((len(lengths) - 1) * 0.95)],
                "p99_tokens": lengths[int((len(lengths) - 1) * 0.99)],
                "max_tokens": max(lengths),
            }

    print("[6/7] Writing task-separated JSONL files...", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_hashes: dict[str, str] = {}
    for task, splits in task_splits.items():
        for split, rows in splits.items():
            path = out_dir / task / f"{split}.jsonl"
            write_jsonl(path, rows)
            rel = path.relative_to(out_dir).as_posix()
            file_hashes[rel] = sha256_file(path)

    source_hashes = {path.name: sha256_file(path) for path in required_bloom}
    attestation_path = bloom_dir / "batch_review_attestation.json"
    source_hashes[attestation_path.name] = sha256_file(attestation_path)

    task_counts = {
        split: {task: len(task_splits[task][split]) for task in (TASK_BLOOM, TASK_QA, TASK_SUM)}
        for split in ("train", "validation", "test")
    }
    actual_train_total = sum(task_counts["train"].values())
    manifest = {
        "version": "multitask_bloom_rewrite_separate_v1",
        "storage_layout": "one_directory_per_task",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "generator_model_role": "Qwen2.5-1.5B-Instruct shared multitask generator",
        "classifier_dataset_excluded": True,
        "tasks": [TASK_BLOOM, TASK_QA, TASK_SUM],
        "persisted_merged_jsonl": False,
        "bloom_source": {
            "path": str(bloom_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
            "version": "bloomshift_final_candidate",
            "files_sha256": source_hashes,
            "heldout_annotation_provenance": annotation_provenance,
            "batch_review_attestation": {
                "path": str(attestation_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": source_hashes[attestation_path.name],
            },
            "training_split_eligibility_policy": (
                "All 1075 canonical BloomShift train rows are used; all 212 validation and 135 test rows "
                "remain held out. Legacy review flags are retained as provenance and never used as row filters. "
                "Owner batch attestation is not independent dual review or inter-rater reliability."
            ),
            "train_rows": len(bloom_train),
            "validation_rows": len(bloom_val),
            "test_rows": len(bloom_test),
        },
        "qa_source": {
            "dataset": "rajpurkar/squad",
            "requested_train_fraction": 0.95,
            "retained_train_rows_after_early_stop": qa_train_available,
            "prepared_train_rows": len(task_splits[TASK_QA]["train"]),
            "retained_validation_rows_after_early_stop": qa_validation_available,
            "prepared_validation_rows": len(task_splits[TASK_QA]["validation"]),
            "retained_test_rows_after_early_stop": qa_test_available,
            "prepared_test_rows": len(task_splits[TASK_QA]["test"]),
            "official_validation_used_as_test": True,
            "evaluation_sampling": {
                "validation_target_rows": qa_val_target,
                "test_target_rows": qa_test_target,
                "seed_offsets": {"validation": 303, "test": 505},
                "policy": "Seed-shuffle source candidates and stop conversion at the requested count; train/validation context groups remain isolated.",
            },
            "long_context_policy": {
                "max_seq_length": args.max_seq_length,
                "answer_preserving_context_window": True,
            },
        },
        "summarization_source": {
            "dataset": "ccdv/arxiv-summarization",
            "official_splits_preserved": True,
            "split_names": {"train": "train", "validation": "validation", "test": "test"},
            "retained_train_rows_after_early_stop": arxiv_train_available,
            "prepared_train_rows": len(task_splits[TASK_SUM]["train"]),
            "retained_validation_rows_after_early_stop": arxiv_validation_available,
            "prepared_validation_rows": len(task_splits[TASK_SUM]["validation"]),
            "retained_test_rows_after_early_stop": arxiv_test_available,
            "prepared_test_rows": len(task_splits[TASK_SUM]["test"]),
            "evaluation_sampling": {
                "validation_target_rows": sum_val_target,
                "test_target_rows": sum_test_target,
                "seed_offsets": {"validation": 404, "test": 606},
                "policy": "Seed-shuffle each official source split and stop tokenization at the requested count; over-budget rows are skipped.",
            },
            "long_context_policy": {
                "max_seq_length": args.max_seq_length,
                "over_budget_examples_excluded": arxiv_excluded,
                "policy": "Exclude complete SFT sequences above the fixed token budget rather than truncating article inputs while retaining full abstracts.",
            },
        },
        "tokenizer": {
            "model_id": args.model_id,
            "max_seq_length": args.max_seq_length,
            "sequence_policy": "all persisted complete SFT sequences must fit; fail rather than rely on trainer truncation",
            "bounded_token_count_check": True,
            "checked_sequences": token_counts,
        },
        "upstream_datasets": UPSTREAM_DATASET_INFO,
        "generated_files_sha256": file_hashes,
        "training_data_policy": {
            "policy": "retain_bloom_all_and_apply_optional_independent_auxiliary_caps",
            "ratio_based_downsampling": False,
            "requested_caps": {
                TASK_QA: args.max_qa_train,
                TASK_SUM: args.max_sum_train,
            },
            "available_training_rows_before_caps": {
                TASK_BLOOM: len(bloom_train),
                TASK_QA: qa_train_available,
                TASK_SUM: arxiv_train_available,
            },
            "retained_training_rows": task_counts["train"],
            "task_train_fractions_of_retained_union": {
                task: round(count / actual_train_total, 6)
                for task, count in task_counts["train"].items()
            },
            "note": "QA and summarization caps are applied during source iteration, avoiding full conversion of unused rows. Validation/test auxiliary tasks are capped to BloomShift split counts; task files remain separate.",
        },
        "counts": task_counts,
        "notes": [
            "No combined train.jsonl, validation.jsonl, or test.jsonl is persisted.",
            "Each task's train/validation/test data is stored in its own directory.",
            "BloomShift training rows are retained; optional QA/summarization training caps are independent of evaluation caps.",
            "QA and summarization caps are applied during seeded source iteration, stopping expensive conversion once each target count is reached.",
            "QA train/validation group boundaries are established before row selection to prevent context leakage across those splits.");
            "The 0.5B Bloom classifier dataset is not part of this generator training corpus.",
            "BloomShift's fixed split boundaries and source-group isolation are preserved.",
            "Upstream SQuAD and arXiv revisions are not pinned; preserve this manifest and all generated hashes for reproducibility.",
            "Owner batch attestation is not independent gold-label validation or inter-rater reliability.",
            "Never tune model choices using the held-out test split.",
        ],
    }
    print("[7/7] Writing manifest and preparation report...", flush=True)
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = {
        "layout": "task-separated",
        "persisted_merged_jsonl": False,
        "train_rows": actual_train_total,
        "validation_rows": sum(task_counts["validation"].values()),
        "test_rows": sum(task_counts["test"].values()),
        "by_task": task_counts,
        "output_dir": str(out_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "generated_files_sha256": file_hashes,
    }
    (EXPERIMENT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "reports" / "multitask_separate_prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[SUCCESS] Task-separated dataset preparation completed.", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
