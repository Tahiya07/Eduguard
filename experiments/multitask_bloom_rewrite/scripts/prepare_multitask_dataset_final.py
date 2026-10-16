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
    print(f"BloomShift {split}: included={len(out)}/{len(rows)} under owner batch attestation")
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare SQuAD with context-group isolation and answer-preserving windows.

    The official SQuAD validation split is held out as the QA test split. A
    validation split is carved out of official train by exact context group.
    Windowing is character-based so the gold answer text and its character span
    remain exact; each candidate is checked against the complete SFT token budget.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from prompts import build_sft_text

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
    official_train = list(ds["train"])
    official_dev = list(ds["validation"])

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

    def convert(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
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

    return convert(train_raw, "train"), convert(val_raw, "validation"), convert(official_dev, "test")

def load_arxiv_summarization(
    tokenizer_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length: int = 8192,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Load arXiv summarization using its official splits.

    Keep the dataset's existing train/validation/test boundaries. Exclude an
    example only when its complete prompt+target SFT sequence exceeds the
    declared token budget; never truncate an article while retaining its
    full-paper abstract.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from prompts import build_sft_text

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

    def convert(split_rows: Any, split: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, row in enumerate(split_rows):
            article = str(row.get("article", "")).strip()
            abstract = str(row.get("abstract", "")).strip()
            if not article or not abstract:
                continue

            record = {
                "task": TASK_SUM,
                "split": split,
                "id": str(row.get("id") or f"arxiv_{split}_{i:07d}"),
                "article": article,
                "abstract": abstract,
                "title": str(row.get("title", "")).strip(),
            }
            # Probe only max_seq_length + 1 tokens. This is sufficient to decide
            # whether the full SFT sequence exceeds the budget and avoids asking
            # the tokenizer to process pathological, extremely long articles.
            full_len = len(
                tokenizer(
                    build_sft_text(TASK_SUM, record),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_seq_length + 1,
                )["input_ids"]
            )
            if full_len > max_seq_length:
                excluded[split] += 1
                continue
            out.append(record)
        return out

    excluded = {"train": 0, "validation": 0, "test": 0}
    train_rows = convert(ds["train"], "train")
    val_rows = convert(ds["validation"], "validation")
    test_rows = convert(ds["test"], "test")
    return train_rows, val_rows, test_rows, excluded


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

    qa_train, qa_val, qa_test = load_squad(
        args.seed,
        tokenizer_model_id=args.model_id,
        max_seq_length=args.max_seq_length,
    )
    arxiv_train, arxiv_val, arxiv_test, arxiv_excluded = load_arxiv_summarization(
        tokenizer_model_id=args.model_id,
        max_seq_length=args.max_seq_length,
    )
    if not qa_train or not qa_val or not qa_test:
        raise SystemExit("SQuAD preparation produced an empty split.")
    if not arxiv_train or not arxiv_val or not arxiv_test:
        raise SystemExit("arXiv summarization preparation produced an empty split.")

    # Keep every prepared training example. Do not downsample or ratio-mix task
    # datasets during preparation; each task remains in its own directory.
    # A shared training run can load these independent files later.
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
            for row in rows:
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
            "available_train_rows_after_group_split": len(qa_train),
            "prepared_train_rows": len(task_splits[TASK_QA]["train"]),
            "validation_rows_after_group_split": len(qa_val),
            "official_validation_rows_used_as_test": len(qa_test),
            "official_validation_used_as_test": True,
            "long_context_policy": {
                "max_seq_length": args.max_seq_length,
                "answer_preserving_context_window": True,
            },
        },
        "summarization_source": {
            "dataset": "ccdv/arxiv-summarization",
            "official_splits_preserved": True,
            "split_names": {"train": "train", "validation": "validation", "test": "test"},
            "available_train_rows_after_length_filter": len(arxiv_train),
            "prepared_train_rows": len(task_splits[TASK_SUM]["train"]),
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
            "policy": "retain_all_prepared_training_examples",
            "ratio_based_downsampling": False,
            "task_train_counts": task_counts["train"],
            "task_train_fractions_of_full_union": {
                task: round(count / actual_train_total, 6)
                for task, count in task_counts["train"].items()
            },
            "note": "Fractions describe the retained full dataset sizes; they are not target sampling ratios.",
        },
        "counts": task_counts,
        "notes": [
            "No combined train.jsonl, validation.jsonl, or test.jsonl is persisted.",
            "Each task's train/validation/test data is stored in its own directory.",
            "All prepared training examples are retained; no fixed-ratio downsampling occurs during preparation.",
            "The 0.5B Bloom classifier dataset is not part of this generator training corpus.",
            "BloomShift's fixed split boundaries and source-group isolation are preserved.",
            "Upstream SQuAD and arXiv revisions are not pinned; preserve this manifest and all generated hashes for reproducibility.",
            "Owner batch attestation is not independent gold-label validation or inter-rater reliability.",
            "Never tune model choices using the held-out test split.",
        ],
    }
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
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
