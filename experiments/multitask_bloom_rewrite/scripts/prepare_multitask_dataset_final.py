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
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "multitask_bloom_rewrite_final"

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


def normalize_bloom(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    from prompts import build_generation_prompt, build_prompt_only_text, build_sft_text

    out = []
    for row in rows:
        required = (
            "source_id", "group_id", "source_question",
            "target_bloom_level", "target_rewrite", "example_id"
        )
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"Bloom row missing {missing}: {row}")

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
        }
        record["sft_text"] = build_sft_text(TASK_BLOOM, record)
        record["prompt_text"] = build_prompt_only_text(TASK_BLOOM, record)
        record["generation_prompt"] = build_generation_prompt(TASK_BLOOM, record)
        out.append(record)
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


def sample_for_mix(
    bloom: list[dict[str, Any]],
    qa: list[dict[str, Any]],
    summ: list[dict[str, Any]],
    seed: int,
    bloom_ratio: float,
    qa_ratio: float,
    sum_ratio: float,
) -> list[dict[str, Any]]:
    if abs((bloom_ratio + qa_ratio + sum_ratio) - 1.0) > 1e-9:
        raise ValueError("Train mix ratios must sum to 1.")

    if not bloom:
        raise ValueError("BloomShift train split is empty.")

    target_total = round(len(bloom) / bloom_ratio)
    target_qa = round(target_total * qa_ratio)
    target_sum = round(target_total * sum_ratio)

    rng = random.Random(seed)
    qa = list(qa)
    summ = list(summ)
    rng.shuffle(qa)
    rng.shuffle(summ)

    if len(qa) < target_qa or len(summ) < target_sum:
        raise ValueError(
            f"Insufficient auxiliary data for requested mix: "
            f"need QA={target_qa}, Sum={target_sum}; "
            f"available QA={len(qa)}, Sum={len(summ)}"
        )

    rows = list(bloom) + qa[:target_qa] + summ[:target_sum]
    rng.shuffle(rows)
    return rows


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row["task"] for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bloom-dir", default=str(DEFAULT_BLOOM_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mix-bloom", type=float, default=0.40)
    parser.add_argument("--mix-qa", type=float, default=0.30)
    parser.add_argument("--mix-sum", type=float, default=0.30)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    args = parser.parse_args()

    bloom_dir = Path(args.bloom_dir)
    if not bloom_dir.is_absolute():
        bloom_dir = REPO_ROOT / bloom_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    required_bloom = [bloom_dir / "train.json", bloom_dir / "validation.json", bloom_dir / "test.json"]
    missing = [str(p) for p in required_bloom if not p.exists()]
    if missing:
        raise SystemExit("Missing canonical BloomShift files:\n" + "\n".join(missing))

    if args.max_seq_length < 16:
        raise SystemExit("--max-seq-length must be at least 16.")
    ratios = (args.mix_bloom, args.mix_qa, args.mix_sum)
    if any(r < 0 for r in ratios) or not any(r > 0 for r in ratios):
        raise SystemExit("Train mix ratios must be non-negative and at least one must be positive.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise SystemExit("Train mix ratios must sum to 1.")
    if args.mix_bloom <= 0:
        raise SystemExit("--mix-bloom must be positive because the target train size is anchored to BloomShift.")

    bloom_train = normalize_bloom(load_json(required_bloom[0]), "train")
    bloom_val = normalize_bloom(load_json(required_bloom[1]), "validation")
    bloom_test = normalize_bloom(load_json(required_bloom[2]), "test")

    # Source-group isolation is already encoded by the canonical BloomShift split.
    def group_set(rows: list[dict[str, Any]]) -> set[Any]:
        return {r["group_id"] for r in rows}

    if group_set(bloom_train) & group_set(bloom_val):
        raise SystemExit("BloomShift train/validation group leakage detected.")
    if group_set(bloom_train) & group_set(bloom_test):
        raise SystemExit("BloomShift train/test group leakage detected.")
    if group_set(bloom_val) & group_set(bloom_test):
        raise SystemExit("BloomShift validation/test group leakage detected.")

    qa_train, qa_val, qa_test = load_squad(
        args.seed,
        tokenizer_model_id=args.model_id,
        max_seq_length=args.max_seq_length,
    )
    sum_train, sum_val, sum_test, arxiv_excluded = load_arxiv_summarization(
        tokenizer_model_id=args.model_id, max_seq_length=args.max_seq_length
    )

    if not qa_train or not qa_val or not qa_test:
        raise SystemExit("SQuAD preparation produced an empty split.")
    if not sum_train or not sum_val or not sum_test:
        raise SystemExit("arXiv summarization preparation produced an empty split.")

    train_rows = sample_for_mix(
        bloom_train, qa_train, sum_train, args.seed,
        args.mix_bloom, args.mix_qa, args.mix_sum
    )
    val_rows = enrich_prompts(bloom_val + qa_val + sum_val)
    test_rows = enrich_prompts(bloom_test + qa_test + sum_test)

    # The canonical BloomShift rows are already prompt-enriched above.
    train_rows = enrich_prompts(train_rows)

    # Final hard gate: no task may reach training with a sequence that the
    # trainer would silently truncate. Probe only max_seq_length + 1 tokens.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    token_counts: dict[str, dict[str, int]] = {}
    for split_name, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
        overlength = []
        for row in rows:
            full_len = bounded_token_count(tokenizer, row["sft_text"], args.max_seq_length)
            prompt_len = bounded_token_count(tokenizer, row["prompt_text"], args.max_seq_length)
            if full_len > args.max_seq_length:
                overlength.append(f"{row['task']}:{row['id']}")
            if prompt_len >= args.max_seq_length:
                raise SystemExit(
                    f"{split_name} prompt leaves no assistant target tokens: "
                    f"{row['task']}:{row['id']} (prompt token count exceeds the usable budget)."
                )
        if overlength:
            raise SystemExit(
                f"{split_name} contains {len(overlength)} overlength SFT sequence(s); "
                f"the builder will not emit data the trainer silently truncates. "
                f"First examples: {overlength[:10]}"
            )
        token_counts[split_name] = {"rows": len(rows), "overlength": len(overlength)}

    for name, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
        tasks = {r["task"] for r in rows}
        if not tasks <= ALLOWED_TASKS:
            raise SystemExit(f"Unexpected task(s) in {name}: {tasks - ALLOWED_TASKS}")
        if not all(r.get("sft_text") and r.get("prompt_text") for r in rows):
            raise SystemExit(f"Missing SFT/prompt text in {name}.")

    # No Bloom example may cross the frozen BloomShift split boundaries.
    test_bloom_ids = {r["id"] for r in bloom_test}
    if test_bloom_ids & {r["id"] for r in train_rows if r["task"] == TASK_BLOOM}:
        raise SystemExit("BloomShift test examples leaked into generator train.")
    if test_bloom_ids & {r["id"] for r in val_rows if r["task"] == TASK_BLOOM}:
        raise SystemExit("BloomShift test examples leaked into generator validation.")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "validation.jsonl", val_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    source_hashes = {
        p.name: sha256_file(p)
        for p in required_bloom
    }
    generated_hashes = {
        name: sha256_file(out_dir / name)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
    }
    manifest = {
        "version": "multitask_bloom_rewrite_final_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "generator_model_role": "Qwen2.5-1.5B-Instruct shared multitask generator",
        "classifier_dataset_excluded": True,
        "tasks": [TASK_BLOOM, TASK_QA, TASK_SUM],
        "bloom_source": {
            "path": str(bloom_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
            "version": "bloomshift_final_candidate",
            "files_sha256": source_hashes,
            "train_rows": len(bloom_train),
            "validation_rows": len(bloom_val),
            "test_rows": len(bloom_test),
        },
        "qa_source": {
            "dataset": "rajpurkar/squad",
            "requested_train_fraction": 0.95,
            "actual_train_fraction_of_official_train": round(len(qa_train) / max(1, len(qa_train) + len(qa_val)), 6),
            "train_rows_after_group_split": len(qa_train),
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
            "long_context_policy": {
                "max_seq_length": args.max_seq_length,
                "over_budget_examples_excluded": arxiv_excluded,
                "policy": "exclude complete SFT sequences above the fixed training budget rather than truncating full-paper inputs while retaining their abstracts",
            },
        },
        "tokenizer": {
            "model_id": args.model_id,
            "max_seq_length": args.max_seq_length,
            "sequence_policy": "all emitted complete SFT sequences must fit; builder fails rather than relying on trainer truncation",
            "bounded_token_count_check": True,
            "checked_sequences": token_counts,
        },
        "upstream_datasets": UPSTREAM_DATASET_INFO,
        "generated_files_sha256": generated_hashes,
        "train_mix": {
            "requested": {TASK_BLOOM: args.mix_bloom, TASK_QA: args.mix_qa, TASK_SUM: args.mix_sum},
            "actual": {k: round(v / max(1, len(train_rows)), 6) for k, v in counts(train_rows).items()},
        },
        "counts": {
            "train": counts(train_rows),
            "validation": counts(val_rows),
            "test": counts(test_rows),
        },
        "notes": [
            "Upstream Hugging Face split fingerprints are recorded; dataset revisions are not pinned by this script, so preserve the manifest and generated file hashes with the experiment.",
            "Bloom transformation data is taken only from the canonical BloomShift final candidate.",
            "The 0.5B Bloom classifier dataset is not included in this generator corpus.",
            "Train/validation/test are kept task-aware and prompt-conditioned.",
            "BloomShift group isolation is checked before writing the corpus.",
            "Do not tune hyperparameters on the held-out test split.",
        ],
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = {
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "test_rows": len(test_rows),
        "by_task": manifest["counts"],
        "output_dir": str(out_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    (EXPERIMENT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "reports" / "multitask_final_prepare_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
