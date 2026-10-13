#!/usr/bin/env python
"""Prepare the final 1.5B multitask generator corpus.

Final generator tasks:
  1. Bloom transformation from the canonical BloomShift final candidate corpus.
  2. Extractive/grounded QA from SQuAD 1.1.
  3. Summarization from arXiv paper/abstract pairs (ccdv/arxiv-summarization).

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
        # Expansion candidates remain in the canonical split files for review,
        # but must not enter SFT/evaluation corpora until explicitly approved.
        if row.get("training_eligible") is False:
            continue
        if row.get("review_status") == "pending_human_validation":
            continue
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


def load_squad(
    seed: int,
    tokenizer_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length: int = 8192,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("rajpurkar/squad")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True)
    official_train = list(ds["train"])
    official_dev = list(ds["validation"])

    # SQuAD contains multiple questions for the same paragraph. Split by
    # exact context so questions from one paragraph cannot cross train/validation.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in official_train:
        groups.setdefault(str(row.get("context", "")), []).append(row)

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

    def convert(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            answers = row.get("answers", {})
            texts = answers.get("text", [])
            starts = answers.get("answer_start", [])
            context = str(row.get("context", ""))
            question = str(row.get("question", ""))
            answer = str(texts[0]) if texts else ""
            answer_start = starts[0] if starts else None
            if not context or not question or not answer:
                continue

            # Keep every QA example answerable while making pathological
            # contexts fit the fixed SFT budget. Only contexts that would
            # exceed the budget are windowed; the answer span is retained.
            # This is deterministic and applies equally to train/validation/test.
            if split in {"train", "validation", "test"}:
                base_record = {
                    "task": TASK_QA,
                    "split": split,
                    "id": str(row["id"]),
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "answer_start": answer_start,
                    "title": row.get("title"),
                }
                from prompts import build_sft_text

                if len(tokenizer(build_sft_text(TASK_QA, base_record), add_special_tokens=False)["input_ids"]) > max_seq_length:
                    if answer_start is None or context[answer_start:answer_start + len(answer)] != answer:
                        raise ValueError(
                            f"SQuAD answer span mismatch for {row['id']}; cannot safely window context."
                        )

                    # Estimate the available context-token budget from the
                    # fixed system/question/answer portions, then keep a
                    # token window centered on the answer span.
                    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
                    answer_prefix_ids = tokenizer(
                        context[:answer_start], add_special_tokens=False
                    )["input_ids"]
                    answer_ids = tokenizer(
                        answer, add_special_tokens=False
                    )["input_ids"]
                    answer_token_start = len(answer_prefix_ids)
                    answer_token_end = answer_token_start + len(answer_ids)

                    fixed_record = dict(base_record)
                    fixed_record["context"] = ""
                    fixed_tokens = len(
                        tokenizer(
                            build_sft_text(TASK_QA, fixed_record),
                            add_special_tokens=False,
                        )["input_ids"]
                    )
                    available_context_tokens = max_seq_length - fixed_tokens
                    if available_context_tokens <= len(answer_ids):
                        raise ValueError(
                            f"SQuAD answer alone exceeds the available context budget for {row['id']}."
                        )

                    window_tokens = min(len(context_ids), available_context_tokens)
                    extra = window_tokens - len(answer_ids)
                    left = min(answer_token_start, extra // 2)
                    right = extra - left
                    if answer_token_start + right > len(context_ids):
                        right = len(context_ids) - answer_token_start
                        left = min(answer_token_start, extra - right)
                    start_tok = max(0, answer_token_start - left)
                    end_tok = min(len(context_ids), answer_token_end + right)

                    context = tokenizer.decode(
                        context_ids[start_tok:end_tok],
                        skip_special_tokens=True,
                    ).strip()

                    if answer not in context:
                        raise ValueError(
                            f"Answer was lost while windowing SQuAD context for {row['id']}."
                        )
                    answer_start = context.find(answer)

                    base_record["context"] = context
                    base_record["answer_start"] = answer_start

                    final_len = len(
                        tokenizer(
                            build_sft_text(TASK_QA, base_record),
                            add_special_tokens=False,
                        )["input_ids"]
                    )
                    if final_len > max_seq_length:
                        raise ValueError(
                            f"SQuAD context window still exceeds max_seq_length for {row['id']}: "
                            f"{final_len} > {max_seq_length}."
                        )

                out.append(base_record)
        return out

    return convert(train_raw, "train"), convert(val_raw, "validation"), convert(official_dev, "test")


def load_arxiv(
    seed: int,
    tokenizer_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length: int = 8192,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Load arXiv paper/abstract pairs and preserve the dataset's official splits.

    Complete SFT sequences above max_seq_length are excluded consistently in
    every split; papers are not silently truncated while retaining full-paper
    abstracts as targets.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from prompts import build_sft_text

    dataset = load_dataset("ccdv/arxiv-summarization")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True)
    excluded = {"train": 0, "validation": 0, "test": 0}

    def convert(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        out = []
        for i, row in enumerate(rows):
            article = str(row.get("article", "")).strip()
            abstract = str(row.get("abstract", "")).strip()
            if not article or not abstract:
                continue

            # IDs are namespaced by split because this dataset may not expose a
            # stable document ID. Split-level namespace prevents accidental
            # train/validation/test ID collisions in the generated corpus.
            source_id = str(row.get("id") or row.get("paper_id") or f"arxiv_{split}_{i:07d}")
            record = {
                "task": TASK_SUM,
                "split": split,
                "id": f"arxiv_{split}_{source_id}",
                "article": article,
                "abstract": abstract,
                "title": str(row.get("title", "")).strip(),
                "source_dataset": "ccdv/arxiv-summarization",
            }
            full_len = len(
                tokenizer(
                    build_sft_text(TASK_SUM, record),
                    add_special_tokens=False,
                )["input_ids"]
            )
            if full_len > max_seq_length:
                excluded[split] += 1
                continue
            out.append(record)
        return out

    # Preserve the publisher-provided split assignments; do not reshuffle
    # papers across official splits.
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(dataset.keys())
    if missing_splits:
        raise ValueError(
            "ccdv/arxiv-summarization is missing expected official splits: "
            + ", ".join(sorted(missing_splits))
        )
    train_rows = convert(list(dataset["train"]), "train")
    val_rows = convert(list(dataset["validation"]), "validation")
    test_rows = convert(list(dataset["test"]), "test")
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


def cap_evaluation_auxiliary(
    rows: list[dict[str, Any]],
    seed: int,
    split: str,
    max_per_task: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep all Bloom rows and reproducibly cap QA/summarization per evaluation split."""
    kept = [row for row in rows if row["task"] == TASK_BLOOM]
    source_counts = counts(rows)
    for task in (TASK_QA, TASK_SUM):
        candidates = [row for row in rows if row["task"] == task]
        rng = random.Random(f"{seed}:{split}:{task}")
        rng.shuffle(candidates)
        kept.extend(candidates[:max_per_task])
    rng = random.Random(f"{seed}:{split}:final-order")
    rng.shuffle(kept)
    return kept, source_counts


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
    parser.add_argument(
        "--max-eval-rows-per-aux-task",
        type=int,
        default=300,
        help="Maximum QA and summarization rows in each validation/test split; Bloom rows are never capped.",
    )
    args = parser.parse_args()

    if args.max_eval_rows_per_aux_task < 1:
        raise SystemExit("--max-eval-rows-per-aux-task must be >= 1")

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
    sum_train, sum_val, sum_test, arxiv_excluded = load_arxiv(
        args.seed, tokenizer_model_id=args.model_id, max_seq_length=args.max_seq_length
    )

    if not qa_train or not qa_val or not qa_test:
        raise SystemExit("SQuAD preparation produced an empty split.")
    if not sum_train or not sum_val or not sum_test:
        raise SystemExit("arXiv preparation produced an empty split.")

    train_rows = sample_for_mix(
        bloom_train, qa_train, sum_train, args.seed,
        args.mix_bloom, args.mix_qa, args.mix_sum
    )
    # Keep every eligible BloomShift example in validation/test, while limiting
    # each auxiliary task to a reproducible maximum of 300 rows per split.
    # This cap does not affect the 40/30/30 training mixture.
    val_full = bloom_val + qa_val + sum_val
    test_full = bloom_test + qa_test + sum_test
    val_rows, val_source_task_counts = cap_evaluation_auxiliary(
        val_full, args.seed, "validation", args.max_eval_rows_per_aux_task
    )
    test_rows, test_source_task_counts = cap_evaluation_auxiliary(
        test_full, args.seed, "test", args.max_eval_rows_per_aux_task
    )
    val_rows = enrich_prompts(val_rows)
    test_rows = enrich_prompts(test_rows)

    # The canonical BloomShift rows are already prompt-enriched above.
    train_rows = enrich_prompts(train_rows)

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
    manifest = {
        "version": "multitask_bloom_rewrite_final_v1",
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
            "train_fraction_after_deterministic_split": 0.95,
            "official_validation_used_as_test": True,
            "long_context_policy": {
                "max_seq_length": args.max_seq_length,
                "answer_preserving_context_window": True,
            },
            "evaluation_sampling": {
                "policy": "deterministic cap per validation/test split",
                "max_rows_per_split": args.max_eval_rows_per_aux_task,
                "applies_to": ["qa"],
            },
        },
        "summarization_source": {
            "dataset": "ccdv/arxiv-summarization",
            "official_splits_preserved": True,
            "split_names": {
                "train": "train",
                "validation": "validation",
                "test": "test",
            },
            "long_context_policy": {
                "max_seq_length": args.max_seq_length,
                "over_budget_examples_excluded": arxiv_excluded,
                "policy": "exclude complete SFT sequences above the fixed training budget rather than truncating full-paper inputs while retaining their abstracts",
            },
            "evaluation_sampling": {
                "policy": "deterministic cap per validation/test split",
                "max_rows_per_split": args.max_eval_rows_per_aux_task,
                "applies_to": ["summarization"],
            },
        },
        "split_policy": {
            "bloom_rewrite": "canonical BloomShift source-group-disjoint split; counts are inherited from the audited candidate dataset",
            "qa": "SQuAD train is split by context group into 95% train and 5% validation; official SQuAD validation is held out as test",
            "summarization": "preserve ccdv/arxiv-summarization official train/validation/test splits",
            "train_sampling": "balanced training subset uses the configured 40/30/30 task mix; validation and test remain held out and are not sampled into training",
            "evaluation_sampling": "keep all eligible BloomShift validation/test rows; independently sample at most max_eval_rows_per_aux_task QA and summarization rows per split using seed/split/task-specific RNG",
            "note": "A single global 80/10/10 ratio is not imposed across different benchmark datasets because that would discard or leak their official/group-safe split assignments.",
        },
        "train_mix": {
            TASK_BLOOM: args.mix_bloom,
            TASK_QA: args.mix_qa,
            TASK_SUM: args.mix_sum,
        },
        "counts": {
            "train": counts(train_rows),
            "validation": counts(val_rows),
            "test": counts(test_rows),
        },
        "evaluation_sampling": {
            "seed": args.seed,
            "max_rows_per_auxiliary_task_per_split": args.max_eval_rows_per_aux_task,
            "validation_source_counts_before_sampling": val_source_task_counts,
            "test_source_counts_before_sampling": test_source_task_counts,
            "validation_final_counts": counts(val_rows),
            "test_final_counts": counts(test_rows),
            "bloom_rows_capped": False,
        },
        "notes": [
            "Bloom transformation data is taken only from the canonical BloomShift final candidate.",
            "Rows marked training_eligible=false or review_status=pending_human_validation are retained in source split files for review but excluded from the generated SFT/evaluation corpus.",
            "The 0.5B Bloom classifier dataset is not included in this generator corpus.",
            "Train/validation/test are kept task-aware and prompt-conditioned.",
            "BloomShift group isolation is checked before writing the corpus.",
            "Evaluation retains all eligible BloomShift rows and caps QA and summarization at the configured maximum per split with deterministic sampling.",
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
