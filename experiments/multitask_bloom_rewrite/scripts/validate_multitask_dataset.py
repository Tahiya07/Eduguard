#!/usr/bin/env python
"""Strict QC for the task-separated EduGuard 1.5B multitask SFT datasets.

Validates every task/split file independently and never creates a combined JSONL.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for search_path in (REPO_ROOT, EXPERIMENT_DIR, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from dataset_io import TASKS, SPLITS, load_manifest, read_jsonl, validate_separate_dataset
from loss_masking import tokenize_with_assistant_only_loss

REQUIRED = {
    "bloom_rewrite": {
        "task", "split", "id", "group_id", "source_question",
        "target_bloom_level", "target_rewrite", "sft_text", "prompt_text", "generation_prompt",
    },
    "qa": {
        "task", "split", "id", "context", "question", "answer",
        "sft_text", "prompt_text", "generation_prompt",
    },
    "summarization": {
        "task", "split", "id", "article", "abstract",
        "sft_text", "prompt_text", "generation_prompt",
    },
}


def norm(value: str) -> str:
    return re.sub(r"\\s+", " ", str(value).strip().lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/multitask_bloom_rewrite_separate")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()
    if args.max_length < 1:
        raise SystemExit("--max-length must be positive.")

    errors: list[str] = []
    manifest: dict = {}
    try:
        integrity = validate_separate_dataset(root, REPO_ROOT)
        manifest = load_manifest(root)
    except Exception as exc:
        integrity = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"dataset integrity/fingerprint validation failed: {exc}")

    split_task_rows: dict[str, dict[str, list[dict]]] = {}
    for split in SPLITS:
        split_task_rows[split] = {}
        for task in TASKS:
            file_path = root / task / f"{split}.jsonl"
            try:
                split_task_rows[split][task] = read_jsonl(file_path)
            except Exception as exc:
                split_task_rows[split][task] = []
                errors.append(f"cannot read {task}/{split}.jsonl: {exc}")

    splits = {
        split: [row for task in TASKS for row in split_task_rows[split][task]]
        for split in SPLITS
    }
    task_counts = {
        split: {task: len(split_task_rows[split][task]) for task in TASKS}
        for split in SPLITS
    }
    all_rows = [row for rows in splits.values() for row in rows]
    if not all_rows:
        errors.append("all task-separated split files are empty")

    for split in SPLITS:
        for task in TASKS:
            seen: set[str] = set()
            for index, row in enumerate(split_task_rows[split][task], 1):
                missing = REQUIRED[task] - row.keys()
                if missing:
                    errors.append(f"{task}/{split}:{index}: missing fields {sorted(missing)}")
                if row.get("task") != task:
                    errors.append(f"{task}/{split}:{index}: unexpected task tag {row.get('task')!r}")
                if row.get("split") != split:
                    errors.append(f"{task}/{split}:{index}: unexpected split tag {row.get('split')!r}")
                rid = str(row.get("id") or "").strip()
                if not rid:
                    errors.append(f"{task}/{split}:{index}: empty id")
                elif rid in seen:
                    errors.append(f"{task}/{split}:{index}: duplicate ID {rid!r}")
                seen.add(rid)
                for key in ("sft_text", "prompt_text", "generation_prompt"):
                    if not str(row.get(key) or "").strip():
                        errors.append(f"{task}/{split}:{index}: empty {key}")
                if task == "bloom_rewrite":
                    for key in ("group_id", "source_question", "target_bloom_level", "target_rewrite"):
                        if not str(row.get(key) or "").strip():
                            errors.append(f"{task}/{split}:{index}: empty {key}")
                    prompt = str(row.get("generation_prompt", "")).lower()
                    for marker in (
                        "original bloom level:", "source bloom level:",
                        "source bloom level", "original bloom level:",
                    ):
                        if marker in prompt:
                            errors.append(f"{task}/{split}:{index}: forbidden source-level marker in generation prompt")
                elif task == "qa":
                    for key in ("context", "question", "answer"):
                        if not str(row.get(key) or "").strip():
                            errors.append(f"{task}/{split}:{index}: empty {key}")
                    context = str(row.get("context") or "")
                    answer = str(row.get("answer") or "")
                    answer_start = row.get("answer_start")
                    if answer_start is not None:
                        try:
                            start = int(answer_start)
                            if context[start:start + len(answer)] != answer:
                                errors.append(f"{task}/{split}:{index}: answer_start does not select the exact answer")
                        except (TypeError, ValueError):
                            errors.append(f"{task}/{split}:{index}: invalid answer_start {answer_start!r}")
                elif task == "summarization":
                    for key in ("article", "abstract"):
                        if not str(row.get(key) or "").strip():
                            errors.append(f"{task}/{split}:{index}: empty {key}")

    # Exact normalized input overlap is checked within each task, across all
    # split boundaries; BloomShift also has group-level isolation in the loader.
    input_fields = {"bloom_rewrite": "source_question", "qa": "context", "summarization": "article"}
    for task, field in input_fields.items():
        by_split = {
            split: {norm(row.get(field, "")) for row in split_task_rows[split][task]}
            for split in SPLITS
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = by_split[left] & by_split[right]
            if overlap:
                errors.append(f"{task} exact input overlap {left}<->{right}: {len(overlap)} inputs")

    length_stats: dict[str, dict] = {}
    overflow_counts: Counter = Counter()
    prompt_overflow: Counter = Counter()
    masked_checks = 0
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

        def bounded_length(text: str) -> int:
            return len(tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length + 1,
            )["input_ids"])

        for split in SPLITS:
            lengths: list[int] = []
            for task in TASKS:
                for index, row in enumerate(split_task_rows[split][task], 1):
                    full = str(row.get("sft_text") or "")
                    prompt = str(row.get("prompt_text") or "")
                    if not full.strip() or not prompt.strip():
                        continue
                    full_len = bounded_length(full)
                    prompt_len = bounded_length(prompt)
                    lengths.append(full_len)
                    if prompt_len >= args.max_length:
                        prompt_overflow[split] += 1
                        errors.append(f"{task}/{split}:{index}: prompt leaves no assistant target within max_length={args.max_length}")
                    if full_len > args.max_length:
                        overflow_counts[split] += 1
                        errors.append(f"{task}/{split}:{index}: complete SFT sequence exceeds max_length={args.max_length}")
                        continue
                    if prompt_len >= full_len:
                        errors.append(f"{task}/{split}:{index}: prompt is not shorter than complete SFT sequence")
                        continue
                    try:
                        encoded = tokenize_with_assistant_only_loss(tokenizer, full, prompt, args.max_length)
                        labels = encoded["labels"]
                        prompt_length = len(tokenizer(
                            prompt, add_special_tokens=False, truncation=False
                        )["input_ids"])
                        if len(labels) != full_len:
                            errors.append(f"{task}/{split}:{index}: label/input token length mismatch")
                            continue
                        if (
                            prompt_length >= len(labels)
                            or any(label != -100 for label in labels[:prompt_length])
                            or all(label == -100 for label in labels[prompt_length:])
                        ):
                            errors.append(f"{task}/{split}:{index}: invalid assistant-only loss mask")
                        else:
                            masked_checks += 1
                    except Exception as exc:
                        errors.append(f"{task}/{split}:{index}: loss-mask failure: {exc}")
            lengths.sort()
            def pct(q: float) -> int:
                return lengths[min(len(lengths) - 1, int((len(lengths) - 1) * q))] if lengths else 0
            length_stats[split] = {
                "count": len(lengths),
                "min": min(lengths) if lengths else 0,
                "p50": pct(0.50),
                "p95": pct(0.95),
                "p99": pct(0.99),
                "max": max(lengths) if lengths else 0,
                "over_max_seq_length": overflow_counts[split],
                "prompt_at_or_over_max_seq_length": prompt_overflow[split],
            }
    except Exception as exc:
        errors.append(f"tokenizer/tokenization QC could not run: {type(exc).__name__}: {exc}")

    train_total = len(splits["train"])
    train_mix = {
        task: round(count / max(1, train_total), 6)
        for task, count in task_counts["train"].items()
    }
    report = {
        "status": "PASS" if not errors else "FAIL",
        "data_layout": "task-separated",
        "persisted_merged_jsonl": False,
        "max_seq_length": args.max_length,
        "model_id": args.model_id,
        "integrity": integrity,
        "rows": {split: len(splits[split]) for split in SPLITS},
        "task_counts": task_counts,
        "train_mix": train_mix,
        "token_length_stats": length_stats,
        "loss_mask_checks": masked_checks,
        "errors": errors[:100],
        "error_count": len(errors),
    }
    report_path = root / "qc_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
