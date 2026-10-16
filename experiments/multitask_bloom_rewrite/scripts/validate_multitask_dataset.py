#!/usr/bin/env python
"""Strict QC for the final EduGuard 1.5B multitask SFT corpus.

This validates the generated JSONL without modifying it. It checks schema,
split/task isolation, duplicate leakage, prompt construction, and Qwen
tokenization against the configured SFT context limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Running a file directly puts only its scripts/ directory on sys.path.
# Add the repository root explicitly so experiment modules can be imported
# without requiring the repository to be installed as a Python package.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASKS = {"bloom_rewrite", "qa", "summarization"}
REQUIRED = {
    "bloom_rewrite": {"task", "split", "id", "group_id", "source_question", "target_bloom_level", "target_rewrite", "sft_text", "prompt_text", "generation_prompt"},
    "qa": {"task", "split", "id", "context", "question", "answer", "sft_text", "prompt_text", "generation_prompt"},
    "summarization": {"task", "split", "id", "article", "abstract", "sft_text", "prompt_text", "generation_prompt"},
}

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"{path}: blank line at {n}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: invalid JSON at line {n}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {n} is not an object")
            rows.append(row)
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/multitask_bloom_rewrite_final")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()

    root = Path(args.data_dir)
    if args.max_length < 1:
        raise SystemExit("--max-length must be positive.")
    splits = {s: load_jsonl(root / f"{s}.jsonl") for s in ("train", "validation", "test")}
    all_rows = [r for rows in splits.values() for r in rows]

    if not all_rows:
        raise SystemExit("QC FAILED: generated corpus is empty.")

    errors: list[str] = []

    # A final corpus is not reproducible without a valid manifest and matching
    # JSONL hashes. Missing/stale metadata is a QC failure, not a warning.
    manifest_path = root / "dataset_manifest.json"
    manifest = {}
    if not manifest_path.is_file():
        errors.append("missing dataset_manifest.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != "multitask_bloom_rewrite_final_v3_all_bloomshift_rows":
                errors.append(
                    "dataset manifest is not from the all-BloomShift-rows v3 builder; rerun "
                    "prepare_multitask_dataset_final.py"
                )
            expected_counts = manifest.get("counts", {})
            for split_name, rows_for_split in splits.items():
                actual = dict(Counter(r.get("task") for r in rows_for_split))
                expected = expected_counts.get(split_name)
                if expected is not None and actual != expected:
                    errors.append(
                        f"stale dataset_manifest.json: {split_name} task counts "
                        f"manifest={expected}, JSONL={actual}"
                    )
            expected_hashes = manifest.get("generated_files_sha256", {})
            for split_name in ("train", "validation", "test"):
                file_path = root / f"{split_name}.jsonl"
                expected_hash = expected_hashes.get(file_path.name)
                if not expected_hash:
                    errors.append(f"manifest missing SHA-256 for {file_path.name}")
                elif sha256_file(file_path) != expected_hash:
                    errors.append(f"manifest SHA-256 mismatch for {file_path.name}")
            declared_limit = manifest.get("tokenizer", {}).get("max_seq_length")
            if declared_limit is not None and declared_limit != args.max_length:
                errors.append(
                    f"max-length mismatch: manifest={declared_limit}, validator={args.max_length}"
                )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read dataset_manifest.json: {exc}")

    task_counts = {}
    ids_by_split = {}
    group_by_split = defaultdict(set)

    for split, rows in splits.items():
        task_counts[split] = dict(Counter(r.get("task") for r in rows))
        ids = set()
        for i, r in enumerate(rows, 1):
            task = r.get("task")
            if task not in TASKS:
                errors.append(f"{split}:{i}: invalid task {task!r}")
                continue
            missing = REQUIRED[task] - r.keys()
            if missing:
                errors.append(f"{split}:{i}: missing fields {sorted(missing)}")
            rid = str(r.get("id", ""))
            if not rid:
                errors.append(f"{split}:{i}: empty id")
            if rid in ids:
                errors.append(f"{split}:{i}: duplicate id {rid}")
            ids.add(rid)
            if r.get("split") != split:
                errors.append(f"{split}:{i}: embedded split={r.get('split')!r}")
            if not r.get("sft_text") or not r.get("prompt_text") or not r.get("generation_prompt"):
                errors.append(f"{split}:{i}: missing prompt text")
            if task == "bloom_rewrite":
                group_id = str(r.get("group_id", "")).strip()
                if not group_id or group_id.lower() == "none":
                    errors.append(f"{split}:{i}: empty Bloom source group_id")
                else:
                    group_by_split[split].add(group_id)
                if not str(r.get("source_question", "")).strip():
                    errors.append(f"{split}:{i}: empty Bloom source question")
                if not str(r.get("target_rewrite", "")).strip():
                    errors.append(f"{split}:{i}: empty Bloom target rewrite")
                if not str(r.get("target_bloom_level", "")).strip():
                    errors.append(f"{split}:{i}: empty Bloom target level")
                for marker in ("Original Bloom level:", "Source Bloom level:", "source bloom level", "original bloom level:"):
                    if marker.lower() in str(r.get("generation_prompt", "")).lower():
                        errors.append(f"{split}:{i}: forbidden source-level marker in Bloom prompt")
            elif task == "qa":
                if not str(r.get("context", "")).strip() or not str(r.get("question", "")).strip() or not str(r.get("answer", "")).strip():
                    errors.append(f"{split}:{i}: empty QA context/question/answer")
            elif task == "summarization":
                if not str(r.get("article", "")).strip() or not str(r.get("abstract", "")).strip():
                    errors.append(f"{split}:{i}: empty summarization article/abstract")
        ids_by_split[split] = ids

    # IDs must be globally disjoint across held-out boundaries.
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = ids_by_split[a] & ids_by_split[b]
        if overlap:
            errors.append(f"id leakage {a}<->{b}: {len(overlap)} ids")

    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = group_by_split[a] & group_by_split[b]
        if overlap:
            errors.append(f"Bloom group leakage {a}<->{b}: {len(overlap)} groups")

    # Exact normalized source/input leakage, checked within each task.
    task_inputs = {}
    for task in TASKS:
        task_inputs[task] = {}
        for split, rows in splits.items():
            vals = []
            for r in rows:
                if r.get("task") != task:
                    continue
                key_field = {"bloom_rewrite": "source_question", "qa": "context", "summarization": "article"}[task]
                vals.append(sha(norm(r.get(key_field, ""))))
            task_inputs[task][split] = set(vals)
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = task_inputs[task][a] & task_inputs[task][b]
            if overlap:
                errors.append(f"{task} input leakage {a}<->{b}: {len(overlap)} exact normalized inputs")

    # Tokenization and loss-mask integrity.
    from transformers import AutoTokenizer
    from experiments.multitask_bloom_rewrite.loss_masking import tokenize_with_assistant_only_loss

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    length_stats = {}
    overflow_counts = Counter()
    prompt_overflow = Counter()
    masked_checks = 0

    def bounded_length(text: str) -> int:
        # max_length + 1 is enough to detect overflow and avoids full tokenization
        # of very long articles/prompts just to measure their length.
        return len(tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length + 1,
        )["input_ids"])

    for split, rows in splits.items():
        lengths = []
        for i, r in enumerate(rows, 1):
            full = str(r.get("sft_text", ""))
            prompt = str(r.get("prompt_text", ""))
            if not full.strip() or not prompt.strip():
                continue
            full_len = bounded_length(full)
            prompt_len = bounded_length(prompt)
            lengths.append(full_len)
            if prompt_len >= args.max_length:
                prompt_overflow[split] += 1
                errors.append(
                    f"{split}:{i}: prompt leaves no assistant target within max_length={args.max_length}"
                )
            if full_len > args.max_length:
                overflow_counts[split] += 1
                errors.append(
                    f"{split}:{i}: complete SFT sequence exceeds max_length={args.max_length}"
                )
                # Do not invoke loss masking on overlength records: that helper
                # otherwise tokenizes the prompt without truncation.
                continue
            if prompt_len >= full_len:
                errors.append(f"{split}:{i}: prompt is not shorter than complete SFT sequence")
                continue
            try:
                tok = tokenize_with_assistant_only_loss(tokenizer, full, prompt, args.max_length)
                labels = tok["labels"]
                if len(labels) != full_len:
                    errors.append(
                        f"{split}:{i}: trainer tokenization length {len(labels)} != validated length {full_len}"
                    )
                    continue
                p = len(tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"])
                if p >= len(labels) or any(x != -100 for x in labels[:p]) or all(x == -100 for x in labels[p:]):
                    errors.append(f"{split}:{i}: invalid assistant-only loss mask")
                masked_checks += 1
            except Exception as e:
                errors.append(f"{split}:{i}: loss-mask/tokenization failure: {e}")
        lengths.sort()
        def pct(q: float) -> int:
            if not lengths:
                return 0
            return lengths[min(len(lengths)-1, int((len(lengths)-1)*q))]
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

    counts = {s: dict(Counter(r["task"] for r in rows)) for s, rows in splits.items()}
    train_total = len(splits["train"])
    train_mix = {k: round(v / train_total, 6) for k, v in counts["train"].items()}

    report = {
        "status": "PASS" if not errors else "FAIL",
        "max_seq_length": args.max_length,
        "model_id": args.model_id,
        "rows": {s: len(v) for s, v in splits.items()},
        "task_counts": counts,
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
