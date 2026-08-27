#!/usr/bin/env python
"""Build the multi-task train/validation/test corpus.

- Bloom: reuse bloom_rewrite_synth_v2 (all validated rows)
- QA: SQuAD 1.1 official splits, deterministic train subsample
- Summarization: ccdv/pubmed-summarization official splits, deterministic subsample

Task mix (Mix A, pre-registered): Bloom 40% / QA 30% / Sum 30% of TRAIN rows
via controlled sampling (not naive full concatenation).

Does not start training. Does not touch the test set for tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from bloom_validation import validate_bloom_example  # noqa: E402
from paths import (  # noqa: E402
    BLOOM_DATASET_HASH,
    BLOOM_DATASET_VERSION,
    BLOOM_REWRITE_DIR,
    CACHE_DIR,
    DEFAULT_TRAIN_MIX,
    MULTITASK_DATA_DIR,
    QA_TRAIN_SUBSAMPLE,
    REPORTS_DIR,
    SEED,
    SUM_TRAIN_SUBSAMPLE,
    TASK_BLOOM,
    TASK_QA,
    TASK_SUMMARIZATION,
)
from prompts import build_generation_prompt, build_prompt_only_text, build_sft_text  # noqa: E402

MIX_PRESETS = {
    "A": {TASK_BLOOM: 0.40, TASK_QA: 0.30, TASK_SUMMARIZATION: 0.30},
    "B": {TASK_BLOOM: 0.50, TASK_QA: 0.25, TASK_SUMMARIZATION: 0.25},
    "C": {TASK_BLOOM: 0.35, TASK_QA: 0.325, TASK_SUMMARIZATION: 0.325},
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def deterministic_sample(rows: list[dict], k: int, seed: int, id_key: str) -> list[dict]:
    if k >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda x: sha256_text(f"{seed}:{x[1].get(id_key, x[0])}"))
    # Stable shuffle of indices via seed
    order = list(range(len(rows)))
    rng.shuffle(order)
    chosen = sorted(order[:k])
    return [rows[i] for i in chosen]


def load_bloom_rows(split: str) -> list[dict]:
    path = BLOOM_REWRITE_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing Bloom rewrite split: {path}")
    out = []
    for row in read_jsonl(path):
        source_q = row.get("source_question") or row.get("question")
        target = row.get("target_bloom_level") or row.get("target_level")
        rewrite = row.get("target_rewrite") or row.get("rewrite")
        if not source_q or not target or not rewrite:
            continue
        rec = {
            "task": TASK_BLOOM,
            "split": split,
            "example_id": row.get("example_id")
            or row.get("id")
            or sha256_text(f"bloom|{source_q}|{target}|{rewrite}")[:16],
            "source_id": row.get("source_id"),
            "source_question": source_q,
            "source_bloom_level": row.get("source_bloom_level"),
            "target_bloom_level": target,
            "target_rewrite": rewrite,
            "transformation_type": row.get("transformation_type"),
            "synthetic": True,
            "policy_version": row.get("policy_version"),
            "validation_status": row.get("validation_status", "imported"),
            "group_id": row.get("group_id") or row.get("source_id"),
        }
        rec["sft_text"] = build_sft_text(TASK_BLOOM, rec)
        rec["prompt_text"] = build_prompt_only_text(TASK_BLOOM, rec)
        out.append(rec)
    return out


def _qa_usable(ex: dict) -> bool:
    answers = ex.get("answers") or {}
    texts = answers.get("text") if isinstance(answers, dict) else None
    if not texts:
        return False
    context = (ex.get("context") or "").strip()
    question = (ex.get("question") or "").strip()
    return bool(context and question and str(texts[0]).strip())


def _make_qa_rec(ex: dict, hf_split: str, experiment_split: str, i: int) -> dict | None:
    if not _qa_usable(ex):
        return None
    answers = ex.get("answers") or {}
    texts = answers.get("text") if isinstance(answers, dict) else None
    answer = texts[0]
    context = (ex.get("context") or "").strip()
    question = (ex.get("question") or "").strip()
    eid = ex.get("id") or f"squad-{hf_split}-{i}"
    rec = {
        "task": TASK_QA,
        "split": experiment_split,
        "example_id": f"qa-{eid}",
        "source_id": eid,
        "context": " ".join(context.split()[:350]),
        "question": question,
        "answer": str(answer).strip(),
        "synthetic": False,
        "dataset": "rajpurkar/squad",
        "hf_split": hf_split,
        "loader": "official_squad_v1.1_json",
    }
    rec["sft_text"] = build_sft_text(TASK_QA, rec)
    rec["prompt_text"] = build_prompt_only_text(TASK_QA, rec)
    return rec


def load_squad(split: str, subsample: int | None, seed: int) -> tuple[list[dict], dict]:
    """Load SQuAD 1.1 via official JSON with leakage-safe val/test partitioning."""
    from squad_loader import load_squad_split

    if split == "train":
        examples, file_meta = load_squad_split(CACHE_DIR / "squad_official", "train")
        usable_idx = [i for i, ex in enumerate(examples) if _qa_usable(ex)]
        chosen = usable_idx
        meta: dict[str, Any] = {
            "hf_split": "train",
            "raw_loaded": len(examples),
            "usable": len(usable_idx),
            "file_meta": file_meta,
        }
        if subsample is not None and subsample < len(usable_idx):
            fake = [
                {"example_id": f"qa-{(examples[i].get('id') or i)}", "idx": i}
                for i in usable_idx
            ]
            sampled = deterministic_sample(fake, subsample, seed, "example_id")
            chosen = [r["idx"] for r in sampled]
            meta["subsample_to"] = subsample
            meta["subsample_before"] = len(usable_idx)
            meta["selected_ids"] = [r["example_id"] for r in sampled]
        rows = []
        for i in chosen:
            rec = _make_qa_rec(examples[i], "train", "train", i)
            if rec:
                rows.append(rec)
        return rows, meta

    examples, file_meta = load_squad_split(CACHE_DIR / "squad_official", "validation")
    pool: list[tuple[str, int]] = []
    for i, ex in enumerate(examples):
        if not _qa_usable(ex):
            continue
        eid = ex.get("id") or f"squad-validation-{i}"
        pool.append((f"qa-{eid}", i))
    rng = random.Random(seed + 11)
    order = list(range(len(pool)))
    rng.shuffle(order)
    mid = len(order) // 2
    half = order[:mid] if split == "validation" else order[mid:]
    chosen_pairs = [pool[j] for j in half]
    if subsample is not None and len(chosen_pairs) > subsample:
        fake = [{"example_id": eid, "idx": idx} for eid, idx in chosen_pairs]
        sampled = deterministic_sample(fake, subsample, seed, "example_id")
        chosen_pairs = [(r["example_id"], r["idx"]) for r in sampled]
    rows = []
    for eid, i in chosen_pairs:
        rec = _make_qa_rec(examples[i], "validation", split, i)
        if rec is None:
            continue
        rec["squad_partition"] = (
            "official_validation_half_a" if split == "validation" else "official_validation_half_b"
        )
        rows.append(rec)
    meta = {
        "hf_split": "validation",
        "raw_loaded": len(examples),
        "usable": len(rows),
        "partition": "half_a" if split == "validation" else "half_b",
        "selected_ids": [eid for eid, _ in chosen_pairs],
        "file_meta": file_meta,
        "note": (
            "SQuAD has no public labeled test; official validation was "
            "deterministically split into experiment validation/test."
        ),
    }
    return rows, meta


def load_pubmed(split: str, subsample: int | None, seed: int) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("ccdv/pubmed-summarization", split=split, cache_dir=str(CACHE_DIR))
    usable_indices: list[int] = []
    for i in range(len(ds)):
        ex = ds[i]
        if (ex.get("article") or "").strip() and (ex.get("abstract") or "").strip():
            usable_indices.append(i)
    chosen = usable_indices
    meta: dict[str, Any] = {
        "hf_split": split,
        "raw_loaded": len(ds),
        "usable": len(usable_indices),
    }
    sample_seed = seed if split == "train" else seed + 7
    if subsample is not None and subsample < len(usable_indices):
        fake = [{"example_id": f"pubmed-{split}-{i}", "idx": i} for i in usable_indices]
        sampled = deterministic_sample(fake, subsample, sample_seed, "example_id")
        chosen = [r["idx"] for r in sampled]
        key = "subsample_to" if split == "train" else "eval_subsample_to"
        before_key = "subsample_before" if split == "train" else "eval_subsample_before"
        meta[key] = subsample
        meta[before_key] = len(usable_indices)
        meta["selected_ids"] = [r["example_id"] for r in sampled]

    rows = []
    for i in chosen:
        ex = ds[i]
        article = (ex.get("article") or "").strip()
        abstract = (ex.get("abstract") or "").strip()
        eid = f"pubmed-{split}-{i}"
        rec = {
            "task": TASK_SUMMARIZATION,
            "split": split,
            "example_id": eid,
            "source_id": eid,
            "article": " ".join(article.split()[:400]),
            "abstract": abstract,
            "synthetic": False,
            "dataset": "ccdv/pubmed-summarization",
            "hf_split": split,
        }
        rec["sft_text"] = build_sft_text(TASK_SUMMARIZATION, rec)
        rec["prompt_text"] = build_prompt_only_text(TASK_SUMMARIZATION, rec)
        rows.append(rec)
    return rows, meta


def balance_train(
    by_task: dict[str, list[dict]],
    mix: dict[str, float],
    seed: int,
) -> tuple[list[dict], dict]:
    """Controlled sampling so train proportions match mix (approximately).

    Strategy: take all Bloom train rows (priority task), then sample QA/Sum
    so final proportions match mix as closely as possible without exceeding
    available pools. If Bloom is smaller than its share target relative to
    desired total, shrink other tasks; if larger, keep all Bloom and set
    total from Bloom / mix[bloom].
    """
    bloom = list(by_task[TASK_BLOOM])
    qa = list(by_task[TASK_QA])
    summ = list(by_task[TASK_SUMMARIZATION])
    mb, mq, ms = mix[TASK_BLOOM], mix[TASK_QA], mix[TASK_SUMMARIZATION]
    # Desired total from Bloom count
    if mb <= 0:
        raise ValueError("Bloom mix must be > 0")
    desired_total = int(round(len(bloom) / mb))
    n_qa = min(len(qa), int(round(desired_total * mq)))
    n_sum = min(len(summ), int(round(desired_total * ms)))
    # Recompute actual total
    qa_s = deterministic_sample(qa, n_qa, seed + 1, "example_id")
    sum_s = deterministic_sample(summ, n_sum, seed + 2, "example_id")
    combined = bloom + qa_s + sum_s
    rng = random.Random(seed)
    rng.shuffle(combined)
    counts = Counter(r["task"] for r in combined)
    total = len(combined) or 1
    props = {k: round(v / total, 4) for k, v in counts.items()}
    meta = {
        "mix_requested": mix,
        "counts": dict(counts),
        "proportions_actual": props,
        "desired_total_from_bloom": desired_total,
        "selected_qa_ids_hash": sha256_text("|".join(r["example_id"] for r in qa_s)),
        "selected_sum_ids_hash": sha256_text("|".join(r["example_id"] for r in sum_s)),
        "n_qa_selected": len(qa_s),
        "n_sum_selected": len(sum_s),
        "n_bloom": len(bloom),
    }
    return combined, meta


def check_leakage(splits: dict[str, list[dict]]) -> dict[str, Any]:
    """Disjointness at source-question / example-id / group level per task."""
    result: dict[str, Any] = {"ok": True, "details": {}}
    for task in (TASK_BLOOM, TASK_QA, TASK_SUMMARIZATION):
        keys = {}
        for split, rows in splits.items():
            keyset = set()
            for r in rows:
                if r["task"] != task:
                    continue
                if task == TASK_BLOOM:
                    key = str(
                        r.get("group_id") or r.get("source_id") or r["source_question"]
                    ).strip().lower()
                else:
                    key = str(r.get("source_id") or r["example_id"])
                keyset.add(key)
            keys[split] = keyset
        overlaps = {
            "train∩validation": len(keys.get("train", set()) & keys.get("validation", set())),
            "train∩test": len(keys.get("train", set()) & keys.get("test", set())),
            "validation∩test": len(keys.get("validation", set()) & keys.get("test", set())),
        }
        task_ok = all(v == 0 for v in overlaps.values())
        result["details"][task] = {"overlaps": overlaps, "ok": task_ok, "sizes": {s: len(k) for s, k in keys.items()}}
        if not task_ok:
            result["ok"] = False
    # Cross-check: identical full SFT strings across splits (often synthetic
    # Bloom templates). Reported for QC but does NOT fail source-level leakage.
    texts = defaultdict(set)
    for split, rows in splits.items():
        for r in rows:
            texts[r["sft_text"]].add(split)
    cross_items = [
        {"n_splits": len(splits_hit), "splits": sorted(splits_hit)}
        for splits_hit in texts.values()
        if len(splits_hit) > 1
    ]
    result["identical_sft_text_across_splits"] = len(cross_items)
    result["identical_sft_text_note"] = (
        "Identical full ChatML strings across splits are recorded for template "
        "memorization risk; source-id/group leakage is the hard stop condition."
    )
    return result


def assert_no_source_leak_in_bloom(rows: list[dict]) -> int:
    leaks = 0
    for r in rows:
        if r["task"] != TASK_BLOOM:
            continue
        # Will raise if markers present
        build_generation_prompt(TASK_BLOOM, r)
        if "Original Bloom level:" in r["sft_text"] or "Source Bloom level:" in r["sft_text"]:
            leaks += 1
    return leaks


def validate_bloom_subset(rows: list[dict], limit: int = 200) -> dict[str, Any]:
    sample = [r for r in rows if r["task"] == TASK_BLOOM][:limit]
    accepted = 0
    reasons: Counter[str] = Counter()
    for r in sample:
        v = validate_bloom_example(r["source_question"], r["target_bloom_level"], r["target_rewrite"])
        if v.accepted:
            accepted += 1
        else:
            reasons[v.rejection_reason or "rejected"] += 1
    return {
        "sampled": len(sample),
        "accepted": accepted,
        "accept_rate": round(accepted / len(sample), 4) if sample else None,
        "rejection_reasons": dict(reasons),
        "note": "Validation is heuristic offline QC; Bloom synth_v2 was already policy-validated upstream.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", choices=sorted(MIX_PRESETS), default="A")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--qa-train-n", type=int, default=QA_TRAIN_SUBSAMPLE)
    parser.add_argument("--sum-train-n", type=int, default=SUM_TRAIN_SUBSAMPLE)
    parser.add_argument("--sum-eval-n", type=int, default=1500, help="Cap PubMed val/test for practicality")
    parser.add_argument("--qa-eval-n", type=int, default=None, help="Optional cap on SQuAD eval rows")
    args = parser.parse_args()

    mix = MIX_PRESETS[args.mix]
    MULTITASK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify Bloom hash
    bloom_manifest = BLOOM_REWRITE_DIR / "dataset_manifest.json"
    if bloom_manifest.exists():
        man = json.loads(bloom_manifest.read_text(encoding="utf-8"))
        got = man.get("dataset_hash")
        if got != BLOOM_DATASET_HASH:
            print(
                f"WARNING: Bloom dataset hash mismatch. expected={BLOOM_DATASET_HASH} got={got}"
            )
        version = man.get("dataset_version")
        if version != BLOOM_DATASET_VERSION:
            print(f"WARNING: Bloom dataset version {version} != {BLOOM_DATASET_VERSION}")

    bloom_train = load_bloom_rows("train")
    bloom_val = load_bloom_rows("validation")
    bloom_test = load_bloom_rows("test")

    qa_train, qa_train_meta = load_squad("train", args.qa_train_n, args.seed)
    qa_val, qa_val_meta = load_squad("validation", args.qa_eval_n, args.seed)
    # SQuAD: use same official validation as held-out test for metrics; do not mix into train.
    qa_test, qa_test_meta = load_squad("test", args.qa_eval_n, args.seed)

    sum_train, sum_train_meta = load_pubmed("train", args.sum_train_n, args.seed)
    sum_val, sum_val_meta = load_pubmed("validation", args.sum_eval_n, args.seed)
    sum_test, sum_test_meta = load_pubmed("test", args.sum_eval_n, args.seed)

    train_balanced, balance_meta = balance_train(
        {
            TASK_BLOOM: bloom_train,
            TASK_QA: qa_train,
            TASK_SUMMARIZATION: sum_train,
        },
        mix,
        args.seed,
    )
    # Validation/test: union of task splits (no rebalancing required for eval)
    validation = bloom_val + qa_val + sum_val
    test = bloom_test + qa_test + sum_test

    splits = {"train": train_balanced, "validation": validation, "test": test}
    leakage = check_leakage(splits)
    leaks = assert_no_source_leak_in_bloom(train_balanced + validation + test)
    bloom_qc = validate_bloom_subset(train_balanced + validation + test, limit=300)

    counts = {
        split: {
            "total": len(rows),
            "by_task": dict(Counter(r["task"] for r in rows)),
        }
        for split, rows in splits.items()
    }

    for split, rows in splits.items():
        write_jsonl(MULTITASK_DATA_DIR / f"{split}.jsonl", rows)

    # Save selected ID lists (not full QA id list if huge — hashes + counts always)
    selected = {
        "qa_train_ids": qa_train_meta.get("selected_ids"),
        "sum_train_ids": sum_train_meta.get("selected_ids"),
        "sum_val_ids": sum_val_meta.get("selected_ids"),
        "sum_test_ids": sum_test_meta.get("selected_ids"),
    }
    (MULTITASK_DATA_DIR / "selected_ids.json").write_text(
        json.dumps(
            {
                k: {"n": len(v) if v else 0, "sha256": sha256_text("|".join(v)) if v else None}
                for k, v in selected.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Full ID lists for reproducibility
    (MULTITASK_DATA_DIR / "selected_ids_full.json").write_text(
        json.dumps({k: v for k, v in selected.items() if v}, indent=2),
        encoding="utf-8",
    )

    # Dataset hash over train+val+test file bytes
    file_hashes = {s: sha256_file(MULTITASK_DATA_DIR / f"{s}.jsonl") for s in splits}
    corpus_hash = sha256_text("|".join(f"{k}:{v}" for k, v in sorted(file_hashes.items())))

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "mix_preset": args.mix,
        "mix": mix,
        "bloom_dataset_version": BLOOM_DATASET_VERSION,
        "bloom_dataset_hash_expected": BLOOM_DATASET_HASH,
        "counts": counts,
        "balance": balance_meta,
        "leakage": leakage,
        "source_level_leaks_in_prompts": leaks,
        "bloom_qc_sample": bloom_qc,
        "sources": {
            "bloom": {"dir": str(BLOOM_REWRITE_DIR), "splits": {"train": len(bloom_train), "validation": len(bloom_val), "test": len(bloom_test)}},
            "squad": {"train": qa_train_meta, "validation": qa_val_meta, "test": qa_test_meta},
            "pubmed": {"train": sum_train_meta, "validation": sum_val_meta, "test": sum_test_meta},
        },
        "file_sha256": file_hashes,
        "corpus_hash": corpus_hash,
        "prompt_contract": {
            "bloom": "question + target_level → rewrite",
            "qa": "context + question → answer",
            "summarization": "article → abstract",
        },
        "notes": [
            "Train is a controlled-balance union; validation/test are unions of official task splits.",
            "SQuAD has no public labeled test; evaluation uses official validation (tagged as test for experiment protocol).",
            "Bloom supervision is synthetic (bloom_rewrite_synth_v2).",
            "Same subsample IDs are used for 0.5B and 1.5B.",
        ],
    }
    (MULTITASK_DATA_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Task balance report
    balance_report = {
        "mix_preset": args.mix,
        "requested": mix,
        "actual_train_proportions": balance_meta["proportions_actual"],
        "actual_train_counts": balance_meta["counts"],
        "sensitivity_presets": MIX_PRESETS,
        "note": "Mix selection must use validation only; Mix A is pre-registered primary.",
    }
    (REPORTS_DIR / "task_balance.json").write_text(json.dumps(balance_report, indent=2), encoding="utf-8")
    (REPORTS_DIR / "leakage_report.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    (REPORTS_DIR / "dataset_quality.md").write_text(
        "\n".join(
            [
                "# Dataset Quality Report",
                "",
                f"Corpus hash: `{corpus_hash}`",
                "",
                "## Counts",
                "```json",
                json.dumps(counts, indent=2),
                "```",
                "",
                "## Leakage",
                "```json",
                json.dumps(leakage, indent=2),
                "```",
                "",
                "## Task balance",
                "```json",
                json.dumps(balance_report, indent=2),
                "```",
                "",
                "## Bloom QC sample",
                "```json",
                json.dumps(bloom_qc, indent=2),
                "```",
                "",
                "## Provenance",
                f"- Bloom: {BLOOM_DATASET_VERSION} (synthetic transformations)",
                "- QA: rajpurkar/squad (SQuAD 1.1)",
                "- Summarization: ccdv/pubmed-summarization",
                "",
                "Bloom rewrite supervision is synthetic and must be stated as such in the paper.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({"corpus_hash": corpus_hash, "counts": counts, "leakage_ok": leakage["ok"], "source_leaks": leaks}, indent=2))
    if not leakage["ok"] or leaks:
        print("STOP: leakage detected or source Bloom leaked into prompts")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
