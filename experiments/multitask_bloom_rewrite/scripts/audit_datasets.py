#!/usr/bin/env python
"""Audit Figshare Bloom, SQuAD 1.1, and ccdv/pubmed-summarization.

Writes:
  reports/dataset_audit.json
  reports/dataset_audit.md

Does not fabricate statistics. Does not start training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from paths import (  # noqa: E402
    BLOOM_REWRITE_DIR,
    CACHE_DIR,
    FIGSHARE_V1,
    REPORTS_DIR,
    SEED,
)

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def word_stats(texts: list[str]) -> dict[str, Any]:
    lengths = [len(WORD_RE.findall(t or "")) for t in texts]
    if not lengths:
        return {"n": 0}
    lengths_sorted = sorted(lengths)

    def pct(p: float) -> float:
        if not lengths_sorted:
            return 0.0
        idx = min(len(lengths_sorted) - 1, max(0, int(math.ceil(p / 100.0 * len(lengths_sorted)) - 1)))
        return float(lengths_sorted[idx])

    return {
        "n": len(lengths),
        "mean_words": round(sum(lengths) / len(lengths), 3),
        "median_words": float(lengths_sorted[len(lengths_sorted) // 2]),
        "p50_words": pct(50),
        "p90_words": pct(90),
        "p95_words": pct(95),
        "p99_words": pct(99),
        "min_words": float(min(lengths)),
        "max_words": float(max(lengths)),
    }


def near_duplicate_count(texts: list[str], sample_cap: int = 4000) -> dict[str, Any]:
    """Exact-normalized and cheap near-dup estimate (prefix+length buckets)."""
    norm = []
    for t in texts:
        s = re.sub(r"\s+", " ", (t or "").lower().strip())
        norm.append(s)
    exact = len(norm) - len(set(norm))
    # Near-dup: identical first 80 chars among non-exact-empty
    prefixes = [s[:80] for s in norm if len(s) >= 40]
    near = len(prefixes) - len(set(prefixes))
    capped = False
    if len(texts) > sample_cap:
        capped = True
        # already full-pass for exact/prefix; note size
    return {
        "exact_duplicate_rows": exact,
        "near_duplicate_prefix80": near,
        "sampled_or_full": "full",
        "capped_note": capped,
    }


def audit_figshare() -> dict[str, Any]:
    path = FIGSHARE_V1
    report: dict[str, Any] = {
        "name": "figshare_bloom_v1",
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "exists": path.exists(),
        "role": "Bloom classification source pool (not rewrite gold)",
        "license_note": "Figshare Bloom classification dataset used by EduGuard; verify local citation/license docs.",
        "domain": "academic assessment / Bloom taxonomy questions",
        "language": "en (assumed; not language-ID'd in this audit)",
    }
    if not path.exists():
        report["error"] = "file_missing"
        return report
    report["sha256"] = sha256_file(path)
    report["size_bytes"] = path.stat().st_size
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        report["columns"] = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)
    report["raw_examples"] = len(rows)
    questions = [(r.get("question") or "").strip() for r in rows]
    missing_q = sum(1 for q in questions if not q)
    labels = [(r.get("bloom_level") or r.get("original_label") or "").strip() for r in rows]
    missing_label = sum(1 for lab in labels if not lab)
    usable = [
        r
        for r in rows
        if (r.get("question") or "").strip() and (r.get("bloom_level") or r.get("original_label") or "").strip()
    ]
    report["usable_examples"] = len(usable)
    report["missing_values"] = {"question": missing_q, "bloom_label": missing_label}
    report["duplicates"] = near_duplicate_count(questions)
    report["label_distribution"] = dict(Counter(labels))
    report["question_length"] = word_stats(questions)
    # Bloom rewrite synth reference
    bloom_manifest = BLOOM_REWRITE_DIR / "dataset_manifest.json"
    bloom_info: dict[str, Any] = {"path": str(bloom_manifest)}
    if bloom_manifest.exists():
        bloom_info["manifest"] = json.loads(bloom_manifest.read_text(encoding="utf-8"))
        splits = {}
        for split in ("train", "validation", "test"):
            p = BLOOM_REWRITE_DIR / f"{split}.jsonl"
            if p.exists():
                n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
                splits[split] = n
                # hash file
                bloom_info.setdefault("split_sha256", {})[split] = sha256_file(p)
        bloom_info["split_counts"] = splits
        bloom_info["total"] = sum(splits.values())
    report["bloom_rewrite_synth_v2"] = bloom_info
    return report


def _load_hf_dataset(name: str, config: str | None, split: str):
    from datasets import load_dataset

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"path": name, "split": split, "cache_dir": str(CACHE_DIR)}
    if config:
        kwargs["name"] = config
    return load_dataset(**kwargs)


def audit_squad() -> dict[str, Any]:
    from squad_loader import load_squad_split

    report: dict[str, Any] = {
        "name": "rajpurkar/squad (SQuAD 1.1)",
        "loader": "official_json_via_squad_loader",
        "task": "qa",
        "source_citation": "Rajpurkar et al., SQuAD: 100,000+ Questions for Machine Comprehension of Text (EMNLP 2016)",
        "license_note": "CC BY-SA 4.0 (SQuAD)",
        "domain": "Wikipedia reading comprehension",
        "language": "en",
        "urls": {
            "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
            "validation": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
        },
    }
    # Record HF incompatibility if present (informational)
    try:
        from datasets import load_dataset

        load_dataset("rajpurkar/squad", split="train[:1]", cache_dir=str(CACHE_DIR))
        report["hf_rajpurkar_squad_status"] = "load_ok"
    except Exception as exc:  # noqa: BLE001
        report["hf_rajpurkar_squad_status"] = f"failed: {type(exc).__name__}: {exc}"
        report["hf_workaround"] = "official SQuAD 1.1 JSON download"

    splits_out: dict[str, Any] = {}
    try:
        for split in ("train", "validation"):
            rows, meta = load_squad_split(CACHE_DIR / "squad_official", split)
            contexts_sample: list[str] = []
            questions_sample: list[str] = []
            answers_sample: list[str] = []
            question_all_for_dup: list[str] = []
            missing_c = missing_q = malformed = usable = 0
            for ex in rows:
                context = ex.get("context") or ""
                question = ex.get("question") or ""
                if not str(context).strip():
                    missing_c += 1
                if not str(question).strip():
                    missing_q += 1
                question_all_for_dup.append(question)
                texts = (ex.get("answers") or {}).get("text") or []
                if not texts:
                    malformed += 1
                    continue
                usable += 1
                if len(contexts_sample) < 5000:
                    contexts_sample.append(context)
                    questions_sample.append(question)
                    answers_sample.append(texts[0])
            splits_out[split] = {
                "raw_examples": len(rows),
                "usable_examples": usable,
                "malformed_records": malformed,
                "missing_values": {"context": missing_c, "question": missing_q},
                "duplicates_question": near_duplicate_count(question_all_for_dup),
                "context_length": word_stats(contexts_sample),
                "question_length": word_stats(questions_sample),
                "answer_length": word_stats(answers_sample),
                "file_meta": meta,
                "length_stats_note": "Length stats on up to first 5000 usable examples.",
            }
        report["splits"] = splits_out
        report["raw_examples"] = sum(v["raw_examples"] for v in splits_out.values())
        report["usable_examples"] = sum(v["usable_examples"] for v in splits_out.values())
        report["official_test_note"] = (
            "SQuAD 1.1 public release exposes train + dev/validation; "
            "hidden test answers are not public. Experiment bipartitions "
            "official validation into exp val/test."
        )
    except Exception as exc:  # noqa: BLE001
        report["load_error"] = f"{type(exc).__name__}: {exc}"
        report["stop_condition"] = "required_dataset_unavailable_or_incompatible"
    return report


def audit_pubmed() -> dict[str, Any]:
    report: dict[str, Any] = {
        "name": "ccdv/pubmed-summarization",
        "task": "summarization",
        "source_citation": "PubMed scientific abstract summarization (ccdv HF mirror of scientific papers domain)",
        "license_note": "Inspect HF dataset card / provenance at audit time; scientific abstracts may have mixed upstream rights.",
        "domain": "biomedical / scientific articles",
        "language": "en",
        "fields_expected": ["article", "abstract"],
    }
    try:
        from datasets import load_dataset_builder, get_dataset_config_names

        try:
            report["available_configs"] = get_dataset_config_names("ccdv/pubmed-summarization")
        except Exception as exc:  # noqa: BLE001
            report["available_configs_error"] = str(exc)
        builder = load_dataset_builder("ccdv/pubmed-summarization", cache_dir=str(CACHE_DIR))
        info = builder.info
        report["description"] = (info.description or "")[:500]
        report["citation"] = (info.citation or "")[:500]
        report["license_from_builder"] = str(getattr(info, "license", None))
        report["version"] = str(getattr(info, "version", None))
        report["homepage"] = getattr(info, "homepage", None)
    except Exception as exc:  # noqa: BLE001
        report["builder_error"] = f"{type(exc).__name__}: {exc}"

    splits_out: dict[str, Any] = {}
    try:
        for split in ("train", "validation", "test"):
            ds = _load_hf_dataset("ccdv/pubmed-summarization", None, split)
            articles_sample: list[str] = []
            abstracts_sample: list[str] = []
            abstracts_for_dup: list[str] = []
            missing_a = missing_abs = usable = 0
            # Stream; only retain samples for length/dup diagnostics to avoid OOM.
            for i, ex in enumerate(ds):
                article = ex.get("article") or ""
                abstract = ex.get("abstract") or ""
                if not str(article).strip():
                    missing_a += 1
                if not str(abstract).strip():
                    missing_abs += 1
                if str(article).strip() and str(abstract).strip():
                    usable += 1
                if len(abstracts_for_dup) < 20000:
                    abstracts_for_dup.append(abstract)
                if len(articles_sample) < 2000:
                    articles_sample.append(article)
                    abstracts_sample.append(abstract)
            splits_out[split] = {
                "raw_examples": len(ds),
                "usable_examples": usable,
                "malformed_records": len(ds) - usable,
                "missing_values": {"article": missing_a, "abstract": missing_abs},
                "duplicates_abstract": near_duplicate_count(abstracts_for_dup),
                "article_length": word_stats(articles_sample),
                "abstract_length": word_stats(abstracts_sample),
                "length_stats_note": "Length stats on first 2000 rows; dup scan on first 20000 abstracts.",
                "features": list(ds.features.keys()) if hasattr(ds, "features") else None,
            }
        report["splits"] = splits_out
        report["raw_examples"] = sum(v["raw_examples"] for v in splits_out.values())
        report["usable_examples"] = sum(v["usable_examples"] for v in splits_out.values())
    except Exception as exc:  # noqa: BLE001
        report["load_error"] = f"{type(exc).__name__}: {exc}"
        report["stop_condition"] = "required_dataset_unavailable_or_incompatible"
    return report


def to_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Dataset Audit — Multi-task Bloom Rewrite Experiment",
        "",
        f"Generated (UTC): {audit['timestamp_utc']}",
        f"Seed reference: {SEED}",
        "",
        "Statistics below are measured from local files / Hugging Face downloads. "
        "No values are fabricated.",
        "",
    ]
    for key in ("figshare_bloom", "squad", "pubmed_summarization"):
        block = audit.get(key) or {}
        lines.append(f"## {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(block, indent=2)[:120000])
        lines.append("```")
        lines.append("")
    if audit.get("stop_conditions"):
        lines.append("## Stop conditions")
        lines.append("")
        for item in audit["stop_conditions"]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hf", action="store_true", help="Audit Figshare/Bloom only")
    args = parser.parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    audit: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "multitask_bloom_rewrite",
        "seed": SEED,
    }
    stop: list[str] = []
    audit["figshare_bloom"] = audit_figshare()
    if audit["figshare_bloom"].get("error"):
        stop.append("figshare_missing")

    if not args.skip_hf:
        audit["squad"] = audit_squad()
        if audit["squad"].get("stop_condition"):
            stop.append(audit["squad"]["stop_condition"] + ":squad")
        audit["pubmed_summarization"] = audit_pubmed()
        if audit["pubmed_summarization"].get("stop_condition"):
            stop.append(audit["pubmed_summarization"]["stop_condition"] + ":pubmed")
    else:
        audit["squad"] = {"skipped": True}
        audit["pubmed_summarization"] = {"skipped": True}

    audit["stop_conditions"] = stop
    out_json = REPORTS_DIR / "dataset_audit.json"
    out_md = REPORTS_DIR / "dataset_audit.md"
    out_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    out_md.write_text(to_markdown(audit), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if stop:
        print("STOP CONDITIONS:", stop)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
