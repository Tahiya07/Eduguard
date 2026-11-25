#!/usr/bin/env python
"""Build a versioned task-separated dataset whose summarization files contain only eligible rows.

Reads (never writes) the original dataset directory and writes a NEW directory:

    <output>/bloom_rewrite/{split}.jsonl   byte-identical copies (Bloom rewriting eligibility unchanged)
    <output>/qa/{split}.jsonl              byte-identical copies (QA eligibility unchanged)
    <output>/summarization/{split}.jsonl   eligible rows only, original lines unchanged, original order, same split
    <output>/summarization/summarization_eligibility.json   per-row status, reasons, recovery status, provenance
    <output>/summarization/summarization_exclusions.csv     human-readable exclusion list
    <output>/dataset_manifest.json         original manifest with updated summarization counts/hashes + filter record
    <output>/build_report.json             input/output hashes, counts, policy

Filtering happens within each existing split; no row moves between splits. Deterministic: no timestamps.
CPU only. Run from the repository root:

    python experiments/multitask_bloom_rewrite/scripts/build_clean_summarization_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for p in (EXPERIMENT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import summarization_quality as Q  # noqa: E402

SPLITS = ("train", "validation", "test")
DEFAULT_INPUT = "data/multitask_bloom_rewrite_separate"
DEFAULT_OUTPUT = "data/multitask_bloom_rewrite_separate_sumclean_v1"
ELIGIBILITY_FILE = "summarization/summarization_eligibility.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def build(input_dir: Path, output_dir: Path, recovery_file: Path | None = None) -> dict:
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if output_dir == input_dir:
        raise SystemExit("Refusing to write into the original dataset directory.")
    before = tree_hashes(input_dir)
    manifest = json.loads((input_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    recoveries = json.loads(recovery_file.read_text(encoding="utf-8")) if recovery_file else []

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for task in ("bloom_rewrite", "qa"):
        (output_dir / task).mkdir(parents=True)
        for split in SPLITS:
            shutil.copyfile(input_dir / task / f"{split}.jsonl", output_dir / task / f"{split}.jsonl")
    (output_dir / "summarization").mkdir(parents=True)

    per_row, kept_counts, total_counts, reason_counts = [], {}, {}, {}
    placeholder_stats = {}
    for split in SPLITS:
        src = input_dir / "summarization" / f"{split}.jsonl"
        lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
        kept_lines = []
        reasons_here = Counter()
        stats = Counter()
        for n, line in enumerate(lines, 1):
            row = json.loads(line)
            if row.get("split") != split or row.get("task") != "summarization":
                raise SystemExit(f"{src}:{n}: row split/task mismatch")
            abstract_ph = Q.math_placeholders(row["abstract"])
            article_ph = Q.math_placeholders(row["article"])
            stats["rows"] += 1
            stats["abstracts_with_math_placeholders"] += bool(abstract_ph)
            stats["abstract_math_placeholder_occurrences"] += len(abstract_ph)
            stats["articles_with_math_placeholders"] += bool(article_ph)
            stats["article_math_placeholder_occurrences"] += len(article_ph)
            new_row, applied = Q.apply_verified_recoveries(row, recoveries)
            if applied:
                from prompts import build_generation_prompt, build_prompt_only_text, build_sft_text
                new_row["sft_text"] = build_sft_text("summarization", new_row)
                new_row["prompt_text"] = build_prompt_only_text("summarization", new_row)
                new_row["generation_prompt"] = build_generation_prompt("summarization", new_row)
            issues = Q.detect_issues(new_row)
            status = "excluded" if issues else "eligible"
            if status == "eligible":
                kept_lines.append(line if not applied else json.dumps(new_row, ensure_ascii=False) + "\n")
            for r in issues:
                reasons_here[r] += 1
            retention_note = None
            if status == "eligible" and article_ph:
                retention_note = ("Retained: the target abstract is clean; the input article's @xmath placeholders are kept as "
                                  "input-only artefacts (density {:.1%}, below the {:.0%} limit), so the model is never trained to emit them."
                                  .format(Q.article_placeholder_density(row["article"]), Q.MAX_ARTICLE_MATH_PLACEHOLDER_DENSITY))
            per_row.append({
                "id": row["id"], "split": split, "status": status, "reasons": issues,
                "source_line": n, "row_sha256": Q.row_sha256(new_row), "original_row_sha256": Q.row_sha256(row),
                "abstract_math_placeholders": len(abstract_ph), "article_math_placeholders": len(article_ph),
                "article_math_placeholder_density": round(Q.article_placeholder_density(row["article"]), 6),
                "recovery_status": Q.recovery_status(new_row, applied), "recoveries_applied": applied,
                "retention_note": retention_note,
            })
        (output_dir / "summarization" / f"{split}.jsonl").write_text("".join(kept_lines), encoding="utf-8", newline="")
        kept_counts[split], total_counts[split] = len(kept_lines), len(lines)
        reason_counts[split] = dict(sorted(reasons_here.items()))
        placeholder_stats[split] = dict(stats)

    eligibility = {
        "policy_version": Q.POLICY_VERSION,
        "source_dataset": Q.SOURCE_DATASET,
        "source_revision_pinned": False,
        "original_dataset_dir": str(input_dir.relative_to(REPO_ROOT)).replace("\\", "/") if input_dir.is_relative_to(REPO_ROOT) else str(input_dir),
        "original_summarization_files_sha256": {f"summarization/{s}.jsonl": before[f"summarization/{s}.jsonl"] for s in SPLITS},
        "rules": [{"name": n, "description": d} for n, d in Q.RULES],
        "thresholds": {"min_abstract_words": Q.MIN_ABSTRACT_WORDS, "min_article_words": Q.MIN_ARTICLE_WORDS,
                       "max_article_math_placeholder_density": Q.MAX_ARTICLE_MATH_PLACEHOLDER_DENSITY,
                       "threshold_basis": "train split distribution only; no model-performance input"},
        "recovery": {
            "verified_sources_available": bool(recoveries),
            "recovery_entries_supplied": len(recoveries),
            "finding": ("No math expression is recoverable: ccdv/arxiv-summarization stores only @xmathN tokens, abstract "
                        "numbering is independent of the article, and rows carry no arXiv identifier or title to trace the "
                        "original LaTeX. No expression was invented or substituted."),
        },
        "counts": {s: {"original": total_counts[s], "eligible": kept_counts[s], "excluded": total_counts[s] - kept_counts[s]} for s in SPLITS},
        "exclusion_reason_counts": reason_counts,
        "placeholder_statistics": placeholder_stats,
        "eligible_ids": {s: [r["id"] for r in per_row if r["split"] == s and r["status"] == "eligible"] for s in SPLITS},
        "rows": per_row,
    }
    elig_path = output_dir / ELIGIBILITY_FILE
    elig_path.write_text(json.dumps(eligibility, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    with (output_dir / "summarization" / "summarization_exclusions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "reasons", "abstract_math_placeholders", "article_math_placeholders", "recovery_status", "source_line"])
        for r in per_row:
            if r["status"] == "excluded":
                w.writerow([r["id"], r["split"], ";".join(r["reasons"]), r["abstract_math_placeholders"],
                            r["article_math_placeholders"], r["recovery_status"], r["source_line"]])

    new_manifest = json.loads(json.dumps(manifest))
    for s in SPLITS:
        new_manifest["counts"][s]["summarization"] = kept_counts[s]
        rel = f"summarization/{s}.jsonl"
        new_manifest["generated_files_sha256"][rel] = sha256_file(output_dir / rel)
    new_manifest["summarization_quality_filter"] = {
        "policy_version": Q.POLICY_VERSION,
        "derived_from_dataset_dir": eligibility["original_dataset_dir"],
        "derived_from_manifest_sha256": before["dataset_manifest.json"],
        "eligibility_file": ELIGIBILITY_FILE,
        "eligibility_file_sha256": sha256_file(elig_path),
        "counts": eligibility["counts"],
        "unchanged_tasks": {"bloom_rewrite": "byte-identical copy", "qa": "byte-identical copy"},
        "split_policy": "rows filtered within their original split; no row moved; original order kept",
        "builder": "experiments/multitask_bloom_rewrite/scripts/build_clean_summarization_dataset.py",
        "note": "summarization_source.* records upstream preparation history; current per-split counts are in counts.",
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(new_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    after = tree_hashes(input_dir)
    if before != after:
        raise SystemExit("Original dataset directory changed during the build.")
    report = {
        "policy_version": Q.POLICY_VERSION,
        "input_dir_sha256_before": before, "input_dir_sha256_after": after, "input_unchanged": before == after,
        "output_files_sha256": {k: v for k, v in tree_hashes(output_dir).items()},
        "counts": eligibility["counts"], "exclusion_reason_counts": reason_counts,
        "placeholder_statistics": placeholder_stats,
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--recovery-file", default=None, help="Optional JSON list of verified recovery entries (none exist today).")
    a = ap.parse_args()
    resolve = lambda p: (Path(p) if Path(p).is_absolute() else REPO_ROOT / p)  # noqa: E731
    report = build(resolve(a.input_dir), resolve(a.output_dir), resolve(a.recovery_file) if a.recovery_file else None)
    print(json.dumps({k: report[k] for k in ("counts", "exclusion_reason_counts", "placeholder_statistics", "input_unchanged")}, indent=1))


if __name__ == "__main__":
    main()
