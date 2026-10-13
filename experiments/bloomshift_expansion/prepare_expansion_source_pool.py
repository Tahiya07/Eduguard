#!/usr/bin/env python3
"""Build a reproducible pending-review pool of Figshare questions absent from known data.

This script does not generate or validate Bloom rewrites. It only screens exact
normalized source-text matches. Supply every known source inventory with --existing.
Near-duplicate and license review remain explicit human tasks.
"""
from __future__ import annotations
import argparse, csv, json, re, unicodedata
from pathlib import Path

TEXT_KEYS = ("source_question", "question", "original_question", "question_text", "prompt")
TEXT_COLUMNS = ("source_question", "question", "question_text", "Question", "text", "prompt", "Question Text")
TOKEN = re.compile(r"[a-z0-9]+", re.I)
FIELDS = ["source_id", "source_group_id", "source_question", "source_bloom_level",
          "original_label", "source_row", "review_status", "eligibility_status",
          "near_duplicate_status", "review_notes", "source_provenance"]

def normalize(text: str) -> str:
    return " ".join(TOKEN.findall(unicodedata.normalize("NFKC", text or "").casefold()))

def read_existing(path: Path) -> set[str]:
    seen: set[str] = set()
    try:
        raw = path.read_text(encoding="utf-8-sig")
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("data", data.get("records", []))
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    q = next((row.get(k) for k in TEXT_KEYS if row.get(k)), "")
                    if q and normalize(str(q)):
                        seen.add(normalize(str(q)))
            return seen
    except (json.JSONDecodeError, OSError):
        pass
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                q = next((row.get(k) for k in TEXT_COLUMNS if row.get(k)), "")
                if q and normalize(q):
                    seen.add(normalize(q))
    except OSError as exc:
        raise SystemExit(f"Cannot read inventory {path}: {exc}")
    return seen

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Figshare source CSV")
    ap.add_argument("--existing", action="append", default=[],
                    help="Existing JSON/CSV source inventory; repeat for each known inventory")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rows-per-part", type=int, default=700)
    args = ap.parse_args()
    if args.rows_per_part < 1:
        ap.error("--rows-per-part must be positive")
    with open(args.input, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []
    text_col = next((c for c in TEXT_COLUMNS if c in headers), None)
    if not text_col:
        ap.error(f"No question text column found. Headers: {headers}")
    level_col = next((c for c in ("bloom_level", "Bloom Level", "level", "cognitive_level") if c in headers), None)
    label_col = next((c for c in ("original_label", "label", "original_bloom_level") if c in headers), None)
    existing = set()
    for name in args.existing:
        existing.update(read_existing(Path(name)))
    unique, pool = set(), []
    duplicate_rows = 0
    for idx, row in enumerate(rows, start=2):
        question = (row.get(text_col) or "").strip()
        n = normalize(question)
        if not n:
            continue
        if n in unique:
            duplicate_rows += 1
            continue
        unique.add(n)
        if n in existing:
            continue
        sid = f"figshare_row_{idx}"
        pool.append({
            "source_id": sid, "source_group_id": sid, "source_question": question,
            "source_bloom_level": (row.get(level_col) or "").strip() if level_col else "",
            "original_label": (row.get(label_col) or "").strip() if label_col else "",
            "source_row": idx, "review_status": "pending",
            "eligibility_status": "not_reviewed", "near_duplicate_status": "not_checked",
            "review_notes": "", "source_provenance": str(Path(args.input)),
        })
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for part, start in enumerate(range(0, len(pool), args.rows_per_part), start=1):
        path = outdir / f"source_pool_pending_review_part_{part:02d}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(pool[start:start + args.rows_per_part])
    report = {
        "input": args.input, "input_rows": len(rows), "unique_normalized_source_texts": len(unique),
        "existing_inventory_files": args.existing, "existing_unique_normalized_texts": len(existing),
        "exact_unused_source_questions": len(pool), "duplicate_rows_removed_within_input": duplicate_rows,
        "rows_per_part": args.rows_per_part, "review_status": "all pending",
        "warning": "Count is only as complete as the supplied existing inventories. Exact matching does not detect paraphrases or near duplicates. No rewrite generation or human validation occurs.",
    }
    (outdir / "source_pool_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
