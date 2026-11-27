#!/usr/bin/env python
"""Build the blinded human-annotation package for the journal paper.

Sheets (one copy per rater, rows shuffled, no system names):
  01_bloom_rewrite  - 48 source questions (8 per requested Bloom level) x 3 versions:
                      EduGuard v3.9 LoRA, base Qwen2.5-1.5B-Instruct, reference rewrite
  02_qa             - the 62 stratified closed-book QA items x 2 systems (v3.9, base)
  03_summarization  - the 41 stratified summarization items x 2 systems (v3.9, base)
Plus: an answer key (do not share with raters), a codebook of allowed values, guidelines,
and a manifest with seeds and SHA-256 hashes of every input file.

Usage:
  python experiments/multitask_bloom_rewrite/scripts/build_human_annotation.py
Then fill the *_R1.csv / *_R2.csv files and run analyze_human_annotation.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
REPO = EXP.parents[1]
RESULTS = EXP / "results"
# Validation outputs. The v3.9 folder was overwritten by a test-split run, so the v3.9 texts come from
# the exports that survived in the backup (all 341 rewrites + the 62 QA / 41 summary sample items);
# base outputs and full source texts come from the intact base validation run.
LORA_DIR = RESULTS / "qwen15b_multitask_eduguard_v3_9_20261127_validation_backup"
BASE_DIR = RESULTS / "qwen15b_multitask_eduguard_v3_9_20261127_base_validation_backup"
BLOOM_DATA = REPO / "data" / "multitask_bloom_rewrite_separate_eduguard_v3_9" / "bloom_rewrite" / "validation.jsonl"
OUT = REPO / "paper" / "elsevier" / "human_annotation"

SEED = 42
LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
PER_LEVEL = 8
EXCEL_CELL_LIMIT = 32000
SYSTEMS = {"lora": "eduguard_v3_9_lora", "base": "base_qwen2.5_1.5b_instruct", "reference": "reference_rewrite"}

BLOOM_RATER_COLS = [
    "rater_id", "perceived_bloom_level", "matches_requested_level", "topic_preserved",
    "self_contained", "factually_sound", "clarity_1to5", "exam_usability_1to5", "notes",
]
QA_RATER_COLS = ["rater_id", "correctness", "contains_false_statement", "completeness_1to5", "on_topic_and_concise", "notes"]
SUM_RATER_COLS = [
    "rater_id", "faithful", "unsupported_claim_count", "coverage_1to5", "fluency_1to5", "length", "notes",
]

CODEBOOK = [
    # sheet, field, allowed values, definition
    ("bloom_rewrite", "perceived_bloom_level", "Remember | Understand | Apply | Analyze | Evaluate | Create",
     "The level the rewritten question actually demands, judged from the question alone. Decide this BEFORE reading requested_bloom_level."),
    ("bloom_rewrite", "matches_requested_level", "yes | no", "yes if perceived_bloom_level equals requested_bloom_level."),
    ("bloom_rewrite", "topic_preserved", "yes | no", "The rewrite tests the same subject and concept as the source question."),
    ("bloom_rewrite", "self_contained", "yes | no",
     "A student can answer it as written. 'no' if it refers to material that was never given (e.g. 'the supplied diagram', 'the following data') or invents arbitrary constraints."),
    ("bloom_rewrite", "factually_sound", "yes | no | n/a", "No false facts or impossible premises (e.g. 'name the two conditions for deadlock' when there are four)."),
    ("bloom_rewrite", "clarity_1to5", "1 2 3 4 5", "1 = confusing or ungrammatical; 3 = understandable with effort; 5 = clear, exam-ready wording."),
    ("bloom_rewrite", "exam_usability_1to5", "1 2 3 4 5", "1 = unusable; 3 = usable after substantial editing; 5 = usable as is in a university exam."),
    ("qa", "correctness", "correct | partially_correct | incorrect",
     "Against the reference and your subject knowledge. partially_correct = right idea but missing or wrong in an important detail."),
    ("qa", "contains_false_statement", "yes | no", "yes if any sentence is factually wrong, even if the main answer is right."),
    ("qa", "completeness_1to5", "1 2 3 4 5", "Coverage of the key points a good answer needs (1 = none, 5 = all)."),
    ("qa", "on_topic_and_concise", "yes | no", "Answers the question asked without padding or unrelated content."),
    ("summarization", "faithful", "yes | no", "yes only if every claim in system_summary is supported by source_text."),
    ("summarization", "unsupported_claim_count", "0 1 2 ...", "Number of claims not supported by source_text (0 if faithful)."),
    ("summarization", "coverage_1to5", "1 2 3 4 5", "How well the main points of the source are covered (compare with reference_summary)."),
    ("summarization", "fluency_1to5", "1 2 3 4 5", "Readability and grammar."),
    ("summarization", "length", "too_short | appropriate | too_long", "Length relative to the reference summary and the source."),
    ("all", "rater_id", "free text (e.g. R1, R2)", "Your rater code; the same on every row."),
    ("all", "notes", "free text", "Optional: anything that explains a low score."),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[f"{rec['task']}::{rec['id']}"] = rec
    return out


def excel_safe(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    if len(text) > EXCEL_CELL_LIMIT:
        text = text[:EXCEL_CELL_LIMIT] + " […truncated for spreadsheet cell limit; full text in the key file's source_id]"
    # Keep formula-like openings from being executed by spreadsheet software.
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:  # BOM so Excel opens UTF-8 correctly
        writer = csv.DictWriter(fh, fieldnames=header, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def sample_ids_from_existing_sheet(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return [row["id"] for row in csv.DictReader(fh)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--raters", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    base = load_predictions(BASE_DIR / "predictions.jsonl")
    lora: dict[str, dict] = {}
    with (LORA_DIR / "human_eval_predictions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                lora[f"bloom_rewrite::{r['id']}"] = {"task": "bloom_rewrite", "id": r["id"],
                                                     "source_question": r["original_question"],
                                                     "target_bloom_level": r["requested_target_level"],
                                                     "prediction": r["generated_rewrite"]}
    data = {}
    with BLOOM_DATA.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                data[d["id"]] = d
    for k, rec in lora.items():
        d = data.get(rec["id"], {})
        rec.update(reference=d.get("target_rewrite", ""), source_bloom_level=d.get("source_bloom_level", ""),
                   pair_type=d.get("pair_type", ""))
    for sheet, task in (("human_eval_qa.csv", "qa"), ("human_eval_summarization.csv", "summarization")):
        with (LORA_DIR / sheet).open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                k = f"{task}::{row['id']}"
                b = base.get(k, {})
                lora[k] = {"task": task, "id": row["id"], "prediction": row["system_output"],
                           "reference": b.get("reference") or row["reference"],
                           # full texts from the base run (the old export truncated long sources)
                           "question": b.get("question") or row["input"], "article": b.get("article") or row["input"],
                           "qa_type": b.get("qa_type"), "discipline_group": b.get("discipline_group"),
                           "source_type": b.get("source_type"), "length_category": b.get("length_category")}
    key_rows: list[dict] = []

    # ---- 01 Bloom rewrite: 8 sources per requested level, 3 versions each -------------------
    by_level: dict[str, list[dict]] = defaultdict(list)
    for k, rec in sorted(lora.items()):
        if rec["task"] == "bloom_rewrite" and k in base and (rec.get("reference") or "").strip():
            by_level[rec["target_bloom_level"]].append(rec)
    bloom_rows = []
    for level in LEVELS:
        pool = by_level[level]
        chosen = rng.sample(pool, min(PER_LEVEL, len(pool)))
        for rec in chosen:
            key = f"bloom_rewrite::{rec['id']}"
            for system, text in (("lora", rec["prediction"]), ("base", base[key]["prediction"]), ("reference", rec["reference"])):
                bloom_rows.append({"_system": system, "_rec": rec, "rewritten_question": text})
    rng.shuffle(bloom_rows)
    bloom_sheet = []
    for i, row in enumerate(bloom_rows, 1):
        rec = row["_rec"]
        item = f"B{i:03d}"
        bloom_sheet.append({"item_id": item, "source_question": excel_safe(rec["source_question"]),
                            "requested_bloom_level": rec["target_bloom_level"],
                            "rewritten_question": excel_safe(row["rewritten_question"])})
        key_rows.append({"item_id": item, "sheet": "01_bloom_rewrite", "system": SYSTEMS[row["_system"]],
                         "source_id": rec["id"], "requested_or_type": rec["target_bloom_level"],
                         "source_bloom_level": rec.get("source_bloom_level", ""), "pair_type": rec.get("pair_type", ""),
                         "classifier_level": base[f"bloom_rewrite::{rec['id']}"].get("classifier_prediction", "")
                         if row["_system"] == "base" else ""})

    # ---- 02 QA: reuse the evaluation's stratified sample (62 ids) ------------------------------
    qa_ids = sample_ids_from_existing_sheet(LORA_DIR / "human_eval_qa.csv")
    qa_rows = []
    for qid in qa_ids:
        k = f"qa::{qid}"
        if k in lora and k in base:
            for system, rec in (("lora", lora[k]), ("base", base[k])):
                qa_rows.append({"_system": system, "_rec": rec})
    rng.shuffle(qa_rows)
    qa_sheet = []
    for i, row in enumerate(qa_rows, 1):
        rec = row["_rec"]
        item = f"Q{i:03d}"
        qa_sheet.append({"item_id": item, "question": excel_safe(rec["question"]),
                         "reference_answer": excel_safe(rec["reference"]), "system_answer": excel_safe(rec["prediction"])})
        key_rows.append({"item_id": item, "sheet": "02_qa", "system": SYSTEMS[row["_system"]], "source_id": rec["id"],
                         "requested_or_type": rec.get("qa_type") or "", "discipline_group": rec.get("discipline_group") or ""})

    # ---- 03 Summarization: reuse the evaluation's stratified sample (41 ids) -------------------
    sum_ids = sample_ids_from_existing_sheet(LORA_DIR / "human_eval_summarization.csv")
    sum_rows = []
    for sid in sum_ids:
        k = f"summarization::{sid}"
        if k in lora and k in base:
            for system, rec in (("lora", lora[k]), ("base", base[k])):
                sum_rows.append({"_system": system, "_rec": rec})
    rng.shuffle(sum_rows)
    sum_sheet = []
    for i, row in enumerate(sum_rows, 1):
        rec = row["_rec"]
        item = f"S{i:03d}"
        sum_sheet.append({"item_id": item, "source_text": excel_safe(rec.get("article") or ""),
                          "reference_summary": excel_safe(rec["reference"]), "system_summary": excel_safe(rec["prediction"])})
        key_rows.append({"item_id": item, "sheet": "03_summarization", "system": SYSTEMS[row["_system"]], "source_id": rec["id"],
                         "requested_or_type": rec.get("source_type") or "", "length_category": rec.get("length_category") or ""})

    # ---- write rater copies, key, codebook ----------------------------------------------------
    sheets = [
        ("01_bloom_rewrite", ["item_id", "source_question", "requested_bloom_level", "rewritten_question"], BLOOM_RATER_COLS, bloom_sheet),
        ("02_qa", ["item_id", "question", "reference_answer", "system_answer"], QA_RATER_COLS, qa_sheet),
        ("03_summarization", ["item_id", "source_text", "reference_summary", "system_summary"], SUM_RATER_COLS, sum_sheet),
    ]
    written = []
    for name, cols, rater_cols, rows in sheets:
        for r in range(1, args.raters + 1):
            path = out / f"{name}_R{r}.csv"
            write_csv(path, cols + rater_cols, [{**row, "rater_id": f"R{r}"} for row in rows])
            written.append(path.name)
    key_header = ["item_id", "sheet", "system", "source_id", "requested_or_type", "source_bloom_level", "pair_type",
                  "classifier_level", "discipline_group", "length_category"]
    write_csv(out / "04_annotation_key_DO_NOT_SHARE.csv", key_header, key_rows)
    write_csv(out / "05_codebook.csv", ["sheet", "field", "allowed_values", "definition"],
              [dict(zip(["sheet", "field", "allowed_values", "definition"], row)) for row in CODEBOOK])

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "raters": args.raters,
        "counts": {"bloom_rewrite_rows": len(bloom_sheet), "bloom_sources": len(bloom_sheet) // 3,
                   "qa_rows": len(qa_sheet), "summarization_rows": len(sum_sheet)},
        "systems": SYSTEMS,
        "sampling": {
            "bloom_rewrite": f"{PER_LEVEL} validation sources per requested level (random.Random({SEED}).sample, sorted by id)",
            "qa": "the 62 items of results/.../human_eval_qa.csv (stratified by qa_type, seed 42)",
            "summarization": "the 41 items of results/.../human_eval_summarization.csv (stratified by source_type, seed 42)",
        },
        "blinding": "rows shuffled with the same seed; system identity only in 04_annotation_key_DO_NOT_SHARE.csv",
        "inputs_sha256": {
            str(p.relative_to(REPO)): sha256(p)
            for p in [LORA_DIR / "human_eval_predictions.jsonl", BASE_DIR / "predictions.jsonl", BLOOM_DATA,
                      LORA_DIR / "human_eval_qa.csv", LORA_DIR / "human_eval_summarization.csv"]
        },
        "files": written + ["04_annotation_key_DO_NOT_SHARE.csv", "05_codebook.csv"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print("Written to", out)


if __name__ == "__main__":
    main()
