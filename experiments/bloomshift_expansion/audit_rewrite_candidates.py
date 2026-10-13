#!/usr/bin/env python3
"""Deterministic consistency audit for raw EduGuard Bloom rewrite candidates.

This is a screening tool, not a semantic-equivalence or Bloom-level validator.
It reports mechanical contradictions and risk signals for subsequent human review.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

LEVELS = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}
REQUIRED = (
    "target_rewrite",
    "target_bloom_level",
    "opening_verb",
    "feasibility",
    "cognitive_operation",
    "target_operation_evidence",
    "preserved_elements",
    "changed_or_removed_demands",
    "semantic_risks",
    "rationale",
    "feasibility_reason",
)
HIGHER_ORDER_CUES = {
    "Remember": {"evaluate", "evaluates", "evaluation", "justify", "justifies", "critique",
                 "assess", "assesses", "recommend", "recommendations", "defend"},
    "Understand": set(),
    "Apply": {"evaluate", "evaluation", "justify", "critique", "assess", "recommend"},
    "Analyze": set(),
    "Evaluate": set(),
    "Create": set(),
}


def norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def first_word(text):
    match = re.search(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", str(text or "").strip())
    return match.group(0).casefold() if match else ""


def candidate_for(record):
    value = record.get("parsed_candidate")
    return value if isinstance(value, dict) else None


def audit(record):
    issues = []
    candidate = candidate_for(record)
    generation_id = record.get("generation_id", "")
    source_id = record.get("source_id", "")
    if candidate is None:
        return {
            "generation_id": generation_id,
            "source_id": source_id,
            "target_bloom_level": record.get("target_bloom_level", ""),
            "rewrite": "",
            "flags": "PARSED_CANDIDATE_MISSING_OR_NOT_OBJECT",
            "flag_count": 1,
        }

    missing = [key for key in REQUIRED if key not in candidate]
    if missing:
        issues.append("MISSING_FIELDS:" + ",".join(missing))

    target = str(record.get("target_bloom_level", "")).strip()
    candidate_target = str(candidate.get("target_bloom_level", "")).strip()
    if candidate_target not in LEVELS:
        issues.append("INVALID_CANDIDATE_TARGET_LEVEL")
    if target and candidate_target and target != candidate_target:
        issues.append("TARGET_LEVEL_DIFFERS_FROM_QUEUE_RECORD")

    feasibility = str(candidate.get("feasibility", "")).strip()
    rewrite = str(candidate.get("target_rewrite", "") or "").strip()
    declared_verb = str(candidate.get("opening_verb", "") or "").strip()
    actual_first = first_word(rewrite)

    if feasibility not in {"feasible", "not_feasible"}:
        issues.append("INVALID_FEASIBILITY_VALUE")
    if feasibility == "feasible" and not rewrite:
        issues.append("FEASIBLE_BUT_REWRITE_EMPTY")
    if feasibility == "not_feasible" and rewrite:
        issues.append("NOT_FEASIBLE_BUT_REWRITE_NONEMPTY")
    if feasibility == "not_feasible" and declared_verb:
        issues.append("NOT_FEASIBLE_BUT_OPENING_VERB_NONEMPTY")
    if feasibility == "feasible" and rewrite and not declared_verb:
        issues.append("FEASIBLE_BUT_OPENING_VERB_EMPTY")
    if feasibility == "feasible" and rewrite and declared_verb and actual_first != declared_verb.casefold():
        issues.append("DECLARED_OPENING_VERB_DOES_NOT_MATCH_REWRITE_FIRST_WORD")

    source_question = str(record.get("source_question", "") or "").strip()
    if rewrite and source_question and norm(rewrite) == norm(source_question):
        issues.append("REWRITE_IDENTICAL_TO_SOURCE")

    if candidate_target in HIGHER_ORDER_CUES and rewrite:
        tokens = set(re.findall(r"[a-z]+", rewrite.casefold()))
        cues = sorted(tokens.intersection(HIGHER_ORDER_CUES[candidate_target]))
        if cues:
            issues.append("POSSIBLE_COGNITIVE_DEMAND_CONFLICT:" + ",".join(cues))

    # These fields are model claims, not proof. Flag empty or self-contradictory evidence.
    if feasibility == "feasible" and not str(candidate.get("target_operation_evidence", "")).strip():
        issues.append("TARGET_OPERATION_EVIDENCE_EMPTY")
    if feasibility == "feasible" and not str(candidate.get("cognitive_operation", "")).strip():
        issues.append("COGNITIVE_OPERATION_EMPTY")
    if isinstance(candidate.get("semantic_risks"), list):
        risks = candidate["semantic_risks"]
        if any(norm(item) in {"none", "no risks", "no semantic risks"} for item in risks):
            issues.append("MODEL_CLAIMS_NO_SEMANTIC_RISK_REVIEW_MANUALLY")
    else:
        issues.append("SEMANTIC_RISKS_NOT_A_LIST")

    return {
        "generation_id": generation_id,
        "source_id": source_id,
        "target_bloom_level": target or candidate_target,
        "feasibility": feasibility,
        "opening_verb_declared": declared_verb,
        "opening_word_actual": actual_first,
        "rewrite": rewrite,
        "flags": ";".join(issues),
        "flag_count": len(issues),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw candidate JSONL generated by generate_rewrite_candidates.py")
    parser.add_argument("--report", default="", help="CSV report path (default: <input stem>_audit.csv)")
    args = parser.parse_args()

    input_path = Path(args.input)
    report_path = Path(args.report) if args.report else input_path.with_name(input_path.stem + "_audit.csv")
    records = []
    with input_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                records.append({
                    "generation_id": f"line_{line_number}",
                    "source_id": "",
                    "target_bloom_level": "",
                    "parsed_candidate": None,
                    "_json_error": str(exc),
                })

    rows = [audit(record) for record in records]
    for index, record in enumerate(records):
        if record.get("_json_error"):
            rows[index]["flags"] = "INVALID_JSONL_LINE:" + record["_json_error"]
            rows[index]["flag_count"] = 1

    columns = [
        "generation_id", "source_id", "target_bloom_level", "feasibility",
        "opening_verb_declared", "opening_word_actual", "rewrite", "flags", "flag_count",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    issue_counts = Counter()
    for row in rows:
        for flag in filter(None, row.get("flags", "").split(";")):
            issue_counts[flag.split(":", 1)[0]] += 1

    print(json.dumps({
        "input": str(input_path),
        "records_audited": len(rows),
        "records_with_flags": sum(bool(row.get("flags")) for row in rows),
        "records_without_mechanical_flags": sum(not row.get("flags") for row in rows),
        "flag_counts": dict(sorted(issue_counts.items())),
        "report": str(report_path),
        "warning": "Flags are screening signals, not automatic semantic or Bloom-level judgments. Manually review every candidate before training."
    }, indent=2))


if __name__ == "__main__":
    main()
