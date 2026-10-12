#!/usr/bin/env python3
"""Create a prompt queue only for manually approved, target-feasible source-level pairs.

Input source CSVs must contain review_status=approved. A separate feasibility CSV
must contain source_id,target_bloom_level,feasible,reviewer_notes. This script does
not call a model and does not validate generated rewrites.
"""
import argparse, csv, hashlib, json
from pathlib import Path

LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
OPENERS = {
    "Remember": "Identify, Define, List, State, Recall, Name",
    "Understand": "Explain, Summarize, Interpret, Classify, Describe, Compare",
    "Apply": "Apply, Calculate, Demonstrate, Implement, Use, Solve",
    "Analyze": "Analyze, Differentiate, Examine, Decompose, Trace, Investigate",
    "Evaluate": "Evaluate, Justify, Critique, Assess, Defend, Validate",
    "Create": "Design, Construct, Develop, Formulate, Propose, Compose",
}
PROMPT = """Rewrite the source question for the requested revised Bloom level.
Start with an appropriate action verb rather than a WH-question opening. The verb is
not proof of Bloom alignment: the requested reasoning operation must genuinely match
the target level. Preserve the source's subject, named entities, quantities, constraints,
technical terms, negation, and supplied context. Add no unsupported facts or requirements.
Do not answer the question. If a valid transformation is not feasible without extra
context, return feasibility=not_feasible and explain why. Return JSON with target_rewrite,
target_bloom_level, opening_verb, semantic_risks, feasibility, rationale. Do not claim
human validation.
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-csv", action="append", required=True, help="Pending-review source CSV part; repeat")
    p.add_argument("--feasibility-csv", required=True, help="Human-reviewed source_id,target_bloom_level,feasible,reviewer_notes")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    approved = {}
    for path in a.source_csv:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("review_status", "").strip().casefold() == "approved":
                    approved[row["source_id"]] = row
    feasible = []
    with open(a.feasibility_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            level = row.get("target_bloom_level", "").strip()
            if (row.get("source_id") in approved and level in LEVELS
                and row.get("feasible", "").strip().casefold() in {"yes", "true", "1"}):
                feasible.append((approved[row["source_id"]], level, row.get("reviewer_notes", "")))
    jobs = []
    for source, level, notes in feasible:
        q = source["source_question"]
        key = source["source_id"] + "|" + level
        jobs.append({
            "generation_id": "gen_" + hashlib.sha256(key.encode()).hexdigest()[:16],
            "source_id": source["source_id"], "source_group_id": source.get("source_group_id") or source["source_id"],
            "source_question": q, "source_bloom_level": source.get("source_bloom_level", ""),
            "target_bloom_level": level, "preferred_opening_verbs": OPENERS[level],
            "feasibility_review_notes": notes, "prompt_version": "eduguard-bloomshift-expansion-v1",
            "prompt": PROMPT, "status": "pending_generation", "model_id": "",
            "generation_parameters": {}, "human_review_status": "not_reviewed"
        })
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(json.dumps({"approved_sources": len(approved), "feasible_source_target_pairs": len(jobs),
                      "status": "prompt queue only; no rewrites generated or validated",
                      "output": str(out)}, indent=2))

if __name__ == "__main__":
    main()
