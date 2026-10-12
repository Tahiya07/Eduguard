#!/usr/bin/env python3
"""Build an unvalidated draft-generation queue from BloomShift expansion source candidates.

Creates up to five source-to-different-level jobs per source. It does not approve source
quality, guarantee target feasibility, generate text, or mark examples as validated.
The local generation model must explicitly return feasibility=not_feasible when it cannot
preserve the source's content/context while eliciting the target cognitive operation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
OPENERS = {
    "Remember": ["Identify", "Define", "List", "State", "Recall", "Name", "Label", "Match", "Select", "Locate", "Recognize", "Enumerate", "Quote", "Reproduce", "Sequence", "Count", "Tabulate", "Retrieve"],
    "Understand": ["Explain", "Summarize", "Interpret", "Classify", "Paraphrase", "Illustrate", "Describe", "Distinguish", "Compare", "Infer", "Translate", "Exemplify", "Outline", "Restate", "Categorize", "Predict", "Clarify", "Relate"],
    "Apply": ["Apply", "Calculate", "Demonstrate", "Implement", "Use", "Solve", "Execute", "Compute", "Operate", "Modify", "Illustrate", "Prepare", "Construct", "Complete", "Determine", "Perform", "Adapt", "Show"],
    "Analyze": ["Analyze", "Differentiate", "Examine", "Decompose", "Trace", "Investigate", "Categorize", "Contrast", "Dissect", "Separate", "Attribute", "Organize", "Map", "Relate", "Diagnose", "Inspect", "Compare", "Distinguish"],
    "Evaluate": ["Evaluate", "Justify", "Critique", "Assess", "Defend", "Validate", "Appraise", "Rank", "Recommend", "Judge", "Verify", "Prioritize", "Argue", "Rate", "Support", "Select", "Challenge", "Weigh"],
    "Create": ["Design", "Construct", "Develop", "Formulate", "Propose", "Compose", "Devise", "Generate", "Invent", "Plan", "Synthesize", "Engineer", "Produce", "Assemble", "Draft", "Build", "Create", "Model"],
}
PROMPT = """Produce one candidate question rewritten for the specified Revised Bloom target level.
Start the rewritten question with a suitable action verb from the preferred verb bank, but do
not treat the verb alone as proof of alignment. The actual cognitive operation must fit the
target level. Preserve the source's topic, intended object, named entities, technical terms,
numbers, quantities, count requirements, constraints, negation, and supplied context. Do not
answer the question, add unsupported facts, assume missing diagrams/data, or silently remove
requirements. Do not turn a question into a generic learning objective. Keep it answerable.
If a meaning-preserving, context-preserving transformation that genuinely elicits the target
operation is not feasible, return feasibility="not_feasible" and explain the obstacle; do not
invent a scenario to make it possible.

Return exactly one JSON object with these keys:
{"target_rewrite":"... or empty if not feasible",
 "target_bloom_level":"one of the six levels",
 "opening_verb":"first imperative/action verb or empty",
 "feasibility":"feasible or not_feasible",
 "cognitive_operation":"what the learner must actually do",
 "preserved_elements":["specific source elements retained"],
 "semantic_risks":["possible meaning/context changes, if any"],
 "rationale":"brief reason the task fits the target level"}
Do not claim human validation or certainty about semantic equivalence.
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--targets", default="all", help="all or comma-separated levels")
    p.add_argument("--limit-sources", type=int, default=0, help="0 = all rows; for smoke tests")
    p.add_argument("--include-same-level", action="store_true", help="include same-level paraphrase jobs; not recommended for cross-level transformation evaluation")
    a = p.parse_args()
    targets = LEVELS if a.targets.casefold() == "all" else [x.strip() for x in a.targets.split(",") if x.strip()]
    unknown = set(targets) - set(LEVELS)
    if unknown:
        p.error("Unknown target levels: " + ", ".join(sorted(unknown)))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count_sources = count_jobs = 0
    seen_ids = set()
    with open(a.source_csv, encoding="utf-8-sig", newline="") as src, out.open("w", encoding="utf-8") as dst:
        for row in csv.DictReader(src):
            sid = (row.get("source_id") or "").strip()
            question = (row.get("source_question") or "").strip()
            if not sid or not question or sid in seen_ids:
                continue
            seen_ids.add(sid)
            count_sources += 1
            source_level = (row.get("source_bloom_level") or "").strip()
            for level in targets:
                if not a.include_same_level and level == source_level:
                    continue
                key = sid + "|" + level
                job = {
                    "generation_id": "gen_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                    "source_id": sid,
                    "source_group_id": (row.get("source_group_id") or sid).strip(),
                    "source_question": question,
                    "source_bloom_level": source_level,
                    "target_bloom_level": level,
                    "preferred_opening_verbs": ", ".join(OPENERS[level][(int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16) % len(OPENERS[level])):] + OPENERS[level][:(int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16) % len(OPENERS[level]))]),
                    "prompt_version": "bloomshift-expansion-v1.1",
                    "prompt": PROMPT,
                    "source_review_status": row.get("review_status", "pending"),
                    "source_eligibility_status": row.get("eligibility_status", "not_reviewed"),
                    "status": "pending_generation",
                    "human_review_status": "not_reviewed",
                    "model_id": "",
                    "generation_parameters": {},
                }
                dst.write(json.dumps(job, ensure_ascii=False) + "\n")
                count_jobs += 1
            if a.limit_sources and count_sources >= a.limit_sources:
                break
    print(json.dumps({
        "source_rows_queued": count_sources,
        "draft_jobs": count_jobs,
        "target_levels": targets,
        "include_same_level": a.include_same_level,
        "output": str(out),
        "status": "queue only; no generation or validation performed",
        "warning": "Do not train on generated candidates before source review, semantic review, Bloom alignment review, deduplication, and grouped splitting."
    }, indent=2))

if __name__ == "__main__":
    main()
