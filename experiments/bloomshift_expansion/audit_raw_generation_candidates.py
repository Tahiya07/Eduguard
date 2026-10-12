#!/usr/bin/env python3
"""Structural audit for raw BloomShift generation JSONL; never certifies semantics."""
import argparse, collections, json, re
from pathlib import Path

LEVELS = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}
VERBS = {
"Remember": {"identify","define","list","state","recall","name","label","match","select","locate","recognize","enumerate","quote","reproduce","sequence","count","tabulate","retrieve"},
"Understand": {"explain","summarize","interpret","classify","paraphrase","illustrate","describe","distinguish","compare","infer","translate","exemplify","outline","restate","categorize","predict","clarify","relate"},
"Apply": {"apply","calculate","demonstrate","implement","use","solve","execute","compute","operate","modify","illustrate","prepare","construct","complete","determine","perform","adapt","show"},
"Analyze": {"analyze","differentiate","examine","decompose","trace","investigate","categorize","contrast","dissect","separate","attribute","organize","map","relate","diagnose","inspect","compare","distinguish"},
"Evaluate": {"evaluate","justify","critique","assess","defend","validate","appraise","rank","recommend","judge","verify","prioritize","argue","rate","support","select","challenge","weigh"},
"Create": {"design","construct","develop","formulate","propose","compose","devise","generate","invent","plan","synthesize","engineer","produce","assemble","draft","build","create","model"},
}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--report", required=True)
    a=p.parse_args()
    counts=collections.Counter(); issues=[]; seen=set(); pairs=collections.Counter()
    with open(a.input,encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            try: r=json.loads(line)
            except json.JSONDecodeError as e:
                issues.append({"line":line_no,"issue":"invalid_json","detail":str(e)}); continue
            sid=r.get("source_id",""); level=r.get("target_bloom_level","")
            candidate=r.get("parsed_candidate")
            counts["records"]+=1
            if not sid: issues.append({"line":line_no,"issue":"missing_source_id"})
            if level not in LEVELS: issues.append({"line":line_no,"issue":"invalid_target_level","value":level})
            key=(sid,level)
            if key in seen: issues.append({"line":line_no,"issue":"duplicate_source_target_pair","value":list(key)})
            seen.add(key); pairs[key]+=1
            if not isinstance(candidate,dict):
                counts["unparsed"]+=1; issues.append({"line":line_no,"issue":"candidate_not_parsed_json"}); continue
            feasible=candidate.get("feasibility")
            if feasible not in {"feasible","not_feasible"}:
                issues.append({"line":line_no,"issue":"invalid_feasibility_value","value":feasible})
            text=(candidate.get("target_rewrite") or "").strip()
            verb=(candidate.get("opening_verb") or "").strip().casefold()
            if feasible=="feasible":
                counts["feasible_candidates"]+=1
                if not text: issues.append({"line":line_no,"issue":"empty_rewrite_marked_feasible"})
                first=re.match(r"^[\W_]*([A-Za-z]+)",text)
                first_word=first.group(1).casefold() if first else ""
                if first_word not in VERBS.get(level,set()):
                    issues.append({"line":line_no,"issue":"opening_word_not_in_target_verb_bank","value":first_word,"target":level})
                if verb and first_word != verb:
                    issues.append({"line":line_no,"issue":"opening_verb_field_mismatch","value":{"text_first_word":first_word,"field":verb}})
                if not candidate.get("cognitive_operation"):
                    issues.append({"line":line_no,"issue":"missing_cognitive_operation"})
                if not candidate.get("preserved_elements"):
                    issues.append({"line":line_no,"issue":"missing_preserved_elements"})
            elif feasible=="not_feasible":
                counts["not_feasible"]+=1
            if r.get("human_review_status") not in {"approved","rejected","needs_revision","not_reviewed"}:
                issues.append({"line":line_no,"issue":"unexpected_human_review_status","value":r.get("human_review_status")})
    report={"input":a.input,"counts":dict(counts),"unique_source_target_pairs":len(seen),"issue_count":len(issues),"issues":issues,
            "interpretation":"Structural screening only. Passing does not establish semantic preservation, context preservation, pedagogical validity, or true Bloom-level alignment."}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="issues"},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
