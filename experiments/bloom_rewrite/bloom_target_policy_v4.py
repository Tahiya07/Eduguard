"""Strict validation policy for teacher-generated Bloom rewrites (v4)."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
POLICY_VERSION = "bloom_target_policy_v4_teacher_validated"
STOP = set("a an the and or but if then than that this these those it its of in on at to for from with by as is are was were be been being do does did can could may might will would should must into over under after before during through about against between among within without using used".split())
VERBS = set("define explain describe list name state identify recall recognize summarize interpret classify illustrate apply use calculate compute determine solve implement demonstrate analyze analyse compare contrast differentiate examine evaluate assess critique justify judge defend design develop construct formulate propose create devise produce generate".split())
MARKERS = {
"Remember":("define","identify","name","list","state","recall","recognize","what is","what are"),
"Understand":("explain","describe","summarize","interpret","classify","illustrate","why does","why is","how does","how do","meaning of","purpose of"),
"Apply":("apply","use","calculate","compute","solve","determine","implement","demonstrate","scenario","case","problem","procedure","given a"),
"Analyze":("analyze","analyse","compare","contrast","differentiate","examine","break down","components","parts","relationship","causes","patterns","structure"),
"Evaluate":("evaluate","assess","critique","judge","justify","defend","effectiveness","validity","suitability","criteria","evidence","strengths","limitations","trade-off"),
"Create":("design","develop","construct","formulate","propose","create","devise","produce","plan","strategy","solution","procedure","artifact","model","framework","prototype"),}
META=("the rewritten question","rewritten question:","the answer is","correct answer","as an ai","bloom level","target level","original question:","this question asks","the student should","here is the question")
GENERIC=("compare the main components of","analyze how the parts of","examine the causes and patterns that structure","assess how well","how well does","stated academic criteria","create an original academic artifact","formulate a structured approach for constructing a new solution","develop an original procedure related to","given a concrete case involving","in the following problem about","analyze how the parts of a","design and analyze","design and evaluate","develop and evaluate","explain how a program can be structured")
FORBIDDEN_LEVEL_CUES={
"Remember":("steps involved","steps to","procedure for","how to","implement a program","write a program","modify a program"),
"Understand":("evaluate whether","judge whether","design a new","create a new solution"),
"Apply":(),
"Analyze":(),
"Evaluate":("design a new","create a new","construct a new","develop a new"),
"Create":(),
}
INVENTED_ARTIFACT_CUES=("provided code","provided program","provided data","provided passage","given code","given program","given data","given passage")
SCOPE_EXPANSION_CUES=("input validation","edge cases","error handling","resource utilization","memory utilization","time complexity","user-specified range","valid range")
COGNITIVE_SUPPORT_WORDS=set(("analyze analyse structure logic components parts relationship relationships interactions causes patterns effects criteria evidence effectiveness validity suitability quality strengths limitations trade-off design develop construct formulate propose create devise plan strategy solution procedure artifact model framework prototype".split()))

WORD_RE=re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")
NUMBER_RE=re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])")

@dataclass
class ValidationResult:
    ok: bool
    failure_category: str=""
    reasons:list[str]=field(default_factory=list)
    source_content_recall:float=0.0
    protected_span_recall:float=1.0
    lexical_similarity:float=0.0
    semantic_similarity:float|None=None
    target_signals:list[str]=field(default_factory=list)
    classifier_prediction:str|None=None
    classifier_confidence:float|None=None

def canonical_level(x:str)->str|None:
    a={"remembering":"Remember","knowledge":"Remember","recall":"Remember","understanding":"Understand","comprehension":"Understand","applying":"Apply","application":"Apply","analyzing":"Analyze","analysing":"Analyze","analysis":"Analyze","evaluating":"Evaluate","evaluation":"Evaluate","creating":"Create","synthesis":"Create"}
    x=str(x or "").strip()
    return x if x in BLOOM_LEVELS else a.get(x.lower())

def clean_output(x:str)->str:
    x=(x or "").replace("<|im_start|>assistant","").replace("<|im_end|>","").replace("<|endoftext|>","").strip()
    x=re.sub(r"<think>.*?</think>\s*","",x,flags=re.I|re.S)
    x=re.sub(r"^\s*(?:answer|response|rewrite|rewritten question|question|output)\s*:\s*","",x,flags=re.I)
    x=re.sub(r"\s+"," ",x).strip(" \t\"'")
    return x

def tokens(x:str): return [t.lower() for t in WORD_RE.findall(x or "")]
def content(x:str): return [t for t in tokens(x) if len(t)>2 and t not in STOP and t not in VERBS]
def protected(x:str):
    out=NUMBER_RE.findall(x or "")
    out += [t.lower() for t in tokens(x) if any(c.isdigit() for c in t) or any(c in t for c in "+#/-._")]
    return sorted(set(out))

def _signals(text,target):
    n=text.lower(); return [m for m in MARKERS[target] if m in n]

def _target_ok(text,target):
    n=text.lower(); s=_signals(text,target)
    if any(x in n for x in FORBIDDEN_LEVEL_CUES.get(target,())): return False
    if target=="Evaluate":
        return bool(
            any(x in n for x in ("evaluate","assess","critique","judge","justify","defend"))
            and any(x in n for x in ("criteria","evidence","effectiveness","validity","suitability","quality","strengths","limitations","trade-off"))
        )
    if target=="Create":
        return bool(
            any(x in n for x in ("design","develop","construct","formulate","propose","create","devise"))
            and any(x in n for x in ("plan","strategy","solution","procedure","artifact","model","framework","prototype","program"))
        )
    return bool(s)

def validate_candidate(source_question,target_level,candidate,*,semantic_similarity=None,min_semantic_similarity=.55,classifier_prediction=None,classifier_confidence=None,min_classifier_confidence=.60):
    target=canonical_level(target_level); text=clean_output(candidate); r=[]
    if not target: return ValidationResult(False,"INVALID_TARGET_LEVEL",["invalid_target_level"])
    if not text: return ValidationResult(False,"EMPTY_OUTPUT",["empty_output"])
    if len(text)>600: r.append("too_long")
    if text.count("?")>1: r.append("multiple_questions")
    n=text.lower()
    if any(p in n for p in META): r.append("meta_or_answer_language")
    if "?" not in text and not n.startswith(("define ","identify ","name ","list ","state ","describe ","explain ","summarize ","analyze ","analyse ","compare ","contrast ","examine ","evaluate ","assess ","critique ","justify ","design ","develop ","construct ","formulate ","propose ","create ","apply ","use ")): r.append("invalid_exam_question_form")
    generic=next((p for p in GENERIC if p in n),None)
    if generic: r.append("generic_template:"+generic)

    source_lower=source_question.lower()
    # Do not refer to an artifact as though it was supplied when the source
    # question does not actually contain one.
    artifact_terms=("code","program","data","passage","table","diagram","graph","dataset","document","solution")
    source_has_artifact=any(x in source_lower for x in artifact_terms)
    if target=="Evaluate" and not source_has_artifact and any(x in n for x in INVENTED_ARTIFACT_CUES):
        r.append("invented_artifact_reference")

    # Guard common forms of scope expansion that add requirements absent from
    # the source. These are especially risky for Create/Evaluate rewrites.
    if target in ("Create","Evaluate") and any(x in n for x in SCOPE_EXPANSION_CUES):
        source_scope = set(content(source_question))
        extra_cues=[]
        for cue in SCOPE_EXPANSION_CUES:
            if cue in n:
                cue_terms=[t for t in tokens(cue) if t not in STOP and t not in VERBS]
                if cue_terms and not all(t in source_scope for t in cue_terms):
                    extra_cues.append(cue)
        if extra_cues:
            r.append("scope_expansion:"+extra_cues[0])

    s_norm=re.sub(r"\s+"," ",source_question.lower()).strip(" ?."); c_norm=re.sub(r"\s+"," ",text.lower()).strip(" ?.")
    lex=SequenceMatcher(None,s_norm,c_norm).ratio()
    if s_norm==c_norm: r.append("exact_source_copy")
    if lex>=.93: r.append("near_source_copy")
    st=set(content(source_question)); ct=set(content(text)); recall=len(st&ct)/len(st) if st else 1.0
    if recall<.55: r.append("low_source_content_recall")
    ps=protected(source_question); hits=sum(x in n for x in ps); prec=hits/len(ps) if ps else 1.0
    if prec<.90: r.append("protected_span_loss")
    if not _target_ok(text,target): r.append("target_structure_mismatch")
    if semantic_similarity is not None and semantic_similarity<min_semantic_similarity: r.append("low_semantic_similarity")
    if classifier_prediction is not None and canonical_level(classifier_prediction)!=target: r.append("classifier_target_mismatch")
    if classifier_prediction is not None and classifier_confidence is not None and classifier_confidence<min_classifier_confidence: r.append("classifier_low_confidence")
    if not r: cat=""
    else:
        cat=next(
            (x.upper() for x in (
                "empty_output","meta_or_answer_language","generic_template",
                "invented_artifact_reference","scope_expansion",
                "protected_span_loss","low_source_content_recall",
                "low_semantic_similarity","classifier_target_mismatch",
                "target_structure_mismatch","near_source_copy"
            ) if any(y.startswith(x) for y in r)),
            "QUALITY_REJECTION"
        )
    return ValidationResult(not r,cat,r,round(recall,4),round(prec,4),round(lex,4),semantic_similarity,_signals(text,target),canonical_level(classifier_prediction) if classifier_prediction else None,classifier_confidence)

def build_teacher_messages(source_question,target_level,retry=False):
    target=canonical_level(target_level) or target_level
    guidance={
        "Remember":"Recall facts, definitions, terminology, syntax, or basic information. Do not require implementation, explanation, analysis, judgment, or design.",
        "Understand":"Explain, describe, summarize, interpret, or classify a concept. Do not require implementation or judgment.",
        "Apply":"Use a known rule, concept, method, or procedure to solve or carry out a concrete task.",
        "Analyze":"Break the problem or program into parts and examine relationships, interactions, causes, structure, patterns, or effects.",
        "Evaluate":"Make a judgment about correctness, quality, efficiency, validity, effectiveness, or suitability using explicit or implied criteria. Do not design a new solution.",
        "Create":"Design, construct, formulate, develop, or propose a new solution, program, procedure, model, plan, or artifact.",
    }[target]
    retry_note = (
        "\nREPAIR MODE: The previous candidate failed validation. Generate substantially different wording that fixes the quality problem while preserving the same topic and target level."
        if retry else ""
    )
    system=(
        "You are a senior university assessment editor. "
        "Rewrite the complete student-facing academic exam question so its PRIMARY cognitive demand matches the requested Revised Bloom level.\n\n"
        "Target-level requirement:\n" + guidance +
        "\n\nStrict rules:\n"
        "- Preserve the original topic, technical entities, quantities, constraints, and academic intent.\n"
        "- Change the student's task, not the subject matter.\n"
        "- Do not merely replace a verb.\n"
        "- Do not introduce unrelated content or any new requirements.\n"
        "- Never invent an artifact, code, data, passage, diagram, or other material that the source question does not provide.\n"
        "- Do not add validation, edge cases, implementation features, performance criteria, or other scope unless the source already requires them or they are strictly necessary to express the target cognitive process.\n"
        "- For Evaluate, evaluate the stated task/concept when no concrete artifact is supplied; never pretend that code, data, or another artifact was provided.\n"
        "- For Create, preserve the original constraints and topic while changing the task into a new construction/design task; do not add unrelated features.\n"
        "- Do not answer, explain, or discuss the rewrite.\n"
        "- Do not mention Bloom, the target level, the source question, or these instructions.\n"
        "- Output exactly ONE exam question or valid exam imperative.\n"
        "- The requested level must be the dominant cognitive operation; avoid mixing multiple Bloom levels.\n"
        "- Do not use generic stock phrases.\n"
        "- Be concise and specific." + retry_note
    )
    user=(
        f"Original question:\n{source_question.strip()}\n\n"
        f"Target Bloom level:\n{target}\n\n"
        "Return only the rewritten exam question.\n\n/no_think"
    )
    return [{"role":"system","content":system},{"role":"user","content":user}]
