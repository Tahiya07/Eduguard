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
    if target=="Evaluate": return bool(any(x in n for x in ("evaluate","assess","critique","judge","justify","defend")) and any(x in n for x in ("criteria","evidence","effectiveness","validity","suitability","quality","strengths","limitations","trade-off")))
    if target=="Create": return bool(any(x in n for x in ("design","develop","construct","formulate","propose","create","devise")) and any(x in n for x in ("plan","strategy","solution","procedure","artifact","model","framework","prototype")))
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
        cat=next((x.upper() for x in ("empty_output","meta_or_answer_language","generic_template","protected_span_loss","low_source_content_recall","low_semantic_similarity","classifier_target_mismatch","target_structure_mismatch","near_source_copy") if any(y.startswith(x) for y in r)),"QUALITY_REJECTION")
    return ValidationResult(not r,cat,r,round(recall,4),round(prec,4),round(lex,4),semantic_similarity,_signals(text,target),canonical_level(classifier_prediction) if classifier_prediction else None,classifier_confidence)

def build_teacher_messages(source_question,target_level,retry=False):
    retry_note=" Retry: substantially change the wording and do not use a stock template." if retry else ""
    system=("You are a senior university assessment editor. Rewrite the complete student-facing academic exam question so its required cognitive operation matches the requested Revised Bloom level. Preserve the original topic, technical entities, quantities, constraints, and academic intent. Change the student's task, not the subject matter. Output exactly one exam question or valid exam imperative. Do not answer it, explain it, mention Bloom, mention the source question, or use a generic template. Do not introduce unrelated content."+retry_note)
    guidance={"Remember":"recall, identify, name, list, define, or state facts without analysis or judgment.","Understand":"explain, describe, interpret, summarize, or classify existing material.","Apply":"use knowledge or a procedure in a concrete case, scenario, calculation, or problem.","Analyze":"examine components, relationships, causes, patterns, comparisons, or structure.","Evaluate":"make a justified judgment about quality, effectiveness, validity, or suitability using criteria or evidence.","Create":"design, develop, formulate, construct, or propose a new solution, plan, procedure, model, or artifact grounded in the source topic."}[canonical_level(target_level) or target_level]
    user=f"Original question:\n{source_question.strip()}\n\nTarget Bloom level:\n{target_level}\n\nRequired cognitive task:\n{guidance}\n\nReturn only the rewritten exam question."
    return [{"role":"system","content":system},{"role":"user","content":user}]
