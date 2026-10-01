import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
INFILE=ROOT/"data/rewrite dataset/train.json"
OUT_JSON=ROOT/"data/rewrite dataset/train_balanced_v3.json"
OUT_JSONL=ROOT/"data/rewrite dataset/train_balanced_v3.jsonl"
LEVELS=["Remember","Understand","Apply","Analyze","Evaluate","Create"]
OPS={
"Remember":"recall_key_facts_terms_or_steps",
"Understand":"explain_meaning_or_process",
"Apply":"use_knowledge_in_a_new_equivalent_situation",
"Analyze":"break_down_parts_and_relationships",
"Evaluate":"judge_against_explicit_criteria",
"Create":"produce_an_original_solution_or_design",
}
T={
"Remember":[
"What key facts, terms, or steps must be recalled to answer the following question? {q}",
"State the essential facts, terminology, or procedure needed to answer this question: {q}",
"Identify the knowledge that must be remembered in order to address the following question: {q}",
"List the key facts, concepts, or steps required for the following task: {q}",
],
"Understand":[
"Explain the concept, procedure, or relationship involved in the following question in your own words: {q}",
"Describe what the following question is asking and explain the underlying concept or process: {q}",
"Interpret the main idea or procedure required by the following question and explain it clearly: {q}",
"Summarize what must be understood to answer the following question, using your own words: {q}",
],
"Apply":[
"Apply the relevant concept or procedure from the following question to a new but equivalent situation. Show the steps you would use: {q}",
"Use the knowledge required by the following question to solve a new instance of the same type of problem. Explain your procedure: {q}",
"Demonstrate how the method or concept in the following question would be used when the situation changes while the underlying task remains the same: {q}",
"Work through a new, equivalent case using the concept or procedure required by the following question. Show how you apply it: {q}",
],
"Analyze":[
"Break the following question into its main components, identify how they are related, and explain how those relationships determine the required response: {q}",
"Analyze the parts of the following task and explain the relationships among them before determining what the answer must address: {q}",
"Differentiate the major elements of the following question and examine how those elements interact: {q}",
"Deconstruct the following question into its essential parts and determine how each part contributes to the overall task: {q}",
],
"Evaluate":[
"Assess possible approaches to answering the following question using appropriate subject-specific criteria, and justify your judgment: {q}",
"Evaluate the adequacy of an answer to the following task by applying relevant criteria and supporting your judgment with reasons: {q}",
"Judge the quality of a proposed solution to the following question using clear, relevant criteria. Explain your judgment: {q}",
"Critique an approach to the following task against appropriate standards and justify which aspects are satisfactory or need improvement: {q}",
],
"Create":[
"Develop an original solution or response to the following task. Define its main steps or structure and explain how they address the task: {q}",
"Design a new approach for completing the following task, specifying the components and sequence of your proposed solution: {q}",
"Construct an original response to the following question that brings the relevant elements together into a coherent solution: {q}",
"Formulate an original method for addressing the following question. State the main elements of the method and how they fit together: {q}",
],
}

def pick(a,key):
    n=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    return a[n%len(a)]

def prompt(q,l):
    return ("<|im_start|>system\nRewrite the academic question to the requested Bloom level. "
            "Preserve the source topic, technical concepts, named entities, quantities, and constraints. "
            "Change the cognitive operation without inventing new subject matter. The requested level "
            "must be expressed through the task the student performs, not merely by inserting a Bloom verb. "
            "Output only one student-facing exam question.<|im_end|>\n<|im_start|>user\n"
            f"Original question:\n{q}\n\nTarget Bloom level:\n{l}<|im_end|>")

def eid(s,l,q):
    return hashlib.sha256(f"{s}|{l}|{q}".encode()).hexdigest()[:16]

data=json.loads(INFILE.read_text(encoding="utf-8"))
by={}
for r in data: by.setdefault(r["source_id"],r)
out=[]
for src in by.values():
    q=" ".join(src["source_question"].split())
    for l in LEVELS:
        same=src["source_bloom_level"]==l
        rw=q if same else pick(T[l],src["source_id"]+"|"+l).format(q=q)
        p=prompt(q,l)
        out.append({
            **src,"example_id":eid(src["source_id"],l,rw),
            "target_bloom_level":l,"target_rewrite":rw,
            "transformation_type":f'{src["source_bloom_level"]}->{l}',
            "synthetic_or_original":"original_identity" if same else "controlled_synthetic_v3",
            "synthetic":not same,"dataset_version":"figshare_target_rewrite_balanced_v3",
            "policy_version":"figshare_bloom_target_policy_v3",
            "quality_status":"rule_validated_v3",
            "construction_method":"source_anchored_cognitive_transformation_v3",
            "construction_template":"identity" if same else f"source_anchored_{l.lower()}_v3",
            "target_cognitive_operation":OPS[l],"prompt_text":p,
            "sft_text":p+f"\n<|im_start|>assistant\n{rw}<|im_end|>",
            "source_context_status":src.get("source_context_status","none")
        })
counts={l:sum(r["target_bloom_level"]==l for r in out) for l in LEVELS}
assert len(out)==len(by)*6 and len(set(r["example_id"] for r in out))==len(out)
assert all(v==len(by) for v in counts.values())
OUT_JSON.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
OUT_JSONL.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in out)+"\n",encoding="utf-8")
print(len(out),counts)
