"""Task-specific ChatML prompts for multi-task Qwen SFT.

Bloom contract is two-input only: question + target_level → rewrite.
Source Bloom level must never appear in generator prompts.
"""
from __future__ import annotations

from typing import Any
import hashlib
import re

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

FORBIDDEN_SOURCE_LEVEL_MARKERS = (
    "Original Bloom level:",
    "Source Bloom level:",
    "source bloom level",
    "original bloom level:",
)

BLOOM_SYSTEM = (
    "You are an expert academic assessment editor. "
    "This is the BLOOM_REWRITE task only: rewrite one supplied exam question for the caller's exact target Bloom level. "
    "Never answer it, summarize it, classify it, or switch to QA or summarization. "
    "Preserve the source question's subject, entities, core meaning, constraints, and relevant context. "
    "Never refer to stated/provided conditions, supplied data, inputs, measurements, constraints, or a described scenario unless the source actually provides them. "
    "Change the cognitive operation the student must perform, not just the wording or verb. "
    "Use the target-level action-verb guidance supplied in the user message: choose a suitable verb from that level's bank, "
    "prefer an action-verb-led imperative when it reads naturally, and make the requested thinking operation explicit. "
    "The verb is a cue, not proof of Bloom alignment: the whole task must require the target-level cognition. "
    "Do not add unsupported facts, arbitrary scenarios, new entities, or unrelated requirements. "
    "Do not force an awkward verb; choose another suitable verb from the same target-level bank if needed. "
    "OUTPUT ONLY ONE student-facing exam question as a single sentence or short paragraph. "
    "Do not include a Bloom label, rationale, answer, solution, preamble, bullet list, or meta commentary."
)

# Inference-time verb selection keeps greedy decoding from defaulting to a single verb.
# A semantic cue narrows the candidate pool; a stable hash rotates the pool per question.
_ACTION_VERB_BANKS = {
    "Remember": ["Identify", "List", "State", "Name", "Define", "Recall", "Recognize", "Specify"],
    "Understand": ["Explain", "Describe", "Interpret", "Summarize", "Classify", "Illustrate", "Clarify", "Paraphrase"],
    "Apply": ["Determine", "Calculate", "Solve", "Use", "Apply", "Compute", "Execute", "Estimate", "Predict", "Implement", "Test", "Diagnose", "Demonstrate", "Operate", "Perform"],
    "Analyze": ["Examine", "Compare", "Trace", "Differentiate", "Investigate", "Diagnose", "Contrast", "Categorize", "Analyze", "Distinguish", "Map", "Identify"],
    "Evaluate": ["Justify", "Critique", "Recommend", "Validate", "Rank", "Appraise", "Defend", "Judge", "Assess", "Evaluate"],
    "Create": ["Design", "Develop", "Formulate", "Construct", "Propose", "Devise", "Compose", "Plan", "Synthesize", "Generate"],
}


def select_action_verbs(question: str, target_level: str) -> list[str]:
    """Return a stable ordering of plausible action verbs for this question and level."""
    level = target_level.strip().title()
    if level not in _ACTION_VERB_BANKS:
        raise ValueError(f"Unknown target Bloom level: {target_level!r}")
    q = question.lower()
    if level == "Apply":
        if re.search(r"\b(calculat|comput|numeric|equation|formula|percentage|probability|sum|average|distance|speed|cost|rate)\w*\b", q):
            pool = ["Calculate", "Compute", "Determine", "Estimate", "Solve", "Apply"]
        elif re.search(r"\b(error|debug|fault|failure|bug|diagnos)\w*\b", q):
            pool = ["Diagnose", "Test", "Use", "Apply", "Determine", "Solve"]
        elif re.search(r"\b(given|input|output|scenario|case|example|algorithm|procedure|method)\b", q):
            pool = ["Use", "Apply", "Execute", "Determine", "Solve", "Implement", "Demonstrate"]
        else:
            # Broad concept questions do not supply inputs or conditions for prediction/calculation.
            # Prefer verbs that can introduce a grounded, simple application without inventing data.
            pool = ["Demonstrate", "Use", "Apply", "Implement", "Execute", "Operate", "Perform", "Determine"]
    elif level == "Analyze":
        if re.search(r"\b(compare|versus|difference|differentiate|contrast|similarit)\w*\b", q):
            pool = ["Compare", "Differentiate", "Contrast", "Examine", "Distinguish", "Analyze"]
        elif re.search(r"\b(step|sequence|process|flow|trace|execution|order)\w*\b", q):
            pool = ["Trace", "Examine", "Investigate", "Analyze", "Map", "Identify"]
        elif re.search(r"\b(error|fault|failure|bug|problem|cause|why)\w*\b", q):
            pool = ["Investigate", "Diagnose", "Examine", "Trace", "Analyze", "Identify"]
        else:
            pool = ["Examine", "Investigate", "Analyze", "Distinguish", "Categorize", "Trace", "Compare", "Differentiate"]
    elif level == "Evaluate":
        if re.search(r"\b(recommend|choose|select|alternative|option|best|which)\w*\b", q):
            pool = ["Recommend", "Rank", "Justify", "Appraise", "Evaluate", "Assess"]
        elif re.search(r"\b(valid|correct|accurate|prove|evidence|claim|argument|justify)\w*\b", q):
            pool = ["Validate", "Justify", "Defend", "Critique", "Judge", "Assess"]
        else:
            pool = ["Justify", "Critique", "Appraise", "Defend", "Judge", "Evaluate", "Recommend", "Rank", "Validate"]
    else:
        pool = list(_ACTION_VERB_BANKS[level])
    digest = hashlib.sha256(f"{level}\0{question.strip().casefold()}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % len(pool)
    return pool[offset:] + pool[:offset]


# These are inference-time and SFT prompt instructions for the requested target level.
# Verbs are examples, not a keyword classifier; validate the cognitive operation in context.
BLOOM_TARGET_GUIDANCE = {
    "Remember": (
        "Cognitive operation: retrieve or recognize a fact, term, component, rule, or procedure. "
        "Preferred action verbs (choose the most natural and semantically accurate option): Identify, Define, List, Name, State, Recall, Recognize, Label, Select, Match, Enumerate, Retrieve, Locate, Reproduce, Repeat, Record, Quote, Specify, Indicate, Tabulate, Outline, Cite, Recite, Report, Identify by name, Recall from memory. "
        "Keep the task to retrieval/recognition; do not ask for explanation, comparison, application, judgment, or design. "
        "Example form: 'Identify the main components of [original topic].'"
    ),
    "Understand": (
        "Cognitive operation: explain meaning, interpret, classify, summarize, or describe an existing idea. "
        "Preferred action verbs (choose the most natural and semantically accurate option): Explain, Describe, Summarize, Interpret, Classify, Illustrate, Paraphrase, Discuss, Exemplify, Clarify, Restate, Represent, Translate, Characterize, Report, Associate, Group, Sort, Convert, Infer, Relate meaning, Generalize, Review, Explain in your own words. "
        "Require comprehension of the original concept, not a solution to a new case, a criteria-based judgment, or a new artifact. "
        "Example form: 'Explain how [original concept] works.'"
    ),
    "Apply": (
        "Cognitive operation: use a learned rule, method, or concept to work through a concrete case. "
        "Preferred action verbs (choose the most natural and semantically accurate option): Apply, Use, Calculate, Solve, Demonstrate, Execute, Implement, Determine, Compute, Predict, Operate, Perform, Practice, Prepare, Estimate, Measure, Test, Run, Manipulate, Model, Modify, Complete, Find, Obtain, Carry out, Execute, Illustrate through use, Solve using, Verify by applying a stated rule. "
        "Keep the original topic and require an actual use or procedure, not just explanation. If the source gives no inputs or conditions, use only a simple generic application grounded in the concept; do not invent specific data or imply that conditions were supplied. "
        "Example form: 'Given [relevant conditions], apply [original method] to determine [relevant result].'"
    ),
    "Analyze": (
        "Cognitive operation: break information into parts and examine relationships, structure, causes, patterns, or differences. "
        "Preferred action verbs (choose the most natural and semantically accurate option): Analyze, Compare, Contrast, Differentiate, Examine, Investigate, Categorize, Decompose, Trace, Relate, Organize, Attribute, Structure, Inspect, Probe, Correlate, Separate, Subdivide, Infer, Detect, Uncover, Diagnose, Dissect, Scrutinize, Distinguish, Map, Identify assumptions, Identify patterns, Determine relationships, Examine dependencies. "
        "Make the relationship, parts, cause, or pattern to be examined explicit; do not stop at listing facts or merely asking for an unsupported opinion. "
        "Example form: 'Analyze how [original topic's parts or factors] interact.'"
    ),
    "Evaluate": (
        "Cognitive operation: make and support a judgment using explicit criteria, evidence, standards, or trade-offs. "
        "Preferred action verbs (choose the most natural and semantically accurate option): Evaluate, Assess, Justify, Critique, Appraise, Defend, Judge, Recommend, Validate, Prioritize, Rate, Rank, Argue, Weigh, Verify, Review, Support, Refute, Decide, Substantiate, Endorse, Reject, Audit, Challenge, Benchmark, Critically review, Determine merit, Establish validity, Select with justification, Resolve using criteria. "
        "Require a reasoned judgment tied to criteria/evidence; do not only describe/analyze, and do not turn the task into designing a new artifact. "
        "Example form: 'Evaluate [original approach] against [relevant criteria] and justify the judgment.'"
    ),
    "Create": (
        "Cognitive operation: produce or propose a new artifact, plan, model, design, procedure, or solution by combining ideas. "
        "Preferred action verbs (choose the most natural and semantically accurate option): Design, Develop, Create, Construct, Formulate, Propose, Devise, Compose, Generate, Plan, Invent, Build, Produce, Assemble, Synthesize, Integrate, Engineer, Draft, Author, Architect, Establish, Originate, Adapt, Elaborate, Hypothesize, Combine, Innovate, Model, Write, Invent alternatives, Develop a framework, Design an experiment. "
        "Require a new deliverable and preserve the source topic; add constraints only when grounded in the source or necessary to make the deliverable clear. "
        "Do not merely explain, analyze, or judge an existing item. "
        "Example form: 'Design [a relevant new deliverable] for [original purpose/context].'"
    ),
}

QA_SYSTEM = (
    "You are an academic question-answering assistant. "
    "Answer the question using the supplied context. "
    "Do not invent information."
)

SUMMARIZATION_SYSTEM = (
    "You are an academic scientific summarization assistant. "
    "Summarize the provided text while preserving important factual content. "
    "Do not invent information."
)


def assert_no_source_level_in_prompt(text: str) -> None:
    lowered = text.lower()
    for marker in FORBIDDEN_SOURCE_LEVEL_MARKERS:
        if marker.lower() in lowered:
            raise AssertionError(
                f"Generator prompt must not contain source Bloom level marker {marker!r}"
            )


def render_chatml(messages: list[dict[str, str]], *, add_generation_prompt: bool = False) -> str:
    parts: list[str] = []
    for message in messages:
        parts.append(f"{IM_START}{message['role']}\n{message['content']}{IM_END}")
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "\n".join(parts)


def bloom_messages(
    question: str,
    target_level: str,
    target_rewrite: str | None = None,
    selected_action_verb: str | None = None,
) -> list[dict[str, str]]:
    canonical_target = target_level.strip().title()
    target_guidance = BLOOM_TARGET_GUIDANCE.get(canonical_target)
    if target_guidance is None:
        raise ValueError(f"Unknown target Bloom level: {target_level!r}")
    user = (
        f"Original question:\n{question.strip()}\n\n"
        f"Target Bloom level:\n{canonical_target} (fixed; do not substitute another level)\n\n"
        f"Target-level action verbs and cognitive-operation instructions:\n{target_guidance}\n\n"
        "Rewrite requirements:\n"
        + (
            f"- Begin with this required action verb: {selected_action_verb}. "
            "Use this exact opening verb (capitalization may vary) to express the actual target-level task.\n"
            if selected_action_verb
            else "- Start with a suitable action verb from the requested level when grammatically natural.\n"
        )
        + "- Use the verb to express the actual student task, not as a superficial keyword.\n"
        "- Preserve the original topic, entities, meaning, and relevant context.\n"
        "- Do not invent conditions, data, inputs, measurements, constraints, or a scenario. Do not say 'under the stated conditions' unless the source states conditions.\n"
        "- For broad concept questions, use a simple, relevant application without pretending the source supplied specific facts or constraints.\n"
        "- Return one valid exam imperative or interrogative only; no answer or explanation."
    )
    messages = [
        {"role": "system", "content": BLOOM_SYSTEM},
        {"role": "user", "content": user},
    ]
    if target_rewrite is not None:
        messages.append({"role": "assistant", "content": target_rewrite.strip()})
    return messages


def qa_messages(
    context: str,
    question: str,
    answer: str | None = None,
) -> list[dict[str, str]]:
    user = f"Context:\n{context.strip()}\n\nQuestion:\n{question.strip()}"
    messages = [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": user},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages


def summarization_messages(
    article: str,
    abstract: str | None = None,
) -> list[dict[str, str]]:
    user = f"Text:\n{article.strip()}"
    messages = [
        {"role": "system", "content": SUMMARIZATION_SYSTEM},
        {"role": "user", "content": user},
    ]
    if abstract is not None:
        messages.append({"role": "assistant", "content": abstract.strip()})
    return messages


def build_sft_text(task: str, record: dict[str, Any]) -> str:
    task = task.strip().lower()
    if task == "bloom_rewrite":
        text = render_chatml(
            bloom_messages(
                record["source_question"],
                record["target_bloom_level"],
                record["target_rewrite"],
            )
        )
        assert_no_source_level_in_prompt(text)
        return text
    if task == "qa":
        return render_chatml(
            qa_messages(record["context"], record["question"], record["answer"])
        )
    if task == "summarization":
        return render_chatml(
            summarization_messages(record["article"], record["abstract"])
        )
    raise ValueError(f"Unknown task: {task}")


def build_generation_prompt(task: str, record: dict[str, Any]) -> str:
    task = task.strip().lower()
    if task == "bloom_rewrite":
        text = render_chatml(
            bloom_messages(
                record["source_question"],
                record["target_bloom_level"],
                selected_action_verb=record.get("selected_action_verb"),
            ),
            add_generation_prompt=True,
        )
        assert_no_source_level_in_prompt(text)
        return text
    if task == "qa":
        return render_chatml(
            qa_messages(record["context"], record["question"]),
            add_generation_prompt=True,
        )
    if task == "summarization":
        return render_chatml(
            summarization_messages(record["article"]),
            add_generation_prompt=True,
        )
    raise ValueError(f"Unknown task: {task}")


def build_prompt_only_text(task: str, record: dict[str, Any]) -> str:
    """System+user ChatML without assistant content (for loss masking)."""
    task = task.strip().lower()
    if task == "bloom_rewrite":
        text = render_chatml(
            bloom_messages(record["source_question"], record["target_bloom_level"])
        )
        assert_no_source_level_in_prompt(text)
        return text
    if task == "qa":
        return render_chatml(qa_messages(record["context"], record["question"]))
    if task == "summarization":
        return render_chatml(summarization_messages(record["article"]))
    raise ValueError(f"Unknown task: {task}")
