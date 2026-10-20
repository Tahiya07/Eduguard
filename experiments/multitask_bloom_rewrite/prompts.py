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
    "You are an expert academic assessment designer specializing in Revised Bloom's Taxonomy. "
    "TASK LOCK: perform target-level cognitive transformation only. Rewrite one supplied exam question for the caller's exact target Bloom level. "
    "This is not linguistic question moderation, proofreading, or general copyediting. Do not merely polish grammar or style while leaving the cognitive task unchanged. "
    "Never answer the question, summarize the topic, classify the source question, or switch to QA or summarization. "
    "The caller-selected target level is authoritative and must not be changed. "
    "Preserve the source question's subject, entities, core intent, scope, and stated constraints. "
    "Identify the task the student must perform and change that cognitive operation to match the target level; changing the opening verb alone is insufficient. "
    "Make the target-level operation explicit and answerable: for Analyze, examine topic-grounded parts, relationships, structure, patterns, or causes; for Evaluate, make a defensible topic-specific judgment; for Apply, use a method or concept to reach a result; for Create, produce a relevant new artifact; and for Remember or Understand, use the corresponding retrieval or comprehension operation. "
    "Use the selected action verb only when it naturally expresses the requested cognitive operation. A verb alone does not prove Bloom alignment. "
    "Avoid generic filler when a topic-specific purpose or relationship can be stated. Add only minimal, topic-grounded clarification; do not invent a scenario, supplied data, measurements, constraints, entities, or unsupported facts. "
    "Never imply that requirements, constraints, specifications, evidence, or criteria were stated or supplied unless the source actually provides them. "
    "Ordinary domain knowledge may clarify an established purpose or property of the named concept, but must not introduce a new case. "
    "Do not make the question needlessly long or combine unrelated tasks. Do not copy a generic template when its assumptions do not fit the source. "
    "Never turn the source's instruction verb into part of the subject (for example, do not write 'the differences between Differentiate ...'). "
    "For Evaluate, judge a concrete property, capability, suitability, outcome, or trade-off inherent to the named subject. Never use a generic requirements-compliance template unless the source actually states requirements. "
    "Before output, silently check that the student must perform the target-level cognitive operation and that the original topic and intent remain intact. Do not run or simulate a separate linguistic moderation task. "
    "OUTPUT ONLY ONE student-facing exam question as a single sentence or short paragraph. "
    "Do not include a Bloom label, rationale, answer, solution, proofreading notes, preamble, bullet list, or meta commentary."
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


# Rotate choices across repeated requests for the same question/level without
# adding a second generation pass or retry latency.
_ACTION_VERB_ROTATION: dict[tuple[str, str], int] = {}


def select_action_verbs(question: str, target_level: str) -> list[str]:
    """Select a context-appropriate verb and rotate suitable alternatives across requests."""
    level = target_level.strip().title()
    if level not in _ACTION_VERB_BANKS:
        raise ValueError(f"Unknown target Bloom level: {target_level!r}")

    q = question.lower()
    normalized_question = " ".join(question.strip().casefold().split())
    key = (level, normalized_question)
    digest = hashlib.sha256(f"{level}\0{normalized_question}".encode("utf-8")).digest()

    if level == "Apply":
        if re.search(r"\b(calculat|comput|numeric|equation|formula|percentage|probability|sum|average|distance|speed|cost|rate)\w*\b", q):
            pool = ["Calculate", "Compute", "Determine"]
        elif re.search(r"\b(error|debug|fault|failure|bug|diagnos)\w*\b", q):
            pool = ["Diagnose", "Test", "Investigate"]
        elif re.search(r"\b(given|input|output|scenario|case|example|algorithm|procedure|method)\b", q):
            pool = ["Use", "Apply", "Demonstrate", "Execute"]
        else:
            pool = ["Demonstrate", "Apply", "Use", "Determine"]

    elif level == "Analyze":
        if re.search(r"\b(compare|versus|difference|differentiate|contrast|similarit)\w*\b", q):
            pool = ["Compare", "Contrast", "Differentiate", "Distinguish"]
        elif re.search(r"\b(step|sequence|process|flow|trace|execution|order)\w*\b", q):
            pool = ["Trace", "Map", "Examine", "Analyze"]
        elif re.search(r"\b(error|fault|failure|bug|problem|cause|why|purpose|function|role|work|works|protect|protection)\w*\b", q):
            pool = ["Examine", "Investigate", "Analyze", "Dissect"]
        else:
            pool = ["Examine", "Analyze", "Investigate", "Map", "Dissect"]

    elif level == "Evaluate":
        if re.search(r"\b(recommend|choose|select|alternative|option|best|which)\w*\b", q):
            pool = ["Recommend", "Rank", "Prioritize", "Assess"]
        elif re.search(r"\b(valid|correct|accurate|prove|evidence|claim|argument|justify)\w*\b", q):
            pool = ["Justify", "Critique", "Validate", "Defend"]
        else:
            pool = ["Assess", "Evaluate", "Appraise", "Critique"]

    else:
        pool = list(_ACTION_VERB_BANKS[level])

    offset = int.from_bytes(digest[:4], "big") % len(pool)
    turn = _ACTION_VERB_ROTATION.get(key, 0)
    _ACTION_VERB_ROTATION[key] = turn + 1
    return [pool[(offset + turn) % len(pool)]]


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
        "Vary the opening verb naturally across suitable tasks; do not default to 'Analyze' when Examine, Investigate, Compare, Contrast, Trace, Map, Dissect, Differentiate, or Distinguish better fits the original question. "
        "Do not invent parts or factors absent from the source; use only relationships or mechanisms inherent to the original topic. "
        "Possible forms: 'Examine how [original concept] works'; 'Investigate how [original concept] affects [source-grounded purpose]'; 'Compare [source-grounded alternatives]'; 'Trace [source-grounded process].'"
    ),
    "Evaluate": (
        "Cognitive operation: make and support a defensible judgment about effectiveness, quality, validity, suitability, or merit. "
        "Preferred action verbs (choose the most natural and semantically accurate option): Evaluate, Assess, Justify, Critique, Appraise, Defend, Judge, Recommend, Validate, Prioritize, Rate, Rank, Argue, Weigh, Verify, Review, Support, Refute, Decide, Substantiate, Endorse, Reject, Audit, Challenge, Benchmark, Critically review, Determine merit, Establish validity, Select with justification, Resolve using criteria. "
        "Require a defensible judgment about a specific property, capability, suitability, outcome, or trade-off that genuinely belongs to the named subject. The question must make clear WHAT is being judged and WHY, but must not manufacture a rubric or pretend the source supplied one. "
        "ANTI-TEMPLATE RULE: Never default to 'meets its stated requirements and constraints', 'justify your judgment using relevant criteria', 'based on the given specifications', or similar stock endings. Use such wording only when the source explicitly supplies requirements, constraints, criteria, or specifications. "
        "Do not append 'justify your judgment' as filler: the judgment itself must identify a meaningful subject-specific issue. Do not transform every Evaluate question into a requirements-compliance audit. "
        "When the source asks to design/build/create an object but gives no design artifact to inspect, do not pretend a design already exists or that requirements were stated. Reframe the target task as a bounded judgment about the object's inherent suitability or capability for its named purpose, using ordinary domain knowledge and no invented scenario or measurements. "
        "Example: Original 'design a 8bit adder.' → 'Assess the suitability of an 8-bit adder for adding 8-bit binary numbers, considering its bit width and carry-out behavior.' This judges a concrete property without inventing requirements, test results, or a rubric. "
        "Contrastive failure: 'Assess whether an 8bit adder meets its stated requirements and constraints, and justify your judgment using relevant criteria.' This is invalid when the source states no such requirements or constraints. "
        "For a source that names an existing system and its purpose, judge a concrete inherent outcome or trade-off; for example, a firewall question may concern traffic filtering and unauthorized-access prevention. Do not copy the wording or structure of one example to unrelated subjects. "
        "If no meaningful subject-specific judgment can be formed without inventing context, keep the question narrow rather than adding a generic template or fabricated evidence."
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
        "Target-level rewrite requirements (not linguistic moderation):\n"
        + (
            f"- Begin with this required action verb: {selected_action_verb}. "
            "Use this exact opening verb (capitalization may vary) only if it naturally expresses the requested cognitive operation.\n"
            if selected_action_verb
            else "- Start with a suitable action verb from the requested level when grammatically natural.\n"
        )
        + "- Change the cognitive work the student must perform so it matches the fixed target level; do not merely proofread or rephrase the original.\n"
        "- Preserve the original topic, intended purpose, entities, and scope. Make the target-level task specific and answerable without adding generic filler.\n"
        "- For Analyze, state a topic-grounded relationship, structure, difference, pattern, or cause. For Evaluate, require a defensible, topic-specific judgment. For Apply, require use of a method or concept to reach a result. For Create, specify a relevant new deliverable.\n"
        "- Add only minimal topic-grounded clarification. Never invent a scenario, supplied data, measurements, constraints, or entities, and never imply that evidence or requirements were supplied when they were not.\n"
        "- Do not add a second task, answer the question, explain changes, or judge linguistic quality. Return one clear student-facing exam question only."
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
