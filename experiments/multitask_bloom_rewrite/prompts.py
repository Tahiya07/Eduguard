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
    "Preserve the source question's subject, entities, core intent, scope, and stated constraints. "
    "Silently identify the original task, the target level's distinct cognitive operation, and the smallest change needed to achieve it. "
    "Change what the student must think or do, not merely the opening verb or sentence structure. "
    "Make the task specific and answerable: state the relationship to examine, purpose or outcome to judge, method to apply, or artifact to produce. "
    "Avoid generic filler such as 'using relevant criteria and evidence' when a meaningful topic-specific purpose or criterion can be stated. "
    "Add only minimal, topic-grounded clarification; do not invent a scenario, supplied data, measurements, constraints, entities, or unsupported facts. "
    "Never imply that requirements, constraints, specifications, evidence, or criteria were stated or supplied unless the source actually provides them. "
    "If the source is brief, evaluate an inherent, topic-grounded property or purpose without pretending a rubric or requirements were given. "
    "Ordinary domain knowledge may clarify a topic's established purpose, but must not introduce a new case. "
    "Do not make the question needlessly long or combine unrelated tasks. Before output, silently check topic preservation, target cognition, answerability, natural wording, and absence of invented context; revise if any check fails. "
    "Do not force an awkward verb; use the selected verb only when it expresses the intended task naturally. "
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
        "Require a judgment and a reason for it, grounded in criteria, evidence, standards, or trade-offs relevant to the original topic. Name a topic-specific purpose, outcome, or criterion when safely inferable; avoid vague filler with no indication of what is being judged. "
        "For a question about a known system's purpose, assess how effectively it fulfils that purpose without inventing a deployment, dataset, measurements, or named rubric. Do not merely request an explanation or list of features, and do not ask for a new design. "
        "Do not use generic phrases such as 'meets its stated requirements and constraints' unless the source actually states requirements or constraints. Do not imply that a test, specification, rubric, or evidence set was supplied when it was not. "
        "Choose a criterion grounded in the named topic and original intent; where no specific criterion is supplied, phrase the judgment without claiming that one was specified. "
        "Example: Original 'Explain the purpose of a firewall.' → 'Assess how effectively a firewall fulfils its purpose of controlling network traffic and preventing unauthorized access, and justify your judgment using relevant security considerations.' Treat this as a reasoning pattern, not wording to copy for unrelated topics."
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
        + "- Preserve the original topic and intended purpose, then change the student's cognitive operation to match the target level.\n"
        "- Make the task specific and answerable; state the relevant relationship, outcome, judgment, method, or deliverable instead of adding generic filler.\n"
        "- Add only minimal topic-grounded clarification. Never invent a scenario, supplied data, measurements, constraints, or entities.\n"
        "- For Evaluate, require a defensible judgment and name the relevant purpose/outcome or criterion when safely inferable; do not merely append 'using relevant criteria and evidence'.\n"
        "- For Analyze, specify a source-grounded relationship, structure, difference, or cause; do not invent parts or factors.\n"
        "- For Apply, require use of a method or concept to reach a result. For Create, specify a relevant new deliverable.\n"
        "- Avoid padding, unrelated tasks, or changes to the original scope. Return one clear, natural exam question only; no answer or explanation."
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
