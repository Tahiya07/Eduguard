"""Task-specific ChatML prompts for multi-task Qwen SFT.

Bloom contract is two-input only: question + target_level → rewrite.
Source Bloom level must never appear in generator prompts.
"""
from __future__ import annotations

from typing import Any

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
    "Change the cognitive operation the student must perform, not just the wording or verb. "
    "Use the target-level action-verb guidance supplied in the user message: choose a suitable verb from that level's bank, "
    "prefer an action-verb-led imperative when it reads naturally, and make the requested thinking operation explicit. "
    "The verb is a cue, not proof of Bloom alignment: the whole task must require the target-level cognition. "
    "Do not add unsupported facts, arbitrary scenarios, new entities, or unrelated requirements. "
    "Do not force an awkward verb; choose another suitable verb from the same target-level bank if needed. "
    "OUTPUT ONLY ONE student-facing exam question as a single sentence or short paragraph. "
    "Do not include a Bloom label, rationale, answer, solution, preamble, bullet list, or meta commentary."
)

# These are inference-time and SFT prompt instructions for the requested target level.
# Verbs are examples, not a keyword classifier; validate the cognitive operation in context.
BLOOM_TARGET_GUIDANCE = {
    "Remember": (
        "Cognitive operation: retrieve or recognize a fact, term, component, rule, or procedure. "
        "Preferred action verbs: Identify, Define, List, Name, State, Recall, Recognize, Label, Select, Match. "
        "Keep the task to retrieval/recognition; do not ask for explanation, comparison, application, judgment, or design. "
        "Example form: 'Identify the main components of [original topic].'"
    ),
    "Understand": (
        "Cognitive operation: explain meaning, interpret, classify, summarize, or describe an existing idea. "
        "Preferred action verbs: Explain, Describe, Summarize, Interpret, Classify, Illustrate, Paraphrase, Distinguish, Discuss. "
        "Require comprehension of the original concept, not a solution to a new case, a criteria-based judgment, or a new artifact. "
        "Example form: 'Explain how [original concept] works.'"
    ),
    "Apply": (
        "Cognitive operation: use a learned rule, method, or concept to work through a concrete case. "
        "Preferred action verbs: Apply, Use, Calculate, Solve, Demonstrate, Execute, Implement, Determine, Compute, Predict. "
        "Keep the original topic and provide/use a concrete given situation, inputs, or conditions when needed; require an actual result or procedure, not just explanation. "
        "Example form: 'Given [relevant conditions], apply [original method] to determine [relevant result].'"
    ),
    "Analyze": (
        "Cognitive operation: break information into parts and examine relationships, structure, causes, patterns, or differences. "
        "Preferred action verbs: Analyze, Compare, Contrast, Differentiate, Examine, Investigate, Categorize, Decompose, Trace, Relate. "
        "Make the relationship, parts, cause, or pattern to be examined explicit; do not stop at listing facts or merely asking for an unsupported opinion. "
        "Example form: 'Analyze how [original topic's parts or factors] interact.'"
    ),
    "Evaluate": (
        "Cognitive operation: make and support a judgment using explicit criteria, evidence, standards, or trade-offs. "
        "Preferred action verbs: Evaluate, Assess, Justify, Critique, Appraise, Defend, Judge, Recommend, Validate, Prioritize. "
        "Require a reasoned judgment tied to criteria/evidence; do not only describe/analyze, and do not turn the task into designing a new artifact. "
        "Example form: 'Evaluate [original approach] against [relevant criteria] and justify the judgment.'"
    ),
    "Create": (
        "Cognitive operation: produce or propose a new artifact, plan, model, design, procedure, or solution by combining ideas. "
        "Preferred action verbs: Design, Develop, Create, Construct, Formulate, Propose, Devise, Compose, Generate, Plan. "
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
        "- Start with a suitable action verb from the requested level when grammatically natural.\n"
        "- Use the verb to express the actual student task, not as a superficial keyword.\n"
        "- Preserve the original topic, entities, meaning, and relevant context.\n"
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
            bloom_messages(record["source_question"], record["target_bloom_level"]),
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
