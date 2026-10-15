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
    "You are an expert academic assessment editor. Rewrite the source exam task so "
    "the student's required reasoning genuinely matches the requested Revised Bloom level. "
    "Preserve the source's subject, entities, scope, and essential constraints; do not add "
    "unsupported facts or silently change the scenario. Do not merely swap the opening verb. "
    "Design the task around the cognitive operation described for the requested level. "
    "A verb alone never determines a Bloom level: judge the full action, object, evidence, "
    "criteria, and expected student work. Make the cognitive demand explicit in the task. "
    "Return exactly one student-facing exam question or clear exam imperative, with no answer, "
    "preface, level label, explanation, bullet list, or meta commentary."
)

BLOOM_LEVEL_GUIDANCE: dict[str, str] = {
    "Remember": (
        "Require retrieval or recognition of relevant information, such as identifying, "
        "naming, listing, or stating facts. Do not ask for a causal explanation or judgment."
    ),
    "Understand": (
        "Require explaining, interpreting, classifying, or summarizing meaning in the "
        "source topic. Do not stop at recall, and do not require a defended judgment."
    ),
    "Apply": (
        "Require using a concept, rule, method, or procedure in a specified problem, case, "
        "or concrete situation. Make the application task explicit."
    ),
    "Analyze": (
        "Require examining parts, relationships, assumptions, causes, evidence, or structure. "
        "Ask the student to distinguish how or why components relate, not merely describe them."
    ),
    "Evaluate": (
        "Require a reasoned judgment against explicit or inferable criteria, supported by "
        "evidence or justification. Asking to explain or describe alone is insufficient."
    ),
    "Create": (
        "Require producing or proposing a coherent new design, plan, model, solution, or "
        "artifact that satisfies the source topic and any stated constraints. Naming an idea "
        "or describing an existing design alone is insufficient."
    ),
}

BLOOM_LEVEL_ALIASES = {level.casefold(): level for level in BLOOM_LEVEL_GUIDANCE}

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
    canonical_level = BLOOM_LEVEL_ALIASES.get(target_level.strip().casefold())
    if canonical_level is None:
        raise ValueError(
            f"Unknown target Bloom level {target_level!r}; expected one of "
            f"{', '.join(BLOOM_LEVEL_GUIDANCE)}"
        )
    guidance = BLOOM_LEVEL_GUIDANCE[canonical_level]
    user = (
        f"Original question:\n{question.strip()}\n\n"
        f"Target Bloom level:\n{canonical_level}\n"
        f"Required cognitive operation:\n{guidance}\n\n"
        "Rewrite the task so a student's answer must demonstrate this operation. "
        "Retain the original topic and constraints; do not answer it."
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
