"""Qwen2.5-Instruct SFT format for Bloom target-level rewriting.

Production-aligned task (identical for 0.5B and 1.5B):

    f(original_question, target_bloom_level) -> target_rewrite

Source Bloom level is dataset metadata only and must NEVER appear in
generator prompts (training or evaluation).
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert academic assessment editor. "
    "Rewrite an academic question so that the student's required cognitive "
    "task matches the requested Bloom level. "
    "Preserve the original topic, important technical concepts, and academic "
    "intent. "
    "Do not merely replace verbs. "
    "Output only one student-facing exam question."
)

USER_TEMPLATE = (
    "Original question:\n{source_question}\n\n"
    "Target Bloom level:\n{target_level}"
)

# Forbidden substrings: must never appear in generator prompts.
FORBIDDEN_SOURCE_LEVEL_MARKERS = (
    "Original Bloom level:",
    "Source Bloom level:",
    "source bloom level",
    "original bloom level:",
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_user_message(source_question: str, target_level: str) -> str:
    return USER_TEMPLATE.format(
        source_question=source_question.strip(),
        target_level=target_level,
    )


def build_messages(
    source_question: str,
    target_level: str,
    target_rewrite: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_message(source_question, target_level),
        },
    ]
    if target_rewrite is not None:
        messages.append({"role": "assistant", "content": target_rewrite.strip()})
    return messages


def render_chatml(messages: list[dict[str, str]], *, add_generation_prompt: bool = False) -> str:
    parts: list[str] = []
    for message in messages:
        parts.append(f"{IM_START}{message['role']}\n{message['content']}{IM_END}")
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "\n".join(parts)


def assert_no_source_level_in_prompt(text: str) -> None:
    lowered = text.lower()
    for marker in FORBIDDEN_SOURCE_LEVEL_MARKERS:
        if marker.lower() in lowered:
            raise AssertionError(
                f"Generator prompt must not contain source Bloom level marker {marker!r}"
            )


def build_sft_text(
    source_question: str,
    target_level: str,
    target_rewrite: str,
) -> str:
    text = render_chatml(build_messages(source_question, target_level, target_rewrite))
    assert_no_source_level_in_prompt(text)
    return text


def build_generation_prompt(source_question: str, target_level: str) -> str:
    text = render_chatml(
        build_messages(source_question, target_level),
        add_generation_prompt=True,
    )
    assert_no_source_level_in_prompt(text)
    return text


def describe_training_task() -> dict[str, str]:
    return {
        "TRAINING_TASK": "question + target_level → rewrite",
        "SOURCE_LEVEL": "metadata only",
        "TARGET_LEVEL": "generator input",
        "EVALUATION_TASK": "question + target_level → rewrite",
    }
