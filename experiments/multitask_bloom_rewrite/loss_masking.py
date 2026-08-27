"""Assistant-only loss masking helpers for Qwen ChatML SFT."""
from __future__ import annotations

from typing import Any


def mask_prompt_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    """Return labels where the first prompt_len tokens are -100."""
    if prompt_len < 0:
        raise ValueError("prompt_len must be non-negative")
    prompt_len = min(prompt_len, len(input_ids))
    return [-100] * prompt_len + list(input_ids[prompt_len:])


def tokenize_with_assistant_only_loss(
    tokenizer: Any,
    full_text: str,
    prompt_text: str,
    max_length: int,
) -> dict[str, list[int]]:
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )["input_ids"]
    labels = mask_prompt_labels(tokenized["input_ids"], len(prompt_ids))
    # Safety: if prompt ate the whole sequence, keep at least the last token supervised
    # only when there is a non-empty assistant continuation in full_text beyond prompt.
    if all(x == -100 for x in labels) and len(full_text) > len(prompt_text):
        labels[-1] = tokenized["input_ids"][-1]
    tokenized["labels"] = labels
    return tokenized


def assert_assistant_only_loss(labels: list[int], prompt_len: int) -> None:
    if prompt_len > 0 and any(x != -100 for x in labels[:prompt_len]):
        raise AssertionError("Prompt tokens must be masked with -100")
    if prompt_len < len(labels) and all(x == -100 for x in labels[prompt_len:]):
        raise AssertionError("Assistant region unexpectedly fully masked")
