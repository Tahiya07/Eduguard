"""Assistant-only loss masking helpers for Qwen ChatML SFT.

The prompt and full-example strings are tokenized without adding tokenizer-specific
special tokens. The prompt token sequence must be an exact prefix of the full
sequence; otherwise masking fails instead of silently supervising the wrong span.
"""
from __future__ import annotations

from typing import Any


def mask_prompt_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    """Return labels where exactly the prompt prefix is masked with -100."""
    if prompt_len < 0:
        raise ValueError("prompt_len must be non-negative")
    if prompt_len > len(input_ids):
        raise ValueError(
            f"prompt_len={prompt_len} exceeds input length={len(input_ids)}"
        )
    if prompt_len == len(input_ids):
        raise ValueError("No assistant tokens remain after the prompt.")
    return [-100] * prompt_len + list(input_ids[prompt_len:])


def tokenize_with_assistant_only_loss(
    tokenizer: Any,
    full_text: str,
    prompt_text: str,
    max_length: int,
) -> dict[str, list[int]]:
    if not full_text.strip():
        raise ValueError("full_text must not be empty")
    if not prompt_text.strip():
        raise ValueError("prompt_text must not be empty")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt_text,
        truncation=False,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]

    if not full_ids:
        raise ValueError("Tokenization produced an empty full sequence.")
    if len(prompt_ids) >= len(full_ids):
        raise ValueError(
            "Prompt consumes the entire truncated sequence; increase max_seq_length "
            "or inspect the example."
        )
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt tokens are not an exact prefix of full-example tokens. "
            "This would make assistant-only loss masking unreliable."
        )

    labels = mask_prompt_labels(full_ids, len(prompt_ids))
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )
    tokenized["labels"] = labels
    return tokenized


def assert_assistant_only_loss(labels: list[int], prompt_len: int) -> None:
    if prompt_len < 0 or prompt_len > len(labels):
        raise ValueError("Invalid prompt_len.")
    if any(x != -100 for x in labels[:prompt_len]):
        raise AssertionError("Prompt tokens must be masked with -100")
    if prompt_len == len(labels):
        raise AssertionError("Assistant region is empty")
    if all(x == -100 for x in labels[prompt_len:]):
        raise AssertionError("Assistant region unexpectedly fully masked")


def slice_to_assistant_logits(inputs: dict) -> dict:
    """Restrict a causal-LM forward to logits from the first supervised position on.

    Prompt tokens are masked with -100 and the answer sits at the end of every
    row, so full-vocabulary logits over a long prompt are wasted memory (about
    4.4 GB in fp32 for one 7k-token row of Qwen2.5). Logit t predicts token t+1,
    so keeping positions ``start..L-1`` with ``labels[:, start:]`` gives the same
    loss and gradients. Uses ``logits_to_keep`` of Hugging Face causal LMs.
    """
    labels = inputs.get("labels")
    if labels is None or "logits_to_keep" in inputs:
        return inputs
    supervised = labels.ne(-100)
    if not bool(supervised.any(dim=1).all()):
        return inputs
    first = int(supervised.int().argmax(dim=1).min())
    start = max(first - 1, 0)
    if start == 0:
        return inputs
    sliced = dict(inputs)
    sliced["labels"] = labels[:, start:]
    sliced["logits_to_keep"] = int(labels.shape[1] - start)
    return sliced
