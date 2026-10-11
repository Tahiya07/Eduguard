"""Task-weighted, sequence-normalized causal-LM loss for multitask SFT.

The usual causal-LM loss averages over all assistant tokens in a batch. This
can let long summaries dominate shorter Bloom rewrites even when task row
counts are balanced. This helper first averages loss over supervised tokens
within each example, then applies the configured task weight per example.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TASK_TO_ID = {
    "bloom_rewrite": 0,
    "qa": 1,
    "summarization": 2,
}


def task_weighted_causal_lm_loss(
    logits: Any,
    labels: Any,
    task_ids: Any,
    task_weights: Mapping[int, float],
):
    """Compute weighted mean of per-example assistant-token cross entropy.

    Args:
        logits: Model logits with shape [batch, sequence, vocabulary].
        labels: Token labels with shape [batch, sequence], using -100 for
            prompt/padding positions.
        task_ids: Integer task IDs with shape [batch].
        task_weights: Positive weights keyed by TASK_TO_ID values.

    Returns:
        A scalar tensor suitable for backpropagation and validation loss.
    """
    import torch
    import torch.nn.functional as F

    if logits.ndim != 3:
        raise ValueError(f"logits must be rank 3; got shape {tuple(logits.shape)}")
    if labels.ndim != 2 or tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError("labels must have shape [batch, sequence] matching logits")
    if task_ids.ndim != 1 or task_ids.shape[0] != labels.shape[0]:
        raise ValueError("task_ids must have one entry per batch example")
    if not task_weights or any(float(weight) <= 0 for weight in task_weights.values()):
        raise ValueError("task_weights must contain positive weights")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if shift_logits.shape[1] == 0:
        raise ValueError("Each sequence must contain at least two tokens")

    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    supervised = shift_labels.ne(-100)
    token_counts = supervised.sum(dim=1)
    if torch.any(token_counts == 0):
        raise ValueError("Every example must contain at least one supervised target token")

    per_example_loss = (token_losses * supervised).sum(dim=1) / token_counts
    weights = []
    for task_id in task_ids.detach().cpu().tolist():
        if int(task_id) not in task_weights:
            raise ValueError(f"No configured loss weight for task ID {task_id}")
        weights.append(float(task_weights[int(task_id)]))
    weight_tensor = torch.tensor(
        weights, device=per_example_loss.device, dtype=per_example_loss.dtype
    )
    # Deliberately divide by batch size, not sum(weights): the configured
    # weights must affect both training gradients and eval_loss/checkpoint choice.
    return (per_example_loss * weight_tensor).mean()
