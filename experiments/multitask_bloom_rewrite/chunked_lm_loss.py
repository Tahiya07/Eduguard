"""Memory-bounded causal-LM loss over supervised tokens only.

Hugging Face causal LMs project every position to the full vocabulary and
upcast to fp32 before cross-entropy. For Qwen2.5 (151,936-token vocabulary) a
batch of ~5k padded tokens needs ~3 GB just for logits, although only the answer
tokens (labels != -100) contribute to the loss. In token-budget batches the
earliest answer start in the batch decides how much can be cut, so slicing with
``logits_to_keep`` alone is not enough.

This module runs the decoder once, gathers hidden states at supervised
positions, and applies the LM head + cross-entropy in checkpointed chunks, so
peak logits memory is ``chunk_size x vocab`` regardless of batch shape. The
result equals the model's own loss: sum of token losses divided by
``num_items_in_batch`` (or by the number of supervised tokens when not given).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_loss_sum(hidden: torch.Tensor, targets: torch.Tensor, lm_head: torch.nn.Module) -> torch.Tensor:
    logits = lm_head(hidden).float()
    return F.cross_entropy(logits, targets, reduction="sum")


def decoder_and_head(causal_lm: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Return (decoder, lm_head) of a (possibly PEFT-wrapped) HF causal LM."""
    model = causal_lm
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    head = model.get_output_embeddings()
    if decoder is None or head is None:
        raise TypeError("Model does not expose a decoder and an output embedding layer.")
    return decoder, head


def supervised_token_loss(
    causal_lm: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    num_items_in_batch: torch.Tensor | int | None = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    decoder, lm_head = decoder_and_head(causal_lm)
    hidden = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    # Position t predicts token t+1.
    targets = labels[:, 1:]
    keep = targets.ne(-100)
    hidden = hidden[:, :-1][keep]
    targets = targets[keep]
    if targets.numel() == 0:
        return hidden.sum() * 0.0
    total = hidden.new_zeros((), dtype=torch.float32)
    use_checkpoint = torch.is_grad_enabled() and hidden.requires_grad
    for start in range(0, targets.numel(), chunk_size):
        h = hidden[start:start + chunk_size]
        t = targets[start:start + chunk_size]
        if use_checkpoint:
            total = total + checkpoint(_chunk_loss_sum, h, t, lm_head, use_reentrant=False)
        else:
            total = total + _chunk_loss_sum(h, t, lm_head)
    if num_items_in_batch is None:
        denominator = targets.numel()
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(total.device)
    else:
        denominator = num_items_in_batch
    return total / denominator
