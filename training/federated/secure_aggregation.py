"""Research-grade secure aggregation simulator (pairwise mask cancellation).

This is NOT production MPC. Under documented assumptions the server learns only
the sum of client updates, not individual updates.

Assumptions:
- Synchronous rounds with full participation (or documented dropout handling).
- Pairwise masks cancel when all clients submit (Bonawitz et al. style simplification).
- No colluding client pairs trying to recover others' masks.
- Trusted RNG for mask generation (research simulation only).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import torch


def _pair_seed(round_idx: int, client_a: str, client_b: str, master_seed: int) -> int:
    pair = tuple(sorted((client_a, client_b)))
    material = f"{master_seed}:{round_idx}:{pair[0]}:{pair[1]}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def _mask_tensor(shape: torch.Size, dtype: torch.dtype, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(dtype)


def generate_pairwise_masks(
    client_ids: Sequence[str],
    state_template: Dict[str, torch.Tensor],
    *,
    round_idx: int,
    master_seed: int = 42,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Return per-client masks that sum to zero across clients when all participate."""
    ids = list(client_ids)
    masks: Dict[str, Dict[str, torch.Tensor]] = {cid: {} for cid in ids}
    for i, ci in enumerate(ids):
        for cj in ids[i + 1 :]:
            seed = _pair_seed(round_idx, ci, cj, master_seed)
            for key, tensor in state_template.items():
                m = _mask_tensor(tensor.shape, tensor.dtype, seed)
                masks[ci][key] = masks[ci].get(key, torch.zeros_like(tensor)) + m
                masks[cj][key] = masks[cj].get(key, torch.zeros_like(tensor)) - m
    return masks


def mask_client_update(
    update: Dict[str, torch.Tensor],
    mask: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: update[k] + mask[k] for k in update if k in mask}


def unmask_sum(
    masked_updates: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not masked_updates:
        raise ValueError("no masked updates")
    keys = list(masked_updates[0].keys())
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        acc = None
        for upd in masked_updates:
            t = upd[key].float()
            acc = t if acc is None else acc + t
        out[key] = acc.to(dtype=masked_updates[0][key].dtype)  # type: ignore[union-attr]
    return out


def verify_mask_cancellation(
    original_updates: List[Dict[str, torch.Tensor]],
    masked_updates: List[Dict[str, torch.Tensor]],
    *,
    atol: float = 1e-4,
) -> bool:
    """Verify sum(masked) == sum(original) when masks cancel."""
    if len(original_updates) != len(masked_updates):
        return False
    summed_orig = unmask_sum(original_updates)
    summed_masked = unmask_sum(masked_updates)
    for key in summed_orig:
        if not torch.allclose(summed_orig[key].float(), summed_masked[key].float(), atol=atol):
            return False
    return True
