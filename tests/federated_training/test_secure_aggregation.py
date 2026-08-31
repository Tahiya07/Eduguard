"""Tests for secure aggregation simulator."""

from __future__ import annotations

import torch

from training.federated.secure_aggregation import (
    generate_pairwise_masks,
    mask_client_update,
    unmask_sum,
    verify_mask_cancellation,
)


def test_pairwise_masks_cancel():
    template = {"w": torch.zeros(3)}
    updates = [
        {"w": torch.tensor([1.0, 0.0, 0.0])},
        {"w": torch.tensor([0.0, 1.0, 0.0])},
        {"w": torch.tensor([0.0, 0.0, 1.0])},
    ]
    client_ids = ["a", "b", "c"]
    masks = generate_pairwise_masks(client_ids, template, round_idx=1, master_seed=42)
    masked = [mask_client_update(u, masks[cid]) for u, cid in zip(updates, client_ids)]
    assert verify_mask_cancellation(updates, masked)
    summed = unmask_sum(masked)
    expected = unmask_sum(updates)
    for key in expected:
        assert torch.allclose(summed[key].float(), expected[key].float(), atol=1e-5)
