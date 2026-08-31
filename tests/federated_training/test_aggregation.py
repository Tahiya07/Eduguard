"""Tests for FedAvg aggregation math."""

from __future__ import annotations

import torch

from training.federated.aggregation import (
    apply_delta,
    clip_delta,
    extract_trainable_state,
    fedavg_deltas,
    fedavg_state_dicts,
    is_trainable_key,
    state_dict_to_delta,
)


def test_fedavg_equal_weights():
    global_state = {"w": torch.tensor([1.0, 2.0])}
    local_a = {"w": torch.tensor([2.0, 4.0])}
    local_b = {"w": torch.tensor([0.0, 0.0])}
    delta_a = state_dict_to_delta(local_a, global_state)
    delta_b = state_dict_to_delta(local_b, global_state)
    merged = fedavg_deltas([(1, delta_a), (1, delta_b)], global_state)
    new_state = apply_delta(global_state, merged, scale=1.0)
    assert torch.allclose(new_state["w"], torch.tensor([1.0, 2.0]))


def test_clip_delta():
    delta = {"w": torch.tensor([3.0, 4.0])}
    clipped = clip_delta(delta, clip_norm=2.5)
    norm = float(torch.norm(clipped["w"]))
    assert norm <= 2.5 + 1e-6


def test_fedavg_unequal_sample_weights():
    global_state = {"w": torch.tensor([0.0])}
    local_a = {"w": torch.tensor([10.0])}
    local_b = {"w": torch.tensor([0.0])}
    merged = fedavg_state_dicts([(9, local_a), (1, local_b)])
    assert torch.allclose(merged["w"], torch.tensor([9.0]))


def test_fedavg_includes_lora_and_score_head():
    global_state = {
        "lora_a": torch.tensor([1.0, 1.0]),
        "lora_b": torch.tensor([2.0, 2.0]),
        "score.weight": torch.tensor([3.0]),
    }
    local_a = {
        "lora_a": torch.tensor([3.0, 3.0]),
        "lora_b": torch.tensor([4.0, 4.0]),
        "score.weight": torch.tensor([5.0]),
    }
    local_b = {
        "lora_a": torch.tensor([1.0, 1.0]),
        "lora_b": torch.tensor([2.0, 2.0]),
        "score.weight": torch.tensor([1.0]),
    }
    delta_a = state_dict_to_delta(local_a, global_state)
    delta_b = state_dict_to_delta(local_b, global_state)
    merged_delta = fedavg_deltas([(1, delta_a), (1, delta_b)], global_state)
    new_state = apply_delta(global_state, merged_delta, scale=1.0)
    assert torch.allclose(new_state["lora_a"], torch.tensor([2.0, 2.0]))
    assert torch.allclose(new_state["score.weight"], torch.tensor([3.0]))


def test_trainable_key_includes_lora_and_score():
    assert is_trainable_key("base_model.model.layers.0.self_attn.q_proj.lora_A.default")
    assert is_trainable_key("base_model.score.weight")
    assert not is_trainable_key("base_model.model.layers.0.self_attn.q_proj.weight")
