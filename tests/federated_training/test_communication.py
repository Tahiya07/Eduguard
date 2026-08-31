"""Communication byte accounting and trainable parameter reporting tests."""

from __future__ import annotations

import torch

from training.federated.aggregation import is_trainable_key, trainable_param_count
from training.federated.communication import (
    accumulate_communication,
    bundle_serialized_upload_bytes,
    measure_round_communication,
    serialized_state_bytes,
    trainable_param_breakdown,
)
from training.federated.transport import pack_update, unpack_update


def _sample_state() -> dict:
    return {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.default": torch.randn(4, 2),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.default": torch.randn(2, 4),
        "base_model.score.weight": torch.randn(6, 8),
        "base_model.model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
    }


def test_bundle_serialized_upload_bytes_matches_payload():
    state = _sample_state()
    bundle = pack_update(
        client_id="teacher_site_00",
        round_idx=1,
        role="teacher",
        n_samples=10,
        state=state,
    )
    nbytes = bundle_serialized_upload_bytes(bundle)
    assert nbytes == int(bundle["serialized_update_bytes"])
    assert nbytes > 0
    restored = unpack_update(bundle)
    assert trainable_param_count(restored) == trainable_param_count(state)


def test_trainable_param_breakdown_matches_state():
    state = {k: v for k, v in _sample_state().items() if is_trainable_key(k)}
    breakdown = trainable_param_breakdown(state)
    assert breakdown["total_trainable_parameters"] == trainable_param_count(state)
    assert breakdown["lora_trainable_parameters"] > 0
    assert breakdown["score_head_trainable_parameters"] > 0


def test_serialized_state_bytes_positive():
    state = {k: v for k, v in _sample_state().items() if is_trainable_key(k)}
    assert serialized_state_bytes(state) > trainable_param_count(state)


def test_measure_round_communication_totals():
    bundles = []
    for i in range(3):
        bundle = pack_update(
            client_id=f"teacher_site_{i:02d}",
            round_idx=1,
            role="teacher",
            n_samples=5 + i,
            state={k: v for k, v in _sample_state().items() if is_trainable_key(k)},
        )
        bundles.append(bundle)
    comm = measure_round_communication(bundles, global_state_serialized_bytes=1000)
    assert comm["upload_bytes_total"] == sum(bundle_serialized_upload_bytes(b) for b in bundles)
    assert comm["download_bytes_total"] == 3000
    assert len(comm["per_client_upload_bytes"]) == 3


def test_accumulate_communication_from_history():
    history = [
        {"upload_bytes": 10, "download_bytes": 20},
        {"upload_bytes": 30, "download_bytes": 40},
    ]
    totals = accumulate_communication(history)
    assert totals["total_upload_bytes"] == 40
    assert totals["total_download_bytes"] == 60
    assert totals["per_round_upload_bytes"] == [10, 30]
