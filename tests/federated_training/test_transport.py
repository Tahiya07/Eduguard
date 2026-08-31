"""Tests for transport integrity."""

from __future__ import annotations

import pytest
import torch

from training.federated.transport import pack_update, unpack_update


def test_roundtrip_integrity():
    state = {"layer.lora_A": torch.randn(4, 2), "layer.score.weight": torch.randn(6, 4)}
    bundle = pack_update(
        client_id="c0",
        round_idx=1,
        role="teacher",
        n_samples=10,
        state=state,
    )
    assert bundle["serialized_update_bytes"] > 0
    assert bundle["serialized_update_bytes"] == len(__import__("base64").b64decode(bundle["payload_b64"]))
    out = unpack_update(bundle)
    for key in state:
        assert torch.allclose(state[key], out[key])


def test_rejects_legacy_xor_bundle():
    bundle = {
        "format": "federated_lora_update_v1",
        "encrypted": True,
        "payload_b64": "AAAA",
    }
    with pytest.raises(ValueError, match="Legacy XOR"):
        unpack_update(bundle)
