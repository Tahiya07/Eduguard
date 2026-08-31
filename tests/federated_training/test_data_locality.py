"""Verify client transport never carries raw private data."""

from __future__ import annotations

import base64
import io

import torch

from training.federated.transport import BUNDLE_FORMAT, pack_update, unpack_update


def test_pack_update_contains_only_model_state():
    state = {"lora_A": torch.randn(2, 2)}
    bundle = pack_update(
        client_id="client_0",
        round_idx=1,
        role="teacher",
        n_samples=10,
        state=state,
    )
    assert bundle["format"] == BUNDLE_FORMAT
    assert "payload_b64" in bundle
    assert "question" not in str(bundle).lower()
    assert "answer" not in str(bundle).lower()
    raw = base64.b64decode(bundle["payload_b64"])
    loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    assert "lora_A" in loaded
    roundtrip = unpack_update(bundle)
    assert roundtrip["lora_A"].shape == (2, 2)


def test_bundle_fields_are_metadata_only():
    bundle = pack_update(
        client_id="c1",
        round_idx=0,
        role="teacher",
        n_samples=5,
        state={"x": torch.zeros(1)},
    )
    allowed_keys = {
        "format",
        "transport",
        "client_id",
        "round",
        "role",
        "n_samples",
        "serialized_update_bytes",
        "sha256_plaintext",
        "payload_b64",
        "privacy_note",
    }
    assert set(bundle.keys()) == allowed_keys
