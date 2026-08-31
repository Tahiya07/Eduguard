"""Client update transport with integrity only — NOT encryption or secure aggregation.

Provides SHA-256 integrity over serialized LoRA state payloads.
Confidentiality is NOT provided. Use secure_aggregation.py for masked aggregation.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict

import torch

BUNDLE_FORMAT = "federated_lora_update_v2"
TRANSPORT_KIND = "json+sha256-integrity"


def serialize_state_dict(state: Dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def deserialize_state_dict(payload: bytes) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(io.BytesIO(payload), map_location="cpu")


def pack_update(
    *,
    client_id: str,
    round_idx: int,
    role: str,
    n_samples: int,
    state: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    raw = serialize_state_dict(state)
    digest = hashlib.sha256(raw).hexdigest()
    payload = base64.b64encode(raw).decode("ascii")
    return {
        "format": BUNDLE_FORMAT,
        "transport": TRANSPORT_KIND,
        "client_id": client_id,
        "round": int(round_idx),
        "role": role,
        "n_samples": int(n_samples),
        "serialized_update_bytes": len(raw),
        "sha256_plaintext": digest,
        "payload_b64": payload,
        "privacy_note": (
            "Integrity protection only. This bundle is not encrypted and does not "
            "provide differential privacy or secure aggregation."
        ),
    }


def unpack_update(bundle: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if bundle.get("format") not in {BUNDLE_FORMAT, "federated_lora_update_v1"}:
        raise ValueError(f"unsupported bundle format: {bundle.get('format')!r}")
    raw = base64.b64decode(bundle["payload_b64"])
    if bundle.get("format") == "federated_lora_update_v1" and bundle.get("encrypted"):
        raise ValueError(
            "Legacy XOR bundles are not supported in EduGuard. "
            "Re-export client updates with training.federated.transport."
        )
    expected = bundle.get("sha256_plaintext")
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("update bundle integrity check failed (sha256 mismatch)")
    return deserialize_state_dict(raw)


def save_bundle(path: Path, bundle: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")


def load_bundle(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
