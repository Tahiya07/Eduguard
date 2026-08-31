"""Federated communication byte accounting from serialized transport payloads."""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Mapping

import torch

from training.federated.aggregation import is_trainable_key, trainable_param_count
from training.federated.transport import serialize_state_dict


def bundle_serialized_upload_bytes(bundle: Mapping[str, Any]) -> int:
    """Bytes of the client update payload as transported in a bundle."""
    if bundle.get("serialized_update_bytes") is not None:
        return int(bundle["serialized_update_bytes"])
    if bundle.get("update_bytes") is not None:
        return int(bundle["update_bytes"])
    payload = bundle.get("payload_b64")
    if payload:
        return len(base64.b64decode(payload))
    return 0


def serialized_state_bytes(state: Dict[str, torch.Tensor]) -> int:
    """Size of the torch-serialized trainable state dict."""
    return len(serialize_state_dict(state))


def trainable_param_breakdown(state: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    """Count LoRA vs score-head trainable parameters."""
    lora_params = 0
    score_params = 0
    for key, tensor in state.items():
        if not is_trainable_key(key):
            continue
        n = int(tensor.numel())
        lowered = key.lower()
        if "lora_" in lowered:
            lora_params += n
        elif ".score." in lowered or lowered.endswith("classifier.weight") or lowered.endswith(
            "classifier.bias"
        ):
            score_params += n
    total = trainable_param_count(dict(state))
    return {
        "lora_trainable_parameters": lora_params,
        "score_head_trainable_parameters": score_params,
        "total_trainable_parameters": total,
    }


def measure_round_communication(
    bundles: List[Mapping[str, Any]],
    *,
    global_state_serialized_bytes: int,
) -> Dict[str, Any]:
    """Measure upload/download for one federated round."""
    per_client_upload: Dict[str, int] = {}
    for bundle in bundles:
        client_id = str(bundle.get("client_id", "unknown"))
        nbytes = bundle_serialized_upload_bytes(bundle)
        per_client_upload[client_id] = nbytes

    upload_total = int(sum(per_client_upload.values()))
    n_clients = len(bundles)
    per_client_download = {
        client_id: int(global_state_serialized_bytes) for client_id in per_client_upload
    }
    download_total = int(global_state_serialized_bytes) * n_clients

    return {
        "upload_bytes_total": upload_total,
        "download_bytes_total": download_total,
        "communication_bytes_total": upload_total + download_total,
        "per_client_upload_bytes": per_client_upload,
        "per_client_download_bytes": per_client_download,
        "global_state_serialized_bytes": int(global_state_serialized_bytes),
        "n_clients": n_clients,
    }


def accumulate_communication(
  round_records: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Sum per-round communication records into experiment totals."""
    total_upload = 0
    total_download = 0
    per_round_upload: List[int] = []
    per_round_download: List[int] = []
    for record in round_records:
        upload = int(record.get("upload_bytes") or record.get("upload_bytes_total") or 0)
        download = int(record.get("download_bytes") or record.get("download_bytes_total") or 0)
        total_upload += upload
        total_download += download
        per_round_upload.append(upload)
        per_round_download.append(download)
    return {
        "total_upload_bytes": total_upload,
        "total_download_bytes": total_download,
        "per_round_upload_bytes": per_round_upload,
        "per_round_download_bytes": per_round_download,
        "total_bytes": total_upload + total_download,
    }
