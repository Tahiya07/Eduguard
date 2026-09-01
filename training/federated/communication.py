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


REQUIRED_BUNDLE_COMMUNICATION_KEYS = (
    "trainable_parameter_count",
    "update_bytes",
    "adapter_size_bytes",
)

REQUIRED_ROUND_COMMUNICATION_KEYS = (
    "upload_bytes_total",
    "download_bytes_total",
    "per_client_upload_bytes",
)


def attach_client_communication_metadata(
    bundle: Dict[str, Any],
    local_state: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """Attach explicit communication metadata to a client update bundle."""
    breakdown = trainable_param_breakdown(local_state)
    update_bytes = bundle_serialized_upload_bytes(bundle)
    if update_bytes <= 0:
        raise ValueError("client update_bytes must be positive for non-empty trainable state")
    comm = {
        "trainable_parameter_count": int(breakdown["total_trainable_parameters"]),
        "trainable_param_breakdown": breakdown,
        "update_bytes": int(update_bytes),
        "adapter_size_bytes": int(update_bytes),
    }
    bundle["communication"] = comm
    bundle["trainable_parameters"] = comm["trainable_parameter_count"]
    bundle["update_bytes"] = comm["update_bytes"]
    bundle["trainable_param_breakdown"] = comm["trainable_param_breakdown"]
    return bundle


def require_bundle_communication(bundle: Mapping[str, Any], *, context: str = "bundle") -> None:
    """Raise if a client bundle lacks required communication metadata."""
    comm = bundle.get("communication")
    if not isinstance(comm, dict):
        raise ValueError(f"{context}: missing communication block")
    for key in REQUIRED_BUNDLE_COMMUNICATION_KEYS:
        if comm.get(key) is None:
            raise ValueError(f"{context}: communication.{key} is required")
    if int(comm["update_bytes"]) <= 0:
        raise ValueError(f"{context}: communication.update_bytes must be positive")


def require_round_communication(
    comm: Mapping[str, Any],
    *,
    context: str,
    n_clients: int,
) -> None:
    """Raise if a completed round lacks measured communication totals."""
    if n_clients <= 0:
        raise ValueError(f"{context}: n_clients must be positive")
    for key in REQUIRED_ROUND_COMMUNICATION_KEYS:
        if comm.get(key) is None:
            raise ValueError(f"{context}: missing {key}")
    upload = int(comm["upload_bytes_total"])
    download = int(comm["download_bytes_total"])
    if upload <= 0:
        raise ValueError(f"{context}: upload_bytes_total must be positive")
    if download <= 0:
        raise ValueError(f"{context}: download_bytes_total must be positive")
    per_client = comm.get("per_client_upload_bytes") or {}
    if len(per_client) != n_clients:
        raise ValueError(
            f"{context}: expected per_client_upload_bytes for {n_clients} clients, got {len(per_client)}"
        )


def require_result_communication(
    comm_block: Mapping[str, Any],
    *,
    configured_rounds: int,
    completed_rounds: int,
) -> None:
    """Raise if an executed FL result lacks non-zero communication accounting."""
    if completed_rounds <= 0:
        raise ValueError("completed_rounds must be positive for executed FL results")
    upload = comm_block.get("total_upload_bytes")
    download = comm_block.get("total_download_bytes")
    if upload is None or download is None:
        raise ValueError("result communication missing total_upload_bytes/total_download_bytes")
    if int(upload) <= 0 or int(download) <= 0:
        raise ValueError("result communication totals must be positive after successful FL run")
    if comm_block.get("trainable_parameters") is None:
        raise ValueError("result communication missing trainable_parameters")
    if comm_block.get("adapter_bytes") is None or int(comm_block.get("adapter_bytes") or 0) <= 0:
        raise ValueError("result communication missing positive adapter_bytes")
    per_round_upload = comm_block.get("per_round_upload_bytes") or []
    if len(per_round_upload) != completed_rounds:
        raise ValueError(
            f"per_round_upload_bytes length {len(per_round_upload)} != completed rounds {completed_rounds}"
        )
    if configured_rounds != completed_rounds:
        raise ValueError(
            f"refusing to finalize result: completed {completed_rounds} rounds, configured {configured_rounds}"
        )


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
