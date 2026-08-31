#!/usr/bin/env python
"""Federated server: aggregate client LoRA bundles (weighted FedAvg).

FedProx is applied on the client; the server averages updates.
No server-side Gaussian noise — formal DP must come from validated client DP-SGD only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.paths import ROOT
from training.federated.aggregation import (
    apply_delta,
    clip_delta,
    extract_trainable_state,
    fedavg_deltas,
    load_trainable_state,
    state_dict_to_delta,
    trainable_param_count,
)
from training.federated.communication import (
    measure_round_communication,
    serialized_state_bytes,
    trainable_param_breakdown,
)
from training.federated.config import BLOOM_LABELS, FederatedLoraConfig, make_peft_lora_config
from training.federated.transport import load_bundle, unpack_update

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_global_state(config: FederatedLoraConfig, global_dir: Path) -> Dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(BLOOM_LABELS),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    if (global_dir / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(base, str(global_dir), is_trainable=True)
    else:
        model = get_peft_model(base, make_peft_lora_config(config))
    return extract_trainable_state(model)


def _save_global_adapter(
    config: FederatedLoraConfig,
    state: Dict[str, torch.Tensor],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(BLOOM_LABELS),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(base, make_peft_lora_config(config))
    load_trainable_state(model, state)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "format": "federated_lora_global_v2",
        "aggregation": "fedavg_weighted",
        "client_algorithm": config.algorithm,
        "prox_mu": float(config.prox_mu) if config.algorithm == "fedprox" else 0.0,
        "clip_norm": config.clip_norm,
        "lora": config.lora_config_dict(),
        "base_model": config.base_model,
        "trainable_parameters": trainable_param_count(state),
        "trainable_param_breakdown": trainable_param_breakdown(state),
        "adapter_bytes": serialized_state_bytes(state),
        "privacy_note": "FedAvg aggregation only. No formal DP at server unless client DP validated.",
    }
    (out_dir / "federated_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[server] global adapter saved -> {out_dir}")
    return meta


def aggregate_bundles(
    bundle_paths: List[Path],
    config: FederatedLoraConfig,
    global_dir: Path,
    *,
    diagnostics: bool = False,
) -> tuple[Dict[str, torch.Tensor], dict]:
    global_state = _load_global_state(config, global_dir)
    global_serialized_bytes = serialized_state_bytes(global_state)
    weighted_deltas: List[Tuple[int, Dict[str, torch.Tensor]]] = []
    n_params = trainable_param_count(global_state)
    param_breakdown = trainable_param_breakdown(global_state)
    diagnostic_records: List[dict] = []

    for path in bundle_paths:
        bundle = load_bundle(path)
        local_state = unpack_update(bundle)
        delta = state_dict_to_delta(local_state, global_state)
        delta = clip_delta(delta, config.clip_norm)
        weighted_deltas.append((int(bundle["n_samples"]), delta))
        if diagnostics:
            diagnostic_records.append(
                _client_update_diagnostics(bundle, local_state, global_state, delta)
            )

    merged_delta = fedavg_deltas(weighted_deltas, global_state)
    new_state = apply_delta(global_state, merged_delta, scale=1.0)
    adapter_bytes = serialized_state_bytes(new_state)

    bundles = [load_bundle(path) for path in bundle_paths]
    round_comm = measure_round_communication(
        bundles,
        global_state_serialized_bytes=global_serialized_bytes,
    )
    comm = {
        **round_comm,
        "trainable_parameters": n_params,
        "trainable_param_breakdown": param_breakdown,
        "adapter_bytes": adapter_bytes,
    }
    if diagnostics:
        comm["aggregation_diagnostics"] = {
            "global_trainable_state_norm": _state_norm(global_state),
            "aggregated_update_norm": _state_norm(merged_delta),
            "per_client": diagnostic_records,
        }
    return new_state, comm


def _state_norm(state: Dict[str, torch.Tensor]) -> float:
    import math

    return math.sqrt(sum(float((v * v).sum()) for v in state.values()))


def _client_update_diagnostics(
    bundle: dict,
    local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
) -> dict:
    import math

    def _subset_norm(state: Dict[str, torch.Tensor], needle: str) -> float:
        vals = [v for k, v in state.items() if needle in k.lower()]
        if not vals:
            return 0.0
        return math.sqrt(sum(float((v * v).sum()) for v in vals))

    return {
        "client_id": bundle.get("client_id"),
        "n_samples": int(bundle.get("n_samples", 0)),
        "update_norm": _state_norm(delta),
        "local_state_norm": _state_norm(local_state),
        "lora_a_norm": _subset_norm(delta, "lora_a"),
        "lora_b_norm": _subset_norm(delta, "lora_b"),
        "score_head_norm": _subset_norm(delta, "score"),
        "upload_bytes": int(bundle.get("serialized_update_bytes") or bundle.get("update_bytes") or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate federated LoRA client bundles.")
    parser.add_argument("--bundles", nargs="+", required=True)
    parser.add_argument("--global-adapter", required=True)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--prox-mu", type=float, default=None)
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--comm-report", default=None, help="Write communication JSON to this path")
    parser.add_argument(
        "--aggregation-diagnostics",
        action="store_true",
        help="Include per-client update norm diagnostics in comm report",
    )
    args = parser.parse_args()

    cfg = FederatedLoraConfig(clip_norm=args.clip_norm)
    if args.config_json:
        payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        for key, value in payload.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    if args.base_model:
        cfg.base_model = args.base_model
    cfg.algorithm = args.algorithm
    if args.prox_mu is not None:
        cfg.prox_mu = args.prox_mu

    global_dir = Path(args.global_adapter)
    paths = [Path(p) for p in args.bundles]
    new_state, comm = aggregate_bundles(
        paths,
        cfg,
        global_dir,
        diagnostics=bool(args.aggregation_diagnostics),
    )
    meta = _save_global_adapter(cfg, new_state, global_dir)

    if args.comm_report:
        comm_path = Path(args.comm_report)
        comm_path.parent.mkdir(parents=True, exist_ok=True)
        comm_path.write_text(json.dumps(comm, indent=2), encoding="utf-8")

    summary = {
        "n_clients": len(paths),
        "global_adapter": str(global_dir),
        "clip_norm": cfg.clip_norm,
        "algorithm": cfg.algorithm,
        "prox_mu": float(cfg.prox_mu) if cfg.algorithm == "fedprox" else 0.0,
        "communication": comm,
        "metadata": meta,
    }
    print("EDUGUARD_SERVER_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
