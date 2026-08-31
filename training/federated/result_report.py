"""Federated Bloom FL result report schema and builder.

All federated simulation experiments write JSON matching format
``federated_bloom_result_v1`` via build_federated_result_report().
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from training.federated.aggregation import is_trainable_key
from training.federated.config import FederatedLoraConfig
from training.federated.execution_stats import (
    get_privacy_accounting_steps,
    summarize_execution_actual,
)

RESULT_FORMAT = "federated_bloom_result_v1"

# Top-level keys that must exist in every machine-readable result artifact.
REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "format",
        "run_id",
        "experiment_id",
        "git_revision",
        "config_hash",
        "dataset_hashes",
        "model_identifier",
        "seed",
        "status",
        "start_time",
        "end_time",
        "training",
        "lora",
        "trainable_aggregation_state",
        "communication",
        "metrics",
        "runtime_seconds",
        "client_sample_counts",
        "partition",
    }
)

REQUIRED_TRAINING_KEYS = frozenset(
    {
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "gradient_accumulation_steps",
        "local_epochs",
        "federated_rounds",
        "client_count",
        "algorithm",
        "prox_mu",
        "configured",
        "actual",
    }
)

REQUIRED_TRAINING_CONFIGURED_KEYS = frozenset(
    {
        "local_epochs",
        "federated_rounds",
        "client_count",
        "optimizer_steps_per_client_per_round",
        "optimizer_steps_per_round_all_clients",
        "total_optimizer_steps_estimate",
    }
)

REQUIRED_TRAINING_ACTUAL_KEYS = frozenset(
    {
        "federated_rounds_configured",
        "federated_rounds_completed",
        "optimizer_steps_per_client_per_round",
        "optimizer_steps_per_round_all_clients",
        "total_optimizer_steps_completed",
        "execution_status",
        "privacy_accounting_steps_source",
        "privacy_accounting_steps",
    }
)

REQUIRED_METRICS_KEYS = frozenset(
    {
        "evaluation_status",
        "accuracy",
        "macro_f1",
        "per_class_f1",
    }
)

REQUIRED_COMMUNICATION_KEYS = frozenset(
    {
        "upload_bytes",
        "download_bytes",
        "total_bytes",
        "trainable_parameter_count",
    }
)

REQUIRED_AGGREGATION_STATE_KEYS = frozenset(
    {
        "includes",
        "excludes",
        "aggregation",
        "trainable_parameter_count",
    }
)


def build_training_configured_block(
    cfg: FederatedLoraConfig,
    client_sample_counts: Mapping[str, int],
) -> Dict[str, Any]:
    step_est = estimate_total_optimizer_steps(client_sample_counts, cfg)
    return {
        "local_epochs": float(cfg.local_epochs),
        "federated_rounds": int(cfg.rounds),
        "client_count": int(cfg.num_clients),
        **step_est,
    }


def build_training_actual_block(
    history: Optional[List[Dict[str, Any]]],
    *,
    configured_rounds: int,
) -> Dict[str, Any]:
    return summarize_execution_actual(history or [], configured_rounds=configured_rounds)


def estimate_optimizer_steps_per_client(n_samples: int, cfg: FederatedLoraConfig) -> int:
    """HF Trainer-equivalent optimizer steps for one client local training."""
    if n_samples <= 0:
        return 0
    effective_batch = max(1, int(cfg.batch_size) * int(cfg.grad_accum))
    steps_per_epoch = max(1, math.ceil(n_samples / effective_batch))
    return max(1, int(math.ceil(steps_per_epoch * float(cfg.local_epochs))))


def estimate_total_optimizer_steps(
    client_sample_counts: Mapping[str, int], cfg: FederatedLoraConfig
) -> Dict[str, Any]:
    per_client = {
        cid: estimate_optimizer_steps_per_client(int(n), cfg)
        for cid, n in client_sample_counts.items()
    }
    per_round = int(sum(per_client.values()))
    total = per_round * int(cfg.rounds)
    return {
        "optimizer_steps_per_client_per_round": per_client,
        "optimizer_steps_per_round_all_clients": per_round,
        "total_optimizer_steps_estimate": total,
    }


def flatten_per_class_f1(per_class: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not per_class:
        return {label: None for label in (
            "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
        )}
    out: Dict[str, Optional[float]] = {}
    for label, stats in per_class.items():
        if isinstance(stats, dict):
            out[label] = stats.get("f1")
        else:
            out[label] = None
    return out


def build_metrics_block(
  final_test_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not final_test_metrics:
        return {
            "evaluation_status": "NOT_EVALUATED",
            "accuracy": None,
            "macro_f1": None,
            "per_class_f1": flatten_per_class_f1(None),
            "n_eval": None,
        }
    per_class_raw = final_test_metrics.get("per_class") or final_test_metrics.get("per_class_f1")
    return {
        "evaluation_status": "EVALUATED",
        "accuracy": final_test_metrics.get("accuracy"),
        "macro_f1": final_test_metrics.get("macro_f1"),
        "per_class_f1": flatten_per_class_f1(per_class_raw if isinstance(per_class_raw, dict) else None),
        "n_eval": final_test_metrics.get("n_eval"),
        "quadratic_weighted_kappa": final_test_metrics.get("quadratic_weighted_kappa"),
        "within_one_level_accuracy": final_test_metrics.get("within_one_level_accuracy"),
        "ece": final_test_metrics.get("ece"),
    }


def build_trainable_aggregation_state(
    trainable_parameter_count: Optional[int],
) -> Dict[str, Any]:
    return {
        "includes": [
            "lora_adapter_parameters (lora_A, lora_B on target_modules)",
            "classification_score_head (modules_to_save=['score'])",
        ],
        "excludes": [
            "frozen_base_model_weights",
            "optimizer_state",
            "scheduler_state",
            "non_trainable_buffers",
        ],
        "selection_rule": "training.federated.aggregation.is_trainable_key",
        "aggregation": "fedavg_sample_weighted",
        "trainable_parameter_count": trainable_parameter_count,
    }


def build_federated_result_report(
    *,
    cfg: FederatedLoraConfig,
    run_id: str,
    experiment_id: str,
    git_revision: str,
    config_hash: str,
    dataset_hashes: Dict[str, Optional[str]],
    client_sample_counts: Dict[str, int],
    partition_strategy: str,
    dirichlet_alpha: Optional[float],
    setting_tag: str,
    global_adapter: str,
    results_json_path: str,
    round_checkpoint_path: str,
    communication: Dict[str, Any],
    final_test_metrics: Optional[Dict[str, Any]],
    runtime_seconds: float,
    start_time: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    status: str = "EXECUTED",
) -> Dict[str, Any]:
    end_time = datetime.now(timezone.utc).isoformat()
    configured = build_training_configured_block(cfg, client_sample_counts)
    actual = build_training_actual_block(history, configured_rounds=int(cfg.rounds))
    upload = int(communication.get("total_upload_bytes") or communication.get("upload_bytes") or 0)
    download = int(communication.get("total_download_bytes") or communication.get("download_bytes") or 0)
    trainable_params = communication.get("trainable_parameters") or communication.get(
        "trainable_parameter_count"
    )

    partition_block: Dict[str, Any] = {
        "strategy": partition_strategy,
        "dirichlet_alpha": dirichlet_alpha,
    }

    report: Dict[str, Any] = {
        "format": RESULT_FORMAT,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "git_revision": git_revision,
        "config_hash": config_hash,
        "dataset_hashes": dataset_hashes,
        "model_identifier": cfg.base_model,
        "seed": int(cfg.seed),
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "setting_tag": setting_tag,
        "training": {
            "optimizer": "AdamW",
            "learning_rate": float(cfg.learning_rate),
            "finetune_learning_rate": float(cfg.finetune_learning_rate),
            "weight_decay": float(cfg.weight_decay),
            "lr_scheduler_type": cfg.lr_scheduler_type,
            "warmup_ratio": float(cfg.warmup_ratio),
            "batch_size": int(cfg.batch_size),
            "gradient_accumulation_steps": int(cfg.grad_accum),
            "max_grad_norm": float(cfg.max_grad_norm),
            "local_epochs": float(cfg.local_epochs),
            "federated_rounds": int(cfg.rounds),
            "client_count": int(cfg.num_clients),
            "algorithm": cfg.algorithm,
            "prox_mu": float(cfg.prox_mu) if cfg.algorithm == "fedprox" else 0.0,
            "label_smoothing": float(cfg.label_smoothing),
            "use_class_weights": bool(cfg.use_class_weights),
            "configured": configured,
            "actual": actual,
            # Legacy estimate fields retained at training root for reproducibility.
            "optimizer_steps_per_client_per_round": configured[
                "optimizer_steps_per_client_per_round"
            ],
            "optimizer_steps_per_round_all_clients": configured[
                "optimizer_steps_per_round_all_clients"
            ],
            "total_optimizer_steps_estimate": configured["total_optimizer_steps_estimate"],
        },
        "lora": cfg.lora_config_dict(),
        "trainable_aggregation_state": build_trainable_aggregation_state(
            int(trainable_params) if trainable_params is not None else None
        ),
        "client_sample_counts": {k: int(v) for k, v in client_sample_counts.items()},
        "partition": partition_block,
        "communication": {
            "upload_bytes": upload,
            "download_bytes": download,
            "total_bytes": upload + download,
            "trainable_parameter_count": trainable_params,
            "adapter_size_bytes": communication.get("adapter_bytes"),
            "adapter_size_mb": communication.get("adapter_size_mb"),
            "per_round_upload_bytes": communication.get("per_round_upload_bytes", []),
            "per_round_download_bytes": communication.get("per_round_download_bytes", []),
        },
        "metrics": build_metrics_block(final_test_metrics),
        "runtime_seconds": round(float(runtime_seconds), 4),
        "history": history or [],
        "artifact_paths": {
            "global_adapter": global_adapter,
            "round_checkpoint": round_checkpoint_path,
            "results_json": results_json_path,
            "train_csv": cfg.train_csv,
            "test_csv": cfg.test_csv,
        },
        "trainable_key_predicate": {
            "function": "training.federated.aggregation.is_trainable_key",
            "matches_lora": True,
            "matches_score_head": True,
            "example_keys": [
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default",
                "base_model.score.weight",
            ],
        },
    }
    return report


def validate_result_report(report: Dict[str, Any]) -> List[str]:
    """Return list of validation errors (empty if valid)."""
    errors: List[str] = []
    missing_top = REQUIRED_TOP_LEVEL_KEYS - set(report.keys())
    if missing_top:
        errors.append(f"missing top-level keys: {sorted(missing_top)}")

    training = report.get("training") or {}
    missing_train = REQUIRED_TRAINING_KEYS - set(training.keys())
    if missing_train:
        errors.append(f"missing training keys: {sorted(missing_train)}")

    configured = training.get("configured") or {}
    missing_cfg = REQUIRED_TRAINING_CONFIGURED_KEYS - set(configured.keys())
    if missing_cfg:
        errors.append(f"missing training.configured keys: {sorted(missing_cfg)}")

    actual = training.get("actual") or {}
    missing_actual = REQUIRED_TRAINING_ACTUAL_KEYS - set(actual.keys())
    if missing_actual:
        errors.append(f"missing training.actual keys: {sorted(missing_actual)}")

    if configured.get("total_optimizer_steps_estimate") != training.get(
        "total_optimizer_steps_estimate"
    ):
        errors.append(
            "training.total_optimizer_steps_estimate must mirror training.configured"
        )

    metrics = report.get("metrics") or {}
    missing_metrics = REQUIRED_METRICS_KEYS - set(metrics.keys())
    if missing_metrics:
        errors.append(f"missing metrics keys: {sorted(missing_metrics)}")

    comm = report.get("communication") or {}
    missing_comm = REQUIRED_COMMUNICATION_KEYS - set(comm.keys())
    if missing_comm:
        errors.append(f"missing communication keys: {sorted(missing_comm)}")

    agg = report.get("trainable_aggregation_state") or {}
    missing_agg = REQUIRED_AGGREGATION_STATE_KEYS - set(agg.keys())
    if missing_agg:
        errors.append(f"missing trainable_aggregation_state keys: {sorted(missing_agg)}")

    if report.get("format") != RESULT_FORMAT:
        errors.append(f"format must be {RESULT_FORMAT!r}")

    return errors


def describe_trainable_keys(state_keys: List[str]) -> Dict[str, Any]:
    lora_keys = [k for k in state_keys if is_trainable_key(k) and "lora_" in k.lower()]
    score_keys = [k for k in state_keys if is_trainable_key(k) and "score" in k.lower()]
    frozen_leaked = [k for k in state_keys if not is_trainable_key(k)]
    return {
        "lora_key_count": len(lora_keys),
        "score_head_key_count": len(score_keys),
        "non_trainable_leaked_count": len(frozen_leaked),
    }


def write_schema_sample(
    output_path: str | Path,
    *,
    clients: int = 2,
    rounds: int = 1,
    partition: str = "iid",
    seed: int = 42,
) -> Dict[str, Any]:
    """Write a structural sample result (no training, no fabricated metrics)."""
    from pathlib import Path as P

    from experiments.federated.run_integrity import config_hash as ch
    from experiments.federated.run_integrity import dataset_hashes as dh
    from experiments.federated.run_integrity import git_revision as gr
    from training.federated.partition import partition_csv
    from training.paths import ROOT

    cfg = FederatedLoraConfig(num_clients=clients, rounds=rounds, local_epochs=0.05, seed=seed, partition=partition)
    parts = partition_csv(
        cfg.train_csv,
        num_clients=cfg.num_clients,
        text_col=cfg.text_col,
        label_col=cfg.label_col,
        strategy=cfg.partition,
        dirichlet_alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
    )
    client_sizes = {cid: int(len(frame)) for cid, frame in parts.items()}
    tag = f"schema_sample_{partition}"
    out = P(output_path)
    report = build_federated_result_report(
        cfg=cfg,
        run_id="schema_sample",
        experiment_id="schema_sample",
        git_revision=gr(ROOT),
        config_hash=ch(cfg.to_dict()),
        dataset_hashes=dh(ROOT),
        client_sample_counts=client_sizes,
        partition_strategy=cfg.partition,
        dirichlet_alpha=float(cfg.dirichlet_alpha) if cfg.partition == "non_iid_label" else None,
        setting_tag=tag,
        global_adapter=str(ROOT / "artifacts" / "federated" / "models" / tag),
        results_json_path=str(out),
        round_checkpoint_path=str(ROOT / "artifacts" / "federated" / "runs" / tag / "round_checkpoint.json"),
        communication={
            "total_upload_bytes": 0,
            "total_download_bytes": 0,
            "trainable_parameters": None,
            "adapter_bytes": None,
            "per_round_upload_bytes": [],
            "per_round_download_bytes": [],
        },
        final_test_metrics=None,
        runtime_seconds=0.0,
        start_time=datetime.now(timezone.utc).isoformat(),
        history=[],
        status="STRUCTURE_ONLY",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Federated result report utilities")
    parser.add_argument(
        "--write-schema-sample",
        default=None,
        help="Write structural sample JSON to this path (no training, metrics null)",
    )
    parser.add_argument("--clients", type=int, default=2)
    args = parser.parse_args()
    if args.write_schema_sample:
        report = write_schema_sample(args.write_schema_sample, clients=args.clients)
        errors = validate_result_report(report)
        if errors:
            raise SystemExit(f"schema validation failed: {errors}")
        print(json.dumps({"written": args.write_schema_sample, "format": report["format"]}, indent=2))
    else:
        parser.print_help()
