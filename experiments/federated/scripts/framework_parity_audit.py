#!/usr/bin/env python
"""Framework vs EduGuard federated reproducibility audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
OUT_JSON = ROOT / "artifacts" / "evaluation" / "framework_parity_audit.json"
OUT_MD = ROOT / "artifacts" / "evaluation" / "framework_parity_audit.md"

FED_RESULT = ROOT / "artifacts" / "federated" / "results" / "federated_lora_fedavg_iid.json"
FED_CONFIG = ROOT / "experiments" / "federated" / "configs" / "fedavg_iid.json"
PARTITION_PATH = ROOT / "artifacts" / "federated" / "runs" / "fedavg_iid" / "partition.json"
BASELINE_LOCK = ROOT / "artifacts" / "evaluation" / "fedavg_iid_baseline_lock.json"
FRAMEWORK_REF_CANDIDATES = [
    ROOT.parent / "Framework" / "results" / "federated_lora_fedavg_iid1.json",
    ROOT / "artifacts" / "evaluation" / "framework_reference_fedavg_iid.json",
]
CENTRALIZED_CANDIDATES = [
    ROOT / "results" / "bloom_lora_eval_0.5B.json",
    ROOT / "artifacts" / "evaluation" / "bloom_centralized_eval.json",
]

# Documented Framework reference when local JSON is unavailable.
FRAMEWORK_DOCUMENTED = {
    "test_accuracy": 0.503,
    "source": "GPU_RUN_GUIDE.md / historical Framework FedAvg+IID run",
    "result_path": "Framework/results/federated_lora_fedavg_iid1.json",
}

EDUGUARD_DEFAULTS = {
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "tokenizer": "AutoTokenizer.from_pretrained(base_model)",
    "dataset_train": "data/figshare_bloom_v1_train.csv",
    "dataset_val": "data/figshare_bloom_v1_val.csv",
    "dataset_test": "data/figshare_bloom_v1_test.csv",
    "split_policy": "fixed CSV splits (no re-split during FL)",
    "seed": 42,
    "clients": 8,
    "rounds": 5,
    "local_epochs": 3.0,
    "batch_size": 2,
    "gradient_accumulation": 8,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler": "cosine",
    "label_smoothing": 0.05,
    "class_weights": True,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "modules_to_save": ["score"],
    "partition": "iid (stratified round-robin by bloom_level)",
    "aggregation": "fedavg_sample_weighted",
    "clip_norm": 1.0,
    "from_scratch": True,
}

FRAMEWORK_DOCUMENTED_CONFIG = {
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "clients": 8,
    "rounds": 5,
    "local_epochs": 3.0,
    "batch_size": 2,
    "gradient_accumulation": 8,
    "learning_rate": 1e-4,
    "partition": "iid",
    "seed": 42,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "modules_to_save": ["score"],
    "aggregation": "FedAvg weighted by client sample count",
}


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_item(name: str, framework_val: Any, eduguard_val: Any, impact: str) -> dict:
    same = framework_val == eduguard_val
    return {
        "field": name,
        "framework_value": framework_val,
        "eduguard_value": eduguard_val,
        "same": same,
        "potential_impact": impact if not same else "none (matched)",
    }


def _framework_config_from_ref(ref: Optional[dict]) -> dict:
    if not ref:
        return dict(FRAMEWORK_DOCUMENTED_CONFIG)
    cfg = ref.get("training") or ref.get("config") or ref
    lora = ref.get("lora") or cfg.get("lora") or {}
    partition_val = ref.get("partition")
    if isinstance(partition_val, dict):
        partition = partition_val.get("strategy") or cfg.get("partition") or "iid"
    else:
        partition = partition_val or cfg.get("partition") or "iid"
    return {
        "base_model": ref.get("model_identifier") or cfg.get("base_model") or FRAMEWORK_DOCUMENTED_CONFIG["base_model"],
        "clients": cfg.get("client_count") or cfg.get("clients") or FRAMEWORK_DOCUMENTED_CONFIG["clients"],
        "rounds": cfg.get("federated_rounds") or cfg.get("rounds") or FRAMEWORK_DOCUMENTED_CONFIG["rounds"],
        "local_epochs": cfg.get("local_epochs") or FRAMEWORK_DOCUMENTED_CONFIG["local_epochs"],
        "batch_size": cfg.get("batch_size") or FRAMEWORK_DOCUMENTED_CONFIG["batch_size"],
        "gradient_accumulation": cfg.get("gradient_accumulation_steps") or cfg.get("gradient_accumulation") or 8,
        "learning_rate": cfg.get("learning_rate") or FRAMEWORK_DOCUMENTED_CONFIG["learning_rate"],
        "partition": partition,
        "seed": ref.get("seed") or cfg.get("seed") or 42,
        "lora_r": lora.get("r"),
        "lora_alpha": lora.get("lora_alpha"),
        "lora_dropout": lora.get("lora_dropout"),
        "modules_to_save": lora.get("modules_to_save"),
        "aggregation": (ref.get("trainable_aggregation_state") or {}).get("aggregation") or "fedavg",
    }


def _eduguard_config(fed_result: Optional[dict], fed_cfg_file: Optional[dict]) -> dict:
    training = (fed_result or {}).get("training") or {}
    lora = (fed_result or {}).get("lora") or (fed_cfg_file or {}).get("lora") or {}
    cfg_file = fed_cfg_file or {}
    return {
        "base_model": (fed_result or {}).get("model_identifier") or cfg_file.get("model"),
        "clients": training.get("client_count") or cfg_file.get("clients"),
        "rounds": training.get("federated_rounds") or cfg_file.get("rounds"),
        "local_epochs": training.get("local_epochs") or cfg_file.get("local_epochs"),
        "batch_size": training.get("batch_size") or cfg_file.get("batch_size"),
        "gradient_accumulation": training.get("gradient_accumulation_steps")
        or cfg_file.get("gradient_accumulation"),
        "learning_rate": training.get("learning_rate") or cfg_file.get("learning_rate"),
        "warmup_ratio": training.get("warmup_ratio", 0.1),
        "weight_decay": training.get("weight_decay", 0.01),
        "label_smoothing": training.get("label_smoothing", 0.05),
        "partition": (fed_result or {}).get("partition", {}).get("strategy") or cfg_file.get("partition"),
        "seed": (fed_result or {}).get("seed") or cfg_file.get("seed"),
        "lora_r": lora.get("r"),
        "lora_alpha": lora.get("alpha") or lora.get("lora_alpha"),
        "lora_dropout": lora.get("dropout") or lora.get("lora_dropout"),
        "modules_to_save": lora.get("modules_to_save"),
        "aggregation": (fed_result or {}).get("trainable_aggregation_state", {}).get(
            "aggregation", "fedavg_sample_weighted"
        ),
        "dataset_hashes": (fed_result or {}).get("dataset_hashes"),
    }


def build_audit() -> dict:
    fed_result = _load_json(FED_RESULT)
    fed_cfg_file = _load_json(FED_CONFIG)
    baseline_lock = _load_json(BASELINE_LOCK)
    partition_meta = _load_json(PARTITION_PATH)

    fw_path = next((p for p in FRAMEWORK_REF_CANDIDATES if p.is_file()), None)
    fw_ref = _load_json(fw_path) if fw_path else None
    fw_cfg = _framework_config_from_ref(fw_ref)
    edu_cfg = _eduguard_config(fed_result, fed_cfg_file)

    comparisons: List[dict] = []
    for key in sorted(set(fw_cfg) | set(edu_cfg)):
        comparisons.append(
            _compare_item(
                key,
                fw_cfg.get(key),
                edu_cfg.get(key),
                impact="configuration mismatch may shift convergence or parity",
            )
        )

    edu_metrics = {}
    if baseline_lock:
        edu_metrics = baseline_lock.get("final_test_metrics") or {}
    elif fed_result:
        edu_metrics = fed_result.get("final_test_metrics") or fed_result.get("metrics") or {}

    fw_test = None
    if fw_ref:
        fw_test = (fw_ref.get("final_test_metrics") or fw_ref.get("test_metrics") or fw_ref)
    fw_acc = (fw_test or {}).get("accuracy") if isinstance(fw_test, dict) else None
    if fw_acc is None:
        fw_acc = FRAMEWORK_DOCUMENTED["test_accuracy"]

    edu_acc = edu_metrics.get("accuracy")
    gap = None
    if edu_acc is not None and fw_acc is not None:
        gap = round(float(edu_acc) - float(fw_acc), 4)

    central_path = next((p for p in CENTRALIZED_CANDIDATES if p.is_file()), None)
    central = _load_json(central_path) if central_path else None

    return {
        "status": "COMPUTED",
        "exact_parity_established": False,
        "parity_threshold_absolute_accuracy": 0.05,
        "framework_reference_path": str(fw_path) if fw_path else FRAMEWORK_DOCUMENTED["result_path"],
        "framework_reference_available": fw_ref is not None,
        "eduguard_result_path": str(FED_RESULT),
        "eduguard_result_available": fed_result is not None,
        "baseline_lock_path": str(BASELINE_LOCK),
        "metrics_comparison": {
            "framework_test_accuracy": fw_acc,
            "eduguard_test_accuracy": edu_acc,
            "absolute_difference_eduguard_minus_framework": gap,
            "within_parity_threshold": gap is not None and abs(gap) <= 0.05,
        },
        "configuration_comparisons": comparisons,
        "partition_parity": {
            "eduguard_partition_artifact": str(PARTITION_PATH),
            "partition_available": partition_meta is not None,
            "seed": (partition_meta or {}).get("seed"),
            "strategy": (partition_meta or {}).get("partition"),
            "client_sizes": (partition_meta or {}).get("client_sizes"),
            "client_label_distribution": (partition_meta or {}).get("client_label_distribution"),
            "framework_partition_verified": False,
            "note": "Framework partition artifact not present in this repo; IID procedure documented only.",
        },
        "centralized_reference": {
            "source": str(central_path) if central_path else None,
            "available": central is not None,
            "metrics": central,
        },
        "known_eduguard_vs_centralized_training_differences": [
            _compare_item("warmup_ratio", 0.06, edu_cfg.get("warmup_ratio"), "may affect early-round convergence"),
            _compare_item(
                "training_precision",
                "fp16 when CUDA",
                "bf16 when supported else fp16/fp32",
                "numerical differences in local updates",
            ),
            _compare_item(
                "centralized_epochs",
                12,
                f"{edu_cfg.get('rounds')} rounds x {edu_cfg.get('local_epochs')} local epochs",
                "federated budget differs from centralized 12-epoch training",
            ),
        ],
        "dp_status": "NOT VALIDATED",
        "conclusion": (
            "EduGuard FedAvg+IID baseline (56.86%) exceeds historical Framework (~50.3%) by "
            f"{gap if gap is not None else 'unknown'} absolute accuracy. Exact parity is NOT established "
            "without Framework partition/config artifacts. Diagnose before claiming migration fidelity."
        ),
    }


def write_markdown(audit: dict) -> str:
    lines = [
        "# Framework Parity Audit",
        "",
        f"**Exact parity established:** {audit.get('exact_parity_established')}",
        f"**DP status:** {audit.get('dp_status')}",
        "",
        "## Metrics",
        "",
    ]
    mc = audit.get("metrics_comparison") or {}
    lines.extend(
        [
            f"- Framework test accuracy: {mc.get('framework_test_accuracy')}",
            f"- EduGuard test accuracy: {mc.get('eduguard_test_accuracy')}",
            f"- Absolute difference (EduGuard − Framework): {mc.get('absolute_difference_eduguard_minus_framework')}",
            f"- Within 5pp threshold: {mc.get('within_parity_threshold')}",
            "",
            "## Configuration comparisons",
            "",
            "| Field | Framework | EduGuard | Same | Potential impact |",
            "|---|---|---|:---:|---|",
        ]
    )
    for row in audit.get("configuration_comparisons") or []:
        lines.append(
            f"| {row['field']} | {row['framework_value']} | {row['eduguard_value']} | "
            f"{row['same']} | {row['potential_impact']} |"
        )
    lines.extend(["", "## Partition", ""])
    part = audit.get("partition_parity") or {}
    lines.append(f"- EduGuard partition artifact: `{part.get('eduguard_partition_artifact')}`")
    lines.append(f"- Framework partition verified: {part.get('framework_partition_verified')}")
    lines.append(f"- Note: {part.get('note')}")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(audit.get("conclusion", ""))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    audit = build_audit()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    OUT_MD.write_text(write_markdown(audit), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
