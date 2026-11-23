#!/usr/bin/env python
"""Resource gate for multi-task LoRA training.

Never starts training. Prints TRAINING NOT STARTED — INSUFFICIENT RESOURCES
when the machine cannot safely host the job.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from paths import REPORTS_DIR, RESULTS_DIR  # noqa: E402

MODEL_ESTIMATES = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "approx_params_b": 0.49,
        "fp16_weights_gb": 1.0,
        "fp32_weights_gb": 2.0,
        "optimizer_adamw_gb_est": 4.0,
        "activation_gb_est": 2.0,
        "estimated_cpu_lora_gb": 12.0,
        "estimated_gpu_lora_gb": 6.0,
        "min_available_ram_gb": 10.0,
        "hf_download_gb_est": 1.5,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "approx_params_b": 1.54,
        "fp16_weights_gb": 3.1,
        "fp32_weights_gb": 6.2,
        "optimizer_adamw_gb_est": 12.0,
        "activation_gb_est": 4.0,
        "estimated_cpu_lora_gb": 28.0,
        "estimated_gpu_lora_gb": 12.0,
        "min_available_ram_gb": 22.0,
        "hf_download_gb_est": 3.5,
    },
}


def collect_resources() -> dict:
    info = {
        "cpu_count": os.cpu_count(),
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "ram_total_gb": None,
        "ram_available_gb": None,
        "disk_free_gb": None,
        "torch_version": None,
        "psutil": False,
        "bitsandbytes_available": importlib.util.find_spec("bitsandbytes") is not None,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["psutil"] = True
        info["ram_total_gb"] = round(vm.total / 1024**3, 2)
        info["ram_available_gb"] = round(vm.available / 1024**3, 2)
        info["cpu_physical"] = psutil.cpu_count(logical=False)
        disk = psutil.disk_usage(str(EXPERIMENT_DIR))
        info["disk_free_gb"] = round(disk.free / 1024**3, 2)
        info["disk_total_gb"] = round(disk.total / 1024**3, 2)
    except ImportError:
        info["warning"] = "psutil missing; RAM/disk could not be measured precisely."
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            info["gpu_vram_free_gb"] = round(free_bytes / 1024**3, 2)
            info["gpu_vram_total_reported_gb"] = round(total_bytes / 1024**3, 2)
    except ImportError:
        pass
    return info


def assess(
    model_id: str,
    resources: dict | None = None,
    *,
    max_seq_length: int = 8192,
    load_in_4bit: bool = False,
) -> dict:
    resources = resources or collect_resources()
    estimate = MODEL_ESTIMATES.get(model_id)
    if estimate is None:
        raise ValueError(f"Unknown model_id: {model_id}")
    ram_total = resources.get("ram_total_gb") or 0.0
    ram_avail = resources.get("ram_available_gb") or 0.0
    cuda = bool(resources.get("cuda_available"))
    vram = resources.get("gpu_vram_gb") or 0.0
    parameter_gb = estimate["approx_params_b"] * (0.65 if load_in_4bit else 2.0)
    adapter_optimizer_gb = 1.0
    # Planning estimate only. The trainer also runs a real maximum-length
    # forward/backward preflight before optimization starts.
    seq_scale = max(1, max_seq_length) / 8192
    activation_gb = 10.0 * seq_scale
    reserve_gb = 2.5
    gpu_needed = parameter_gb + adapter_optimizer_gb + activation_gb + reserve_gb
    needed = gpu_needed if cuda else estimate["estimated_cpu_lora_gb"]
    reasons: list[str] = []
    feasible = True
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive.")
    if load_in_4bit and not cuda:
        feasible = False
        reasons.append("4-bit QLoRA requires a CUDA GPU; CUDA is unavailable.")
    if load_in_4bit and cuda and not resources.get("bitsandbytes_available", False):
        feasible = False
        reasons.append(
            "This config requires bitsandbytes for 4-bit QLoRA; install it in the active GPU environment."
        )
    if ram_total and ram_total < needed:
        feasible = False
        reasons.append(
            f"Total RAM {ram_total} GB < estimated training footprint {needed} GB."
        )
    if not cuda and ram_avail and ram_avail < estimate["min_available_ram_gb"]:
        feasible = False
        reasons.append(
            f"Available RAM {ram_avail} GB < {estimate['min_available_ram_gb']} GB minimum."
        )
    disk_free = resources.get("disk_free_gb")
    if disk_free is not None and disk_free < 12.0:
        feasible = False
        reasons.append(f"Free disk {disk_free} GB < 12 GB for checkpoints/HF cache.")
    if cuda:
        free_vram = resources.get("gpu_vram_free_gb")
        available_vram = float(free_vram if free_vram is not None else vram)
        if available_vram < needed:
            feasible = False
            reasons.append(
                f"Available GPU VRAM {available_vram:.2f} GB < planning "
                f"{needed:.2f} GB estimate for {max_seq_length} tokens "
                f"({'4-bit QLoRA' if load_in_4bit else 'FP16 LoRA'})."
            )
    if feasible and not reasons:
        reasons.append("Resource estimates indicate training may proceed.")
    verdict = (
        "TRAINING MAY PROCEED"
        if feasible
        else "TRAINING NOT STARTED — INSUFFICIENT RESOURCES"
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "resources": resources,
        "estimates": estimate,
        "estimated_footprint_gb": needed,
        "memory_estimate_inputs": {
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "parameter_memory_est_gb": round(parameter_gb, 2),
            "adapter_and_optimizer_memory_est_gb": adapter_optimizer_gb,
            "activation_memory_est_gb": round(activation_gb, 2),
            "safety_reserve_gb": reserve_gb,
            "estimated_gpu_footprint_gb": round(gpu_needed, 2),
            "footprint_basis": "GPU profile" if cuda else "CPU fallback profile",
            "estimate_is_not_a_guarantee": True,
            "training_runs_max_length_forward_backward_preflight": True,
        },
        "feasible": feasible,
        "reasons": reasons,
        "verdict": verdict,
        "memory_breakdown_est_gb": {
            "parameter_memory": round(
                parameter_gb if cuda else estimate["fp32_weights_gb"], 2
            ),
            "optimizer_memory": (
                adapter_optimizer_gb if cuda else estimate["optimizer_adamw_gb_est"]
            ),
            "activation_memory": round(
                activation_gb if cuda else estimate["activation_gb_est"], 2
            ),
            "other_and_safety_memory": round(
                reserve_gb
                if cuda
                else max(
                    0.0,
                    needed
                    - estimate["fp32_weights_gb"]
                    - estimate["optimizer_adamw_gb_est"]
                    - estimate["activation_gb_est"],
                ),
                2,
            ),
            "expected_training_footprint": needed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        default=None,
        help="Repeatable. Default: both Qwen sizes.",
    )
    parser.add_argument("--config", help="Training JSON config to assess its precision and sequence length.")
    parser.add_argument(
        "--defer-vram-check-to-preflight",
        action="store_true",
        help=(
            "Allow a conservative VRAM estimate to proceed only to the configured "
            "maximum-length backward preflight. The preflight remains mandatory."
        ),
    )
    args = parser.parse_args()
    config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (EXPERIMENT_DIR.parents[1] / config_path).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_ids = [str(config["model_id"])]
        profiles = [{
            "max_seq_length": int(config.get("max_seq_length", 8192)),
            "load_in_4bit": bool(config.get("load_in_4bit", False)),
        }]
    else:
        model_ids = args.model_ids or list(MODEL_ESTIMATES.keys())
        profiles = [{"max_seq_length": 8192, "load_in_4bit": False} for _ in model_ids]
    if args.model_ids and args.config:
        raise SystemExit("Use --config or --model-id, not both.")
    if args.defer_vram_check_to_preflight and not args.config:
        raise SystemExit("--defer-vram-check-to-preflight requires --config.")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    resources = collect_resources()
    reports = [assess(mid, resources, **profile) for mid, profile in zip(model_ids, profiles)]
    static_feasible = any(r["feasible"] for r in reports)
    preflight_only = False
    if (
        not static_feasible
        and args.defer_vram_check_to_preflight
        and config
        and bool(config.get("max_length_memory_preflight", False))
        and bool(resources.get("cuda_available"))
    ):
        preflight_only = all(
            r["reasons"]
            and all(reason.startswith("Available GPU VRAM ") for reason in r["reasons"])
            for r in reports
        )
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resources": resources,
        "models": reports,
        "any_feasible": static_feasible,
        "runtime_preflight_required": preflight_only,
        "training_gate_passed": static_feasible or preflight_only,
    }
    path = RESULTS_DIR / "resource_check.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (REPORTS_DIR / "resource_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if preflight_only:
        print(
            "STATIC VRAM ESTIMATE EXCEEDS AVAILABLE MEMORY — "
            "CONTINUING ONLY TO THE REQUIRED MAX-LENGTH BACKWARD PREFLIGHT"
        )
    if not out["training_gate_passed"]:
        print("TRAINING NOT STARTED — INSUFFICIENT RESOURCES")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
