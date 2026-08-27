#!/usr/bin/env python
"""Resource gate for multi-task LoRA training.

Never starts training. Prints TRAINING NOT STARTED — INSUFFICIENT RESOURCES
when the machine cannot safely host the job.
"""
from __future__ import annotations

import argparse
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
    except ImportError:
        pass
    return info


def assess(model_id: str, resources: dict | None = None) -> dict:
    resources = resources or collect_resources()
    estimate = MODEL_ESTIMATES.get(model_id)
    if estimate is None:
        raise ValueError(f"Unknown model_id: {model_id}")
    ram_total = resources.get("ram_total_gb") or 0.0
    ram_avail = resources.get("ram_available_gb") or 0.0
    cuda = bool(resources.get("cuda_available"))
    vram = resources.get("gpu_vram_gb") or 0.0
    needed = estimate["estimated_gpu_lora_gb"] if cuda else estimate["estimated_cpu_lora_gb"]
    reasons: list[str] = []
    feasible = True
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
    if cuda and vram < estimate["estimated_gpu_lora_gb"] * 0.75:
        feasible = False
        reasons.append(
            f"GPU VRAM {vram} GB likely insufficient (est. {estimate['estimated_gpu_lora_gb']} GB)."
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
        "feasible": feasible,
        "reasons": reasons,
        "verdict": verdict,
        "memory_breakdown_est_gb": {
            "parameter_memory": estimate["fp32_weights_gb"],
            "optimizer_memory": estimate["optimizer_adamw_gb_est"],
            "activation_memory": estimate["activation_gb_est"],
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
    args = parser.parse_args()
    model_ids = args.model_ids or list(MODEL_ESTIMATES.keys())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    resources = collect_resources()
    reports = [assess(mid, resources) for mid in model_ids]
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resources": resources,
        "models": reports,
        "any_feasible": any(r["feasible"] for r in reports),
    }
    path = RESULTS_DIR / "resource_check.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (REPORTS_DIR / "resource_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if not out["any_feasible"]:
        print("TRAINING NOT STARTED — INSUFFICIENT RESOURCES")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
