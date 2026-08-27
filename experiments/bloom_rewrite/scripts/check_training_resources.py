#!/usr/bin/env python
"""Resource feasibility check for Bloom rewrite LoRA training.

This script never starts training. The training script imports the same checks
and refuses to run when the machine cannot safely host the job.
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

from paths import RESULTS_DIR  # noqa: E402

# Conservative CPU LoRA estimates in GB. These include weights, AdamW states,
# gradients, activations, and tokenizer overhead. They are not GGUF sizes.
MODEL_ESTIMATES = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "approx_params_b": 0.49,
        "fp32_weights_gb": 2.0,
        "estimated_cpu_lora_gb": 12.0,
        "estimated_gpu_lora_gb": 6.0,
        "min_available_ram_gb": 10.0,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "approx_params_b": 1.54,
        "fp32_weights_gb": 6.2,
        "estimated_cpu_lora_gb": 28.0,
        "estimated_gpu_lora_gb": 12.0,
        "min_available_ram_gb": 22.0,
    },
}


def collect_resources() -> dict:
    info = {
        "cpu_threads": os.cpu_count(),
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "ram_total_gb": None,
        "ram_available_gb": None,
        "swap_total_gb": None,
        "torch_version": None,
        "psutil": False,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["psutil"] = True
        info["ram_total_gb"] = round(vm.total / 1024**3, 2)
        info["ram_available_gb"] = round(vm.available / 1024**3, 2)
        info["swap_total_gb"] = round(psutil.swap_memory().total / 1024**3, 2)
        info["cpu_physical"] = psutil.cpu_count(logical=False)
        try:
            from paths import REPO_ROOT as _ROOT

            disk = psutil.disk_usage(str(_ROOT))
            info["disk_total_gb"] = round(disk.total / 1024**3, 2)
            info["disk_free_gb"] = round(disk.free / 1024**3, 2)
        except Exception:
            info["disk_total_gb"] = None
            info["disk_free_gb"] = None
    except ImportError:
        info["warning"] = "psutil missing; RAM could not be measured."
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
    except ImportError:
        info["torch_version"] = None
    return info


def assess(model_id: str, resources: dict | None = None) -> dict:
    resources = resources or collect_resources()
    estimate = MODEL_ESTIMATES.get(model_id)
    if estimate is None:
        raise ValueError(f"Unknown model_id for resource check: {model_id}")
    ram_total = resources.get("ram_total_gb") or 0.0
    ram_avail = resources.get("ram_available_gb") or 0.0
    cuda = bool(resources.get("cuda_available"))
    vram = resources.get("gpu_vram_gb") or 0.0
    needed = estimate["estimated_gpu_lora_gb"] if cuda else estimate["estimated_cpu_lora_gb"]
    feasible = False
    reasons = []
    if ram_total and ram_total < needed:
        reasons.append(
            f"Total RAM {ram_total} GB is below estimated {needed} GB for {model_id} LoRA."
        )
    if ram_avail and ram_avail < estimate["min_available_ram_gb"] and not cuda:
        reasons.append(
            f"Available RAM {ram_avail} GB is below {estimate['min_available_ram_gb']} GB."
        )
    disk_free = resources.get("disk_free_gb")
    if disk_free is not None and disk_free < 8.0:
        reasons.append(f"Free disk {disk_free} GB is below 8 GB needed for checkpoints and HF weights.")
    if cuda and vram < estimate["estimated_gpu_lora_gb"] * 0.75:
        reasons.append(
            f"GPU VRAM {vram} GB is likely insufficient for {model_id} LoRA "
            f"(estimate {estimate['estimated_gpu_lora_gb']} GB)."
        )
    if not reasons:
        feasible = True
        reasons.append("Resource estimates indicate training may proceed.")
    return {
        "model_id": model_id,
        "feasible": feasible,
        "needed_gb": needed,
        "estimate": estimate,
        "resources": resources,
        "reasons": reasons,
        "verdict": (
            "TRAINING MAY PROCEED"
            if feasible
            else "TRAINING NOT STARTED — insufficient resources"
        ),
        "checked_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        default=None,
        help="Repeatable. Defaults to both Qwen 0.5B and 1.5B Instruct.",
    )
    args = parser.parse_args()
    model_ids = args.model_ids or list(MODEL_ESTIMATES)
    resources = collect_resources()
    reports = [assess(model_id, resources) for model_id in model_ids]
    payload = {"resources": resources, "models": reports}
    out = RESULTS_DIR / "resource_feasibility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if any(not item["feasible"] for item in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
