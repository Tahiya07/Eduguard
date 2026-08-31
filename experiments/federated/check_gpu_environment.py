#!/usr/bin/env python
"""GPU environment readiness check for EduGuard federated research."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.federated.run_integrity import DATASET_FILES, dataset_hashes, file_sha256, git_revision


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return round(usage.free / (1024**3), 2)


def check_environment() -> dict:
    import platform

    report: dict = {
        "format": "gpu_environment_check_v1",
        "ready": False,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": git_revision(),
        "checks": {},
        "blocking_issues": [],
        "warnings": [],
    }

    def add_check(name: str, passed: bool, details: dict, blocking: bool = True) -> None:
        report["checks"][name] = {"passed": passed, **details}
        if not passed and blocking:
            report["blocking_issues"].append(f"{name}: {details.get('message', 'failed')}")
        elif not passed:
            report["warnings"].append(f"{name}: {details.get('message', 'warning')}")

    # PyTorch / CUDA
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        details = {
            "torch_version": torch.__version__,
            "cuda_available": cuda,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        }
        if cuda:
            details["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            details["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
        add_check("pytorch_cuda", cuda, details, blocking=True)
    except Exception as exc:
        add_check("pytorch_cuda", False, {"message": str(exc)}, blocking=True)

    for pkg in ("transformers", "peft", "opacus"):
        try:
            mod = __import__(pkg)
            add_check(
                pkg,
                True,
                {"version": getattr(mod, "__version__", "unknown")},
                blocking=(pkg in {"transformers", "peft"}),
            )
        except ImportError as exc:
            add_check(pkg, False, {"message": str(exc)}, blocking=(pkg in {"transformers", "peft"}))

    # Datasets
    missing = [rel for rel in DATASET_FILES if not (ROOT / rel).is_file()]
    add_check(
        "datasets",
        not missing,
        {"missing": missing, "hashes": dataset_hashes()},
        blocking=True,
    )

    # Disk space (warn below 20 GB, block below 5 GB)
    free_gb = _disk_free_gb(ROOT)
    if free_gb < 5:
        add_check("disk_space", False, {"free_gb": free_gb, "message": "less than 5 GB free"}, blocking=True)
    elif free_gb < 20:
        add_check("disk_space", True, {"free_gb": free_gb, "message": "low disk (<20 GB)"}, blocking=False)
        report["warnings"].append(f"disk_space: only {free_gb} GB free")
    else:
        add_check("disk_space", True, {"free_gb": free_gb}, blocking=False)

    # Training package import
    try:
        import training.federated  # noqa: F401

        add_check("training_package", True, {}, blocking=True)
    except Exception as exc:
        add_check("training_package", False, {"message": str(exc)}, blocking=True)

    report["ready"] = len(report["blocking_issues"]) == 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Check GPU environment for federated research")
    parser.add_argument("--json-out", default=str(ROOT / "artifacts" / "federated" / "gpu_environment_check.json"))
    args = parser.parse_args()

    report = check_environment()
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("EduGuard GPU Environment Check")
    print("==============================")
    print(f"READY: {'YES' if report['ready'] else 'NO'}")
    for issue in report["blocking_issues"]:
        print(f"  BLOCKING: {issue}")
    for warn in report["warnings"]:
        print(f"  WARNING: {warn}")
    if report["checks"].get("pytorch_cuda", {}).get("cuda_available"):
        chk = report["checks"]["pytorch_cuda"]
        print(f"  GPU: {chk.get('gpu_name')} ({chk.get('gpu_memory_gb')} GB)")
    print(f"\nReport: {out}")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
