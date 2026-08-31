#!/usr/bin/env python
"""Record environment versions for GPU reproducibility."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "artifacts" / "evaluation" / "environment_lock.json"
OUT_TXT = ROOT / "artifacts" / "evaluation" / "GPU_ENVIRONMENT.txt"


def collect_environment() -> dict:
    import subprocess

    env = {
        "format": "environment_lock_v1",
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
    }
    for pkg in (
        "torch",
        "transformers",
        "peft",
        "opacus",
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "datasets",
        "tokenizers",
        "sentence_transformers",
        "faiss",
        "llama_cpp",
    ):
        try:
            mod = __import__(pkg)
            env["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            env["packages"][pkg] = None

    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["torch_version"] = torch.__version__
        env["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
    except Exception as exc:
        env["torch_error"] = str(exc)

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL, text=True
        )
        env["git_revision"] = out.strip()
    except Exception:
        env["git_revision"] = "unknown"

    return env


def main() -> int:
    env = collect_environment()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(env, indent=2), encoding="utf-8")

    lines = [
        "EduGuard GPU Environment Lock",
        "==============================",
        f"Python: {env.get('python_version')}",
        f"Platform: {env.get('platform')}",
        f"Git: {env.get('git_revision')}",
        f"PyTorch: {env.get('torch_version', env['packages'].get('torch'))}",
        f"CUDA available: {env.get('cuda_available')}",
    ]
    if env.get("gpu_name"):
        lines.append(f"GPU: {env['gpu_name']} ({env.get('gpu_memory_gb')} GB)")
    lines.append("")
    lines.append("Packages:")
    for pkg, ver in sorted(env["packages"].items()):
        lines.append(f"  {pkg}: {ver}")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
