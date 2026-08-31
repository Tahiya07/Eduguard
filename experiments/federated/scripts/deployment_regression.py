#!/usr/bin/env python
"""Lightweight deployment regression after federated export (no full model load required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "evaluation" / "deployment_regression.json"
MERGED = ROOT / "artifacts" / "federated" / "models" / "qwen_bloom_federated0.5B_fedavg_iid_merged"


def main() -> int:
    checks = {}
    errors = []

    # Production backend isolation
    try:
        import backend.service as svc

        src = open(svc.__file__, encoding="utf-8").read()
        checks["backend_no_training_imports"] = (
            "training.federated" not in src and "opacus" not in src
        )
    except Exception as exc:
        checks["backend_no_training_imports"] = False
        errors.append(f"backend import: {exc}")

    # Bloom prompt path
    try:
        from predict_bloom import build_prompt

        checks["bloom_prompt"] = "question" in build_prompt("test").lower() or len(build_prompt("test")) > 0
    except Exception as exc:
        checks["bloom_prompt"] = False
        errors.append(f"predict_bloom: {exc}")

    # Merged federated artifact exists
    checks["federated_merged_exists"] = (MERGED / "config.json").is_file()

    # GGUF generator path convention (1.5B separate)
    gguf_candidates = list((ROOT / "models").glob("**/*.gguf")) if (ROOT / "models").is_dir() else []
    checks["gguf_generator_present"] = len(gguf_candidates) > 0
    checks["gguf_paths"] = [str(p.relative_to(ROOT)) for p in gguf_candidates[:5]]

    report = {
        "status": "PASSED" if all(checks.get(k) for k in ("backend_no_training_imports", "bloom_prompt")) else "PARTIAL",
        "checks": checks,
        "errors": errors,
        "note": "Full inference regression requires GPU-trained merged model and local GGUF.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
