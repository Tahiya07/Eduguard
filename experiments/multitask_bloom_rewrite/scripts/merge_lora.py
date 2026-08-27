#!/usr/bin/env python
"""Merge LoRA adapter into a full HF checkpoint (does not overwrite base cache)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    adapter = Path(args.adapter)
    output = Path(args.output)
    if not adapter.is_absolute():
        adapter = REPO_ROOT / adapter
    if not output.is_absolute():
        output = REPO_ROOT / output
    if not adapter.exists():
        print(f"STOP: adapter missing: {adapter}")
        raise SystemExit(2)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print("STOP: missing merge dependencies:", exc)
        raise SystemExit(2) from exc

    print("Loading base:", args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float32
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print("Loading adapter:", adapter)
    model = PeftModel.from_pretrained(base, str(adapter))
    print("Merging...")
    merged = model.merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output))
    tok.save_pretrained(str(output))
    _ = AutoModelForCausalLM.from_pretrained(str(output), trust_remote_code=True)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(adapter),
        "output": str(output),
        "verified_reload": True,
        "note": "Original HF base weights in cache are preserved; merged copy is separate.",
    }
    (output / "merge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Merged checkpoint saved to", output)


if __name__ == "__main__":
    main()
