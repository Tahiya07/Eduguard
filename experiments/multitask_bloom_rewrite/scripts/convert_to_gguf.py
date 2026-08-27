#!/usr/bin/env python
"""Convert merged HF checkpoints to GGUF using the installed llama.cpp toolchain.

Inspects available conversion entry points; does not invent CLI syntax.
Does not overwrite models/qwen.gguf.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]


def find_convert_script() -> Path | None:
    candidates = [
        REPO_ROOT / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py",
        REPO_ROOT / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
    ]
    which = shutil.which("convert-hf-to-gguf") or shutil.which("convert_hf_to_gguf.py")
    if which:
        candidates.insert(0, Path(which))
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def find_quantize() -> Path | None:
    names = ["llama-quantize", "quantize", "llama.cpp-quantize"]
    for name in names:
        p = shutil.which(name)
        if p:
            return Path(p)
    for c in [
        REPO_ROOT / "third_party" / "llama.cpp" / "llama-quantize.exe",
        REPO_ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-quantize.exe",
        REPO_ROOT / "llama.cpp" / "build" / "bin" / "llama-quantize",
    ]:
        if c.exists():
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True, help="Merged HF model directory")
    parser.add_argument("--out-gguf", required=True, help="Output GGUF path (non-production)")
    parser.add_argument("--quantize", default="Q4_K_M")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    hf_model = Path(args.hf_model)
    out_gguf = Path(args.out_gguf)
    if not hf_model.is_absolute():
        hf_model = REPO_ROOT / hf_model
    if not out_gguf.is_absolute():
        out_gguf = REPO_ROOT / out_gguf
    if out_gguf.resolve() == (REPO_ROOT / "models" / "qwen.gguf").resolve():
        print("STOP: refusing to overwrite production models/qwen.gguf")
        raise SystemExit(2)
    if not hf_model.exists():
        print(f"STOP: HF model missing: {hf_model}")
        raise SystemExit(2)

    convert = find_convert_script()
    quantize = find_quantize()
    discovery = {
        "convert_script": str(convert) if convert else None,
        "quantize_bin": str(quantize) if quantize else None,
    }
    if convert is None:
        print("STOP: GGUF conversion unavailable — convert_hf_to_gguf.py not found")
        print(json.dumps(discovery, indent=2))
        raise SystemExit(2)

    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    f16_path = out_gguf.with_suffix(".f16.gguf")
    t0 = time.time()
    cmd = [args.python, str(convert), str(hf_model), "--outfile", str(f16_path), "--outtype", "f16"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    quant_type = args.quantize
    if quantize is None:
        print("WARNING: quantize binary not found; leaving F16 GGUF only.")
        final = f16_path
        quant_type = "F16"
    else:
        qcmd = [str(quantize), str(f16_path), str(out_gguf), quant_type]
        print("Running:", " ".join(qcmd))
        subprocess.run(qcmd, check=True)
        final = out_gguf
    elapsed = time.time() - t0
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(hf_model),
        "conversion_tool": str(convert),
        "quantize_tool": str(quantize) if quantize else None,
        "quantization_type": quant_type,
        "gguf_path": str(final),
        "gguf_size_bytes": final.stat().st_size if final.exists() else None,
        "conversion_time_sec": round(elapsed, 3),
        "discovery": discovery,
    }
    meta_path = final.with_suffix(final.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
