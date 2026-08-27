#!/usr/bin/env python
"""Merge a LoRA adapter into the HF base model, then convert to GGUF Q4_K_M.

Do NOT quantize before the HF/merged model has been evaluated, unless
--allow-unvalidated is passed. This script never trains GGUF.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def merge_adapter(adapter_dir: Path, merged_dir: Path, base_model: str | None) -> Path:
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = adapter_dir.resolve()
    merged_dir.mkdir(parents=True, exist_ok=True)
    cfg = PeftConfig.from_pretrained(str(adapter_dir))
    base = base_model or cfg.base_model_name_or_path
    print("Merging", adapter_dir, "into", base)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    meta = {
        "base_model": base,
        "adapter_dir": str(adapter_dir),
        "merged_utc": datetime.now(timezone.utc).isoformat(),
    }
    (merged_dir / "merge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return merged_dir


def convert_to_gguf(merged_dir: Path, gguf_dir: Path, quant: str) -> Path:
    gguf_dir.mkdir(parents=True, exist_ok=True)
    convert_script = shutil.which("convert_hf_to_gguf.py")
    llama_convert = None
    for candidate in (
        Path(convert_script) if convert_script else None,
        REPO_ROOT / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
    ):
        if candidate and candidate.is_file():
            llama_convert = candidate
            break
    if llama_convert is None:
        raise SystemExit(
            "llama.cpp convert_hf_to_gguf.py not found. Clone llama.cpp and pass "
            "--convert-script, or add it to PATH. Conversion was not performed."
        )
    f16_path = gguf_dir / "model-f16.gguf"
    q_path = gguf_dir / f"model-{quant}.gguf"
    python = sys.executable
    run([python, str(llama_convert), str(merged_dir), "--outfile", str(f16_path), "--outtype", "f16"])
    quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
    if not quantize_bin:
        raise SystemExit(
            f"Wrote F16 GGUF to {f16_path} but llama-quantize was not found. "
            "Install llama.cpp tools to produce Q4_K_M."
        )
    run([quantize_bin, str(f16_path), str(q_path), quant])
    meta = {
        "quant": quant,
        "f16": str(f16_path),
        "quantized": str(q_path),
        "bytes": q_path.stat().st_size if q_path.is_file() else None,
        "converted_utc": datetime.now(timezone.utc).isoformat(),
    }
    (gguf_dir / "gguf_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return q_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--gguf-dir", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="Permit quantization before HF evaluation. Not recommended.",
    )
    parser.add_argument("--hf-metrics", default=None, help="Path to HF evaluation metrics.json")
    args = parser.parse_args()

    if not args.allow_unvalidated:
        if not args.hf_metrics or not Path(args.hf_metrics).is_file():
            raise SystemExit(
                "Refusing to quantize before HF/merged evaluation. "
                "Pass --hf-metrics path/to/metrics.json or --allow-unvalidated."
            )

    merged_dir = Path(args.merged_dir)
    if not args.skip_merge:
        merge_adapter(Path(args.adapter_dir), merged_dir, args.base_model)
    if not args.skip_gguf:
        convert_to_gguf(merged_dir, Path(args.gguf_dir), args.quant)
    print("Done. Do not replace production GGUF from this experiment until selection is complete.")


if __name__ == "__main__":
    main()
