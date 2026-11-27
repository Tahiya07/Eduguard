#!/usr/bin/env python
"""Convert a merged HF checkpoint to GGUF and quantize it (default Q4_K_M), then verify the file.

Steps
  1. HF -> GGUF (F16) with llama.cpp's convert_hf_to_gguf.py (needs a llama.cpp source checkout;
     its bundled gguf-py is used, so no extra pip installs are required).
  2. F16 -> Q4_K_M with the llama-quantize binary if present, otherwise with the
     llama-cpp-python binding (llama_model_quantize) -- the same library that serves the app,
     so the result is guaranteed to load in EduGuard.
  3. Verify: header metadata (architecture, quantization type, chat template), load with
     llama-cpp-python, check its tokenizer against the HF tokenizer, and run one greedy smoke prompt.

Never overwrites models/qwen.gguf; refuses to overwrite any existing output unless --force.

Example
  python experiments/multitask_bloom_rewrite/scripts/convert_to_gguf.py \
      --hf-model experiments/multitask_bloom_rewrite/models/qwen15b_multitask_eduguard_v3_9_20261127/merged \
      --out-gguf models/gguf/qwen15b_multitask_eduguard_v3_9_q4_k_m.gguf \
      --llama-cpp-dir third_party/llama.cpp
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
PRODUCTION_GGUF = REPO_ROOT / "models" / "qwen.gguf"

FTYPES = {"Q4_K_M": "LLAMA_FTYPE_MOSTLY_Q4_K_M", "Q4_K_S": "LLAMA_FTYPE_MOSTLY_Q4_K_S",
          "Q5_K_M": "LLAMA_FTYPE_MOSTLY_Q5_K_M", "Q6_K": "LLAMA_FTYPE_MOSTLY_Q6_K",
          "Q8_0": "LLAMA_FTYPE_MOSTLY_Q8_0"}
GGUF_FILE_TYPES = {0: "F32", 1: "F16", 7: "Q8_0", 15: "Q4_K_M", 14: "Q4_K_S", 17: "Q5_K_M", 18: "Q6_K", 32: "BF16"}

CLONE_HINT = (
    "Get the converter once (source only, no build needed):\n"
    "  git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp\n"
    "then pass --llama-cpp-dir third_party/llama.cpp (or set LLAMA_CPP_DIR)."
)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def find_convert_script(llama_cpp_dir: str | None) -> Path | None:
    roots = [llama_cpp_dir, os.environ.get("LLAMA_CPP_DIR"), REPO_ROOT / "third_party" / "llama.cpp",
             REPO_ROOT / "llama.cpp", Path.home() / "llama.cpp"]
    for root in roots:
        if root and (resolve(root) / "convert_hf_to_gguf.py").is_file():
            return resolve(root) / "convert_hf_to_gguf.py"
    which = shutil.which("convert_hf_to_gguf.py")
    return Path(which) if which else None


def find_quantize_binary(llama_cpp_dir: str | None) -> Path | None:
    for name in ("llama-quantize", "llama-quantize.exe"):
        if shutil.which(name):
            return Path(shutil.which(name))
    roots = [llama_cpp_dir, os.environ.get("LLAMA_CPP_DIR"), REPO_ROOT / "third_party" / "llama.cpp"]
    for root in roots:
        if not root:
            continue
        for rel in ("build/bin/Release/llama-quantize.exe", "build/bin/llama-quantize.exe",
                    "build/bin/llama-quantize", "llama-quantize.exe"):
            if (resolve(root) / rel).is_file():
                return resolve(root) / rel
    return None


def quantize_with_binding(src: Path, dst: Path, quant: str, threads: int) -> None:
    import llama_cpp

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp, FTYPES[quant])
    params.nthread = threads
    rc = llama_cpp.llama_model_quantize(str(src).encode(), str(dst).encode(), ctypes.byref(params))
    if rc != 0:
        raise RuntimeError(f"llama_model_quantize returned {rc}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_header(path: Path) -> dict:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))

    def field(key: str):
        f = reader.fields.get(key)
        if f is None:
            return None
        part = f.parts[f.data[0]]
        if part.dtype.kind == "u" and part.dtype.itemsize == 1:
            return bytes(part).decode("utf-8", "replace")
        return part.tolist()[0]

    ftype = field("general.file_type")
    return {
        "architecture": field("general.architecture"),
        "name": field("general.name"),
        "file_type": GGUF_FILE_TYPES.get(ftype, ftype),
        "context_length": field("qwen2.context_length"),
        "has_chat_template": bool(field("tokenizer.chat_template")),
        "eos_token_id": field("tokenizer.ggml.eos_token_id"),
    }


def verify(gguf_path: Path, hf_model: Path, expected_quant: str, threads: int) -> dict:
    """Load the exported file the way the app does and check it behaves like the HF model's tokenizer/template."""
    sys.path.insert(0, str(EXPERIMENT_DIR))
    from llama_cpp import Llama
    from transformers import AutoTokenizer

    from prompts import bloom_messages, render_chatml

    header = read_header(gguf_path)
    problems = []
    if header["architecture"] != "qwen2":
        problems.append(f"architecture {header['architecture']!r} != 'qwen2'")
    if header["file_type"] != expected_quant:
        problems.append(f"file_type {header['file_type']!r} != {expected_quant!r}")
    if not header["has_chat_template"]:
        problems.append("no tokenizer.chat_template in GGUF")

    tok = AutoTokenizer.from_pretrained(str(hf_model), trust_remote_code=True)
    prompt = render_chatml(bloom_messages("List the stages of mitosis.", "Analyze"), add_generation_prompt=True)
    llm = Llama(model_path=str(gguf_path), n_ctx=2048, n_threads=threads, n_gpu_layers=0, verbose=False)
    hf_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    llama_ids = list(llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
    parity = hf_ids == llama_ids
    if not parity:
        problems.append("llama.cpp tokenization differs from the HF tokenizer on the smoke prompt")
    if header["eos_token_id"] is not None and header["eos_token_id"] != tok.convert_tokens_to_ids("<|im_end|>") \
            and header["eos_token_id"] != tok.eos_token_id:
        problems.append(f"GGUF eos id {header['eos_token_id']} matches neither <|im_end|> nor the HF eos id")
    t0 = time.perf_counter()
    out = llm(prompt, max_tokens=48, temperature=0.0, top_k=1, top_p=1.0, repeat_penalty=1.0,
              stop=["<|im_end|>", "<|im_start|>"])
    smoke = out["choices"][0]["text"].strip()
    if not smoke:
        problems.append("empty smoke generation")
    return {"header": header, "tokenizer_parity": parity, "smoke_prompt": "List the stages of mitosis. -> Analyze",
            "smoke_output": smoke, "smoke_seconds": round(time.perf_counter() - t0, 2),
            "ok": not problems, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-model", required=True, help="Merged HF model directory (from merge_lora.py)")
    parser.add_argument("--out-gguf", required=True, help="Output GGUF path, e.g. models/gguf/<name>_q4_k_m.gguf")
    parser.add_argument("--quantize", default="Q4_K_M", choices=sorted(FTYPES) + ["F16"])
    parser.add_argument("--outtype", default="f16", choices=["f16", "bf16", "f32"],
                        help="Intermediate precision before quantization (f16 keeps more mantissa bits than bf16)")
    parser.add_argument("--llama-cpp-dir", default=None, help="llama.cpp source checkout containing convert_hf_to_gguf.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep the F16 GGUF after quantizing")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    hf_model, out_gguf = resolve(args.hf_model), resolve(args.out_gguf)
    if out_gguf.resolve() == PRODUCTION_GGUF.resolve():
        raise SystemExit("STOP: refusing to overwrite production models/qwen.gguf; write elsewhere and switch "
                         "GENERATOR_MODEL_PATH after evaluation.")
    if not (hf_model / "config.json").is_file():
        raise SystemExit(f"STOP: merged HF model missing (no config.json): {hf_model}")
    if out_gguf.exists() and not args.force:
        raise SystemExit(f"STOP: {out_gguf} exists; pass --force to overwrite.")

    convert = find_convert_script(args.llama_cpp_dir)
    if convert is None:
        raise SystemExit("STOP: convert_hf_to_gguf.py not found.\n" + CLONE_HINT)
    quant_bin = find_quantize_binary(args.llama_cpp_dir) if args.quantize != "F16" else None

    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    intermediate = out_gguf.with_name(out_gguf.stem + f".{args.outtype}.gguf")
    t0 = time.time()
    cmd = [args.python, str(convert), str(hf_model), "--outfile", str(intermediate), "--outtype", args.outtype]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    if args.quantize == "F16":
        intermediate.replace(out_gguf)
        quant_tool = None
    elif quant_bin is not None:
        qcmd = [str(quant_bin), str(intermediate), str(out_gguf), args.quantize, str(args.threads)]
        print("Running:", " ".join(qcmd), flush=True)
        subprocess.run(qcmd, check=True)
        quant_tool = str(quant_bin)
    else:
        print(f"llama-quantize not found; quantizing to {args.quantize} with the llama-cpp-python binding.", flush=True)
        quantize_with_binding(intermediate, out_gguf, args.quantize, args.threads)
        import llama_cpp

        quant_tool = f"llama-cpp-python {llama_cpp.__version__} llama_model_quantize"
    if intermediate.exists() and not args.keep_intermediate and intermediate != out_gguf:
        intermediate.unlink()
    elapsed = time.time() - t0

    merge_meta = hf_model / "merge_meta.json"
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(hf_model),
        "source_merge_meta": json.loads(merge_meta.read_text(encoding="utf-8")) if merge_meta.is_file() else None,
        "conversion_tool": str(convert),
        "intermediate_outtype": args.outtype,
        "quantize_tool": quant_tool,
        "quantization_type": args.quantize,
        "imatrix": None,
        "gguf_path": str(out_gguf),
        "gguf_size_bytes": out_gguf.stat().st_size,
        "gguf_sha256": sha256(out_gguf),
        "conversion_time_sec": round(elapsed, 1),
    }
    if not args.skip_verify:
        meta["verification"] = verify(out_gguf, hf_model, args.quantize, args.threads)
    meta_path = out_gguf.with_suffix(out_gguf.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    if not args.skip_verify and not meta["verification"]["ok"]:
        raise SystemExit("VERIFY FAILED: " + "; ".join(meta["verification"]["problems"]))
    print(f"\nOK: {out_gguf}\nNext: evaluate it with\n  python experiments/multitask_bloom_rewrite/scripts/"
          f"evaluate_rewrite.py --condition gguf --gguf {out_gguf} --split validation")


if __name__ == "__main__":
    main()
