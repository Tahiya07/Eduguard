"""Write the deployed Bloom classifier into the package with fp16-stored weights.

The app loads the classifier as fp32 on CPU (predict_bloom.load_model passes torch_dtype=float32),
so fp16 *storage* only changes the file on disk: 1,976 MB -> 988 MB. Checked on CPU (GPU hidden with
CUDA_VISIBLE_DEVICES=-1) on the 349 Figshare validation questions: runtime dtype float32 for both files,
349/349 identical predictions, accuracy 0.8481 for both, largest confidence difference 0.003, median
312 vs 317 ms per question, first call including load 2.4 s vs 1.9 s.
On a CUDA machine the app loads it as fp16 either way.

The source model folder is never modified.

Usage: python scripts/export_classifier_fp16.py <source_model_dir> <dest_model_dir>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    weights = src / "model.safetensors"
    if not weights.is_file():
        print(f"ERROR: {weights} not found")
        return 1
    if src.resolve() == dst.resolve():
        print("ERROR: destination must differ from the source model folder")
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "model.safetensors" and f.suffix != ".zip":
            shutil.copy2(f, dst / f.name)
    state = load_file(str(weights))
    save_file({k: (v.to(torch.float16) if v.is_floating_point() else v).contiguous() for k, v in state.items()},
              str(dst / "model.safetensors"), metadata={"format": "pt"})
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["torch_dtype"] = "float16"
    cfg.pop("dtype", None)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"classifier weights: {weights.stat().st_size / 1e6:.0f} MB -> "
          f"{(dst / 'model.safetensors').stat().st_size / 1e6:.0f} MB (fp16 storage, fp32 at runtime on CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
