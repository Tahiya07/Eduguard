#!/usr/bin/env python
"""Measure CPU-only GGUF deployment cost for a rewrite generator.

File size is recorded, but compliance with the ~1 GB memory budget is based
on measured RSS/USS during actual inference, not on GGUF file size.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import REWRITE_DATA_DIR  # noqa: E402
from prompt_format import build_generation_prompt  # noqa: E402


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def process_memory() -> dict:
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        out = {"rss_bytes": int(mem.rss), "rss_mb": round(mem.rss / 1024**2, 2)}
        try:
            full = proc.memory_full_info()
            if hasattr(full, "uss"):
                out["uss_bytes"] = int(full.uss)
                out["uss_mb"] = round(full.uss / 1024**2, 2)
        except Exception:
            out["uss_bytes"] = None
            out["uss_mb"] = None
        return out
    except ImportError:
        return {"rss_bytes": None, "uss_bytes": None, "error": "psutil missing"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--dataset-dir", default=str(REWRITE_DATA_DIR))
    parser.add_argument("--n-prompts", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    gguf = Path(args.gguf)
    if not gguf.is_file():
        raise SystemExit(f"GGUF not found: {gguf}")
    data_dir = Path(args.dataset_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    rows = read_jsonl(data_dir / "test.jsonl", args.n_prompts)
    if not rows:
        raise SystemExit("No test prompts available.")

    file_size = gguf.stat().st_size
    baseline = process_memory()
    t0 = time.perf_counter()
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit(
            "llama-cpp-python is required for GGUF resource measurement. "
            f"Import failed: {exc}"
        ) from exc

    llm = Llama(
        model_path=str(gguf),
        n_ctx=args.ctx_size,
        n_threads=args.threads,
        n_gpu_layers=0,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )
    startup_s = time.perf_counter() - t0
    after_load = process_memory()

    latencies = []
    first_token = []
    for row in rows:
        prompt = build_generation_prompt(
            row["source_question"],
            row["target_bloom_level"],
        )
        started = time.perf_counter()
        first_s = None
        try:
            stream = llm(
                prompt,
                max_tokens=args.max_tokens,
                temperature=0.0,
                stream=True,
            )
            for i, _chunk in enumerate(stream):
                if i == 0:
                    first_s = time.perf_counter() - started
            elapsed = time.perf_counter() - started
        except TypeError:
            out = llm(prompt, max_tokens=args.max_tokens, temperature=0.0)
            elapsed = time.perf_counter() - started
            first_s = None
            _ = out
        latencies.append(elapsed)
        if first_s is not None:
            first_token.append(first_s)

    peak = process_memory()
    report = {
        "gguf": str(gguf),
        "gguf_file_size_bytes": file_size,
        "gguf_file_size_mb": round(file_size / 1024**2, 2),
        "startup_s": startup_s,
        "generation_latency_s": {
            "n": len(latencies),
            "mean": sum(latencies) / len(latencies),
            "max": max(latencies),
            "min": min(latencies),
        },
        "first_token_latency_s": {
            "n": len(first_token),
            "mean": (sum(first_token) / len(first_token)) if first_token else None,
        },
        "baseline_memory": baseline,
        "after_load_memory": after_load,
        "peak_memory": peak,
        "cpu_threads": args.threads,
        "context_size": args.ctx_size,
        "cpu_only": True,
        "memory_budget_mb": 1024,
        "meets_1gb_budget_by_peak_rss": (
            peak.get("rss_mb") is not None and peak["rss_mb"] <= 1024
        ),
        "note": "Do not claim budget compliance from GGUF file size alone.",
        "measured_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(args.output) if args.output else Path(str(gguf) + ".resource.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
