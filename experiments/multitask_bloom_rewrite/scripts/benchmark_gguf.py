#!/usr/bin/env python
"""Benchmark experiment GGUF files (never modifies production models/qwen.gguf)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from paths import RESULTS_DIR  # noqa: E402
from prompts import bloom_messages, render_chatml  # noqa: E402


def rss_uss_mb() -> dict:
    try:
        import psutil

        p = psutil.Process()
        mi = p.memory_info()
        out = {"rss_mb": round(mi.rss / 1024**2, 2)}
        full = p.memory_full_info()
        if hasattr(full, "uss"):
            out["uss_mb"] = round(full.uss / 1024**2, 2)
        return out
    except Exception:
        return {"rss_mb": None, "uss_mb": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    gguf = Path(args.gguf)
    if not gguf.is_absolute():
        gguf = REPO_ROOT / gguf
    if not gguf.exists():
        print(f"STOP: GGUF missing: {gguf}")
        raise SystemExit(2)
    if gguf.resolve() == (REPO_ROOT / "models" / "qwen.gguf").resolve():
        print("NOTE: benchmarking production baseline models/qwen.gguf (read-only).")

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        print("STOP: llama-cpp-python not available:", exc)
        raise SystemExit(2) from exc

    prompts = []
    for i in range(args.n):
        messages = bloom_messages(
            f"What is photosynthesis process number {i}?",
            "Analyze",
        )
        prompts.append(render_chatml(messages, add_generation_prompt=True))

    t_start = time.perf_counter()
    llm = Llama(
        model_path=str(gguf),
        n_ctx=args.ctx,
        n_threads=args.threads,
        verbose=False,
    )
    startup = time.perf_counter() - t_start
    mem_after_load = rss_uss_mb()

    latencies = []
    tokens_per_sec = []
    for prompt in prompts:
        t0 = time.perf_counter()
        out = llm(
            prompt,
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repeat_penalty=1.0,
        )
        dt = time.perf_counter() - t0
        latencies.append(dt)
        n_tok = out.get("usage", {}).get("completion_tokens") or args.max_tokens
        tokens_per_sec.append(n_tok / dt if dt > 0 else None)

    lat_sorted = sorted(latencies)
    p50 = lat_sorted[len(lat_sorted) // 2]
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gguf": str(gguf),
        "gguf_size_bytes": gguf.stat().st_size,
        "startup_sec": round(startup, 4),
        "threads": args.threads,
        "context_size": args.ctx,
        "n_prompts": args.n,
        "max_tokens": args.max_tokens,
        "p50_latency_sec": round(p50, 4),
        "p95_latency_sec": round(p95, 4),
        "mean_latency_sec": round(statistics.mean(latencies), 4),
        "tokens_per_sec_mean": round(
            statistics.mean([t for t in tokens_per_sec if t is not None]), 4
        )
        if any(tokens_per_sec)
        else None,
        "peak_memory_after_load": mem_after_load,
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.0,
            "decoding": "greedy",
        },
    }
    out_path = Path(args.out) if args.out else RESULTS_DIR / "gguf" / f"{gguf.stem}_benchmark.json"
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
