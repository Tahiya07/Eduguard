#!/usr/bin/env python
"""GGUF quality+resource benchmark using identical generation settings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--n-prompts", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=512)
    args = parser.parse_args()
    gguf = Path(args.gguf)
    if not gguf.is_file():
        raise SystemExit(
            json.dumps(
                {
                    "status": "GGUF NOT FOUND",
                    "path": str(gguf),
                    "note": "Conversion has not been run. Do not fabricate GGUF metrics.",
                },
                indent=2,
            )
        )
    from measure_rewrite_resources import main as measure_main

    sys.argv = [
        sys.argv[0],
        "--gguf", str(gguf),
        "--n-prompts", str(args.n_prompts),
        "--threads", str(args.threads),
        "--ctx-size", str(args.ctx_size),
        "--output", str(gguf) + ".resource.json",
    ]
    measure_main()


if __name__ == "__main__":
    main()
