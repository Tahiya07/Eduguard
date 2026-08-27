#!/usr/bin/env python
"""Recompute dataset hash after prompt-format change (includes SFT text)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "bloom_rewrite"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    rows = []
    for split in ("train", "validation", "test"):
        with (DATA / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
    lines = sorted(f"{ex['example_id']}|{ex.get('text', '')}" for ex in rows)
    new_hash = sha("\n".join(lines))
    leaked = sum(
        1
        for row in rows
        if "Original Bloom level:" in (row.get("text") or "")
        or "Source Bloom level:" in (row.get("text") or "")
    )
    print("n", len(rows))
    print("new_hash", new_hash)
    print("leaked", leaked)
    if leaked:
        raise SystemExit("source Bloom still in SFT text")
    for name in ("dataset_manifest.json", "dataset_statistics.json"):
        path = DATA / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dataset_hash"] = new_hash
        if name == "dataset_statistics.json":
            payload["prompt_format"] = "question_plus_target_only_v2"
            meth = payload.setdefault("methodology", {})
            meth["generator_task"] = "question + target_level -> rewrite"
            meth["source_bloom_level_role"] = "metadata_only_not_in_generator_prompt"
            meth["prompt_format_version"] = "production_aligned_v2"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    subprocess.check_call(
        [sys.executable, str(ROOT / "experiments/bloom_rewrite/scripts/prepare_dataset.py"), "--validate-only"],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    main()
