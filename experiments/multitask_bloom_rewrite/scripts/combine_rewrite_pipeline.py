#!/usr/bin/env python
"""Score the app's deployed Bloom-rewrite pipeline from two finished GGUF evaluations.

The app writes each rewrite with the base model + v3.9 LoRA at reduced strength ("blend") and keeps it
when the Bloom classifier agrees with the requested level and it does not refer to "supplied" material
the question never gave; otherwise it uses the full-strength v3.9 rewrite. This replays that rule per
item on the stored predictions (no new generation) and reports blend / v3.9 / pipeline side by side.

  python experiments/multitask_bloom_rewrite/scripts/combine_rewrite_pipeline.py \
      --blend <results>_gguf_q4_k_m_blend050 --full <results>_gguf_q4_k_m
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
SUPPLIED = re.compile(r"\b(?:the|a|an)\s+supplied\b", re.I)


def load(path: Path) -> dict[str, dict]:
    out = {}
    for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("task") == "bloom_rewrite":
                out[r["id"]] = r
    return out


def invents(r: dict) -> bool:
    return bool(SUPPLIED.search(r.get("prediction") or "")) and "supplied" not in (r.get("source_question") or "").lower()


def accepted(r: dict) -> bool:
    return bool((r.get("prediction") or "").strip()) and bool(r.get("target_match")) and not invents(r)


def summary(records: list[dict]) -> dict:
    n = len(records) or 1
    per_level = {}
    for lvl in LEVELS:
        sub = [r for r in records if r["target_bloom_level"] == lvl]
        per_level[lvl] = round(sum(bool(r.get("target_match")) for r in sub) / len(sub), 4) if sub else None
    return {
        "n": len(records),
        "target_accuracy": round(sum(bool(r.get("target_match")) for r in records) / n, 4),
        "fully_validated": round(sum(bool(r.get("fully_validated")) for r in records) / n, 4),
        "invents_supplied_material": round(sum(invents(r) for r in records) / n, 4),
        "trivial_copy": round(sum(bool(r.get("trivial_transform")) for r in records) / n, 4),
        "per_level_target_accuracy": per_level,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend", required=True, type=Path)
    ap.add_argument("--full", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    blend, full = load(args.blend), load(args.full)
    ids = sorted(set(blend) & set(full))
    pipeline, fallbacks = [], 0
    for i in ids:
        if accepted(blend[i]):
            pipeline.append(blend[i])
        else:
            pipeline.append(full[i])
            fallbacks += 1
    result = {
        "rule": "keep blend if classifier target match and no invented 'supplied' material; else full-strength v3.9",
        "n_items": len(ids),
        "fallback_rate": round(fallbacks / (len(ids) or 1), 4),
        "blend": summary([blend[i] for i in ids]),
        "full_v3_9": summary([full[i] for i in ids]),
        "pipeline": summary(pipeline),
        "sources": {"blend": str(args.blend), "full": str(args.full)},
    }
    out = args.out or args.blend / "pipeline_vs_full.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = ["| Measure | Blend (base + LoRA) | Full v3.9 | Deployed pipeline |", "|---|---:|---:|---:|"]
    for m in ("target_accuracy", "fully_validated", "invents_supplied_material", "trivial_copy"):
        rows.append(f"| {m} | {result['blend'][m]} | {result['full_v3_9'][m]} | {result['pipeline'][m]} |")
    for lvl in LEVELS:
        rows.append(f"| {lvl} accuracy | {result['blend']['per_level_target_accuracy'][lvl]} | "
                    f"{result['full_v3_9']['per_level_target_accuracy'][lvl]} | {result['pipeline']['per_level_target_accuracy'][lvl]} |")
    text = f"Items: {len(ids)} · fallback to full v3.9: {result['fallback_rate']:.1%}\n\n" + "\n".join(rows)
    out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
