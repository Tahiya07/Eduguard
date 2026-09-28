from pathlib import Path

ROOT = Path(__file__).resolve().parent

for split in ("train", "validation", "test"):
    split_dir = ROOT / split
    parts = sorted(split_dir.glob(f"{split}_*.jsonl"))
    if not parts:
        raise SystemExit(f"No shards found for {split}")
    out = ROOT / f"{split}.jsonl"
    with out.open("w", encoding="utf-8", newline="") as dst:
        for part in parts:
            with part.open("r", encoding="utf-8", newline="") as src:
                dst.write(src.read())
    print(f"{split}: {len(parts)} shards -> {out.name}")
