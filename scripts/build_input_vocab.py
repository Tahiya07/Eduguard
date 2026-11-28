"""Build input_vocab.txt, the word list input_validation.py uses to reject gibberish questions.

Words come from the project's own question, QA and lecture datasets, so the list covers the
academic and CSE vocabulary teachers actually type. Words seen fewer than MIN_COUNT times
are dropped so typos in the data do not become "valid" words.

Usage: .venv/Scripts/python.exe scripts/build_input_vocab.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "input_vocab.txt"
MIN_COUNT = 2
WORD = re.compile(r"[a-z]{2,}")

SOURCES_JSONL = [
    *sorted((ROOT / "data" / "multitask_bloom_rewrite_separate_eduguard_v3_9").glob("*/*.jsonl")),
    *sorted((ROOT / "data" / "summarization_v3_1").glob("*.jsonl")),
]
SOURCES_CSV = [ROOT / "data" / "figshare_combined_dataset.csv"]


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def main() -> None:
    counts: Counter[str] = Counter()
    for path in SOURCES_JSONL:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                for text in strings(json.loads(line)):
                    counts.update(WORD.findall(text.lower()))
    for path in SOURCES_CSV:
        if path.exists():
            with path.open(encoding="utf-8", errors="ignore", newline="") as fh:
                for row in csv.reader(fh):
                    for text in row:
                        counts.update(WORD.findall(text.lower()))
    words = sorted(w for w, c in counts.items() if c >= MIN_COUNT)
    OUT.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"{len(words):,} words -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
