#!/usr/bin/env python
"""Template memorization diagnostics for synthetic Bloom rewrites."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from paths import REPORTS_DIR, TASK_SEPARATED_DATA_DIR  # noqa: E402

WORD_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def ngrams(toks: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    # Inspect training targets only. Held-out validation/test targets must not
    # influence curation or training decisions.
    train = read_jsonl(TASK_SEPARATED_DATA_DIR / "bloom_rewrite" / "train.jsonl")
    train_rewrites = [r.get("target_rewrite") or "" for r in train]
    train_by_level = {}
    for level in sorted({str(r.get("target_bloom_level", "")) for r in train}):
        rewrites = [
            str(r.get("target_rewrite") or "")
            for r in train
            if r.get("target_bloom_level") == level
        ]
        patterns = Counter(tuple(tokens(text)[:5]) for text in rewrites if tokens(text))
        train_by_level[level] = {
            "rows": len(rewrites),
            "distinct_openings": len(patterns),
            "top_openings": [
                {"opening": " ".join(pattern), "count": count}
                for pattern, count in patterns.most_common(10)
            ],
        }

    # Within-training n-gram repetition is diagnostic only, not proof of
    # semantic template reuse.
    train_bigrams = Counter()
    for t in train_rewrites:
        train_bigrams.update(ngrams(tokens(t), 2))
    exact_duplicates = [
        {"rewrite": text, "count": count}
        for text, count in Counter(
            text.strip().lower() for text in train_rewrites if text.strip()
        ).most_common()
        if count > 1
    ]

    report = {
        "dataset": str((TASK_SEPARATED_DATA_DIR / "bloom_rewrite" / "train.jsonl").relative_to(REPO_ROOT)),
        "train_n": len(train),
        "per_level_opening_patterns": train_by_level,
        "most_repeated_training_bigrams": [
            {"bigram": " ".join(gram), "count": count}
            for gram, count in train_bigrams.most_common(30)
        ],
        "exact_training_rewrite_duplicates": exact_duplicates,
        "warning": (
            "Repeated wording is a review signal only. Confirm cognitive alignment, "
            "source preservation, and meaning manually before changing examples."
        ),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "template_memorization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    md = [
        "# Template Memorization Report",
        "",
        "This diagnostic reads only Bloom training targets. It does not inspect held-out rows.",
        "",
        "```json",
        json.dumps(report, indent=2),
        "```",
        "",
    ]
    (REPORTS_DIR / "template_memorization_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
