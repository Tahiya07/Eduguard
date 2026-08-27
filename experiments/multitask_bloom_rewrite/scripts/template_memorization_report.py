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

from paths import BLOOM_REWRITE_DIR, REPORTS_DIR  # noqa: E402

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
    train = read_jsonl(BLOOM_REWRITE_DIR / "train.jsonl")
    test = read_jsonl(BLOOM_REWRITE_DIR / "test.jsonl")
    train_rewrites = [r.get("target_rewrite") or "" for r in train]
    test_rewrites = [r.get("target_rewrite") or "" for r in test]

    # Template frequency via leading 5-token patterns
    train_templates = Counter(tuple(tokens(t)[:5]) for t in train_rewrites if tokens(t))
    test_templates = Counter(tuple(tokens(t)[:5]) for t in test_rewrites if tokens(t))
    shared_templates = set(train_templates) & set(test_templates)

    train_bigrams = Counter()
    for t in train_rewrites:
        train_bigrams.update(ngrams(tokens(t), 2))
    test_bigrams = Counter()
    for t in test_rewrites:
        test_bigrams.update(ngrams(tokens(t), 2))
    shared_bi = set(train_bigrams) & set(test_bigrams)

    # Exact rewrite overlap train/test
    exact = set(x.strip().lower() for x in train_rewrites) & set(
        x.strip().lower() for x in test_rewrites
    )

    report = {
        "dataset": "bloom_rewrite_synth_v2",
        "train_n": len(train),
        "test_n": len(test),
        "top_train_templates": [
            {"template": " ".join(k), "count": v}
            for k, v in train_templates.most_common(20)
        ],
        "shared_leading5_templates": len(shared_templates),
        "shared_leading5_templates_pct_of_test_templates": round(
            len(shared_templates) / max(1, len(test_templates)), 4
        ),
        "shared_bigrams": len(shared_bi),
        "exact_rewrite_overlap_train_test": len(exact),
        "warning": (
            "Bloom rewrite supervision is synthetic. High template overlap indicates "
            "risk of template memorization rather than general transformation learning."
        ),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "template_memorization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    md = [
        "# Template Memorization Report",
        "",
        "Bloom rewrite targets are **synthetic**. This report measures template/n-gram overlap.",
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
