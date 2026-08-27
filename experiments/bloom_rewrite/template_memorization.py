"""Template-memorization diagnostics for the synthetic rewrite corpus.

The targets are deterministic policy templates. Matching them is evidence that
a model learned the synthetic transformation patterns, not that it acquired
independent Bloom reasoning.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from bloom_target_policy import REWRITE_TEMPLATES, REWRITE_TEMPLATES_CLAUSE
from grouping import normalize_question


def _ngrams(text: str, n: int = 3) -> set[str]:
    tokens = normalize_question(text).split()
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _template_skeleton(text: str) -> str:
    """Replace the inserted topic span with a placeholder for frequency counts."""
    lowered = text.strip()
    for bank in (REWRITE_TEMPLATES, REWRITE_TEMPLATES_CLAUSE):
        for level, tpls in bank.items():
            for tpl in tpls:
                prefix, _, suffix = tpl.partition("{topic}")
                if lowered.lower().startswith(prefix.lower()) and lowered.lower().endswith(suffix.lower().rstrip(".")):
                    return f"{level}::{prefix.strip()} [TOPIC] {suffix.strip()}"
    return "UNMATCHED"


def analyze_examples(rows: list[dict]) -> dict:
    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        by_split.setdefault(row.get("split", "train"), []).append(row)
    skeletons = Counter(_template_skeleton(r["target_rewrite"]) for r in rows)
    train_rewrites = {normalize_question(r["target_rewrite"]) for r in by_split.get("train", [])}
    test_rewrites = [normalize_question(r["target_rewrite"]) for r in by_split.get("test", [])]
    exact_train_test_rewrite_overlap = sum(1 for t in test_rewrites if t in train_rewrites)

    train_ng = set()
    for row in by_split.get("train", []):
        train_ng |= _ngrams(row["target_rewrite"], 4)
    test_ng_overlap = []
    for row in by_split.get("test", []):
        grams = _ngrams(row["target_rewrite"], 4)
        if grams:
            test_ng_overlap.append(len(grams & train_ng) / len(grams))
    mean_4gram = sum(test_ng_overlap) / len(test_ng_overlap) if test_ng_overlap else None

    unmatched = skeletons.get("UNMATCHED", 0)
    return {
        "n_examples": len(rows),
        "template_skeleton_counts": dict(skeletons),
        "unmatched_template_rows": unmatched,
        "unmatched_rate": unmatched / len(rows) if rows else None,
        "exact_normalized_rewrite_overlap_train_test": exact_train_test_rewrite_overlap,
        "mean_test_4gram_overlap_with_train": mean_4gram,
        "limitation": (
            "High template frequency and train/test n-gram overlap are expected because "
            "targets are synthetic policy templates. Do not claim the model learned "
            "Bloom reasoning solely from matching these patterns."
        ),
    }


def analyze_generation(generated: str, gold: str, train_rewrites_norm: set[str]) -> dict:
    gen_norm = normalize_question(generated)
    gold_norm = normalize_question(gold)
    grams_g = _ngrams(generated, 3)
    grams_gold = _ngrams(gold, 3)
    overlap = (len(grams_g & grams_gold) / len(grams_g | grams_gold)) if (grams_g or grams_gold) else 0.0
    return {
        "template_skeleton": _template_skeleton(generated),
        "exact_match_gold": gen_norm == gold_norm,
        "in_train_rewrites": gen_norm in train_rewrites_norm,
        "gold_trigram_jaccard": overlap,
    }


def write_report(rows: list[dict], path: Path) -> dict:
    report = analyze_examples(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
