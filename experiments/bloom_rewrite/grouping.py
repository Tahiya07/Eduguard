"""Leakage-controlled grouping of near-duplicate and question-family items.

Splits MUST be performed on groups, never on individual rows. A semantic
near-duplicate of a training question must not appear in validation or test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

NORMALIZE_RE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_RE = re.compile(r"\s+")

BLOOM_VERB_TOKENS = frozenset(
    """
    define explain describe discuss list name state identify recall recognize
    summarize interpret classify apply calculate compute solve demonstrate
    analyze analyse compare contrast differentiate examine evaluate assess
    justify critique appraise design develop create propose formulate construct
    suggest recommend distinguish highlight outline determine give show comment
    illustrate mention specify write draw revise participate derive predict
    choose select use implement elaborate briefly using appropriate
    """.split()
)

STOPWORDS = frozenset(
    """
    a an the of in on for to with and or is are be this that it its than more
    under what how why which who when where would should could can you your
    based your understanding support answer show working provide relevant
    example following given determine please briefly clearly one two three
    four five six seven eight nine ten
    """.split()
) | BLOOM_VERB_TOKENS


def normalize_question(text: str) -> str:
    s = str(text).lower().strip()
    s = NORMALIZE_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s)
    return s.strip()


def content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in normalize_question(text).split()
        if len(token) > 2 and token not in STOPWORDS
    )


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


@dataclass(frozen=True)
class GroupingThresholds:
    exact_normalized: bool = True
    sequence_ratio: float = 0.90
    family_jaccard: float = 0.80
    family_min_tokens: int = 4
    family_min_overlap: int = 3


def group_questions(
    questions: list[str],
    thresholds: GroupingThresholds | None = None,
) -> list[int]:
    """Return a group_id for each question index using union-find."""
    thresholds = thresholds or GroupingThresholds()
    n = len(questions)
    uf = UnionFind(n)
    norms = [normalize_question(q) for q in questions]
    tokens = [content_tokens(q) for q in questions]

    by_norm: dict[str, int] = {}
    for i, norm in enumerate(norms):
        if thresholds.exact_normalized and norm in by_norm:
            uf.union(i, by_norm[norm])
        else:
            by_norm[norm] = i

    # Pairwise near-duplicate / family detection. n is typically < 3000.
    lengths = [len(norm) for norm in norms]
    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            ti, tj = tokens[i], tokens[j]
            jac = jaccard(ti, tj)
            if (
                len(ti) >= thresholds.family_min_tokens
                and len(tj) >= thresholds.family_min_tokens
                and len(ti & tj) >= thresholds.family_min_overlap
                and jac >= thresholds.family_jaccard
            ):
                uf.union(i, j)
                continue
            if abs(lengths[i] - lengths[j]) > max(40, 0.45 * max(lengths[i], lengths[j], 1)):
                continue
            if jac < 0.25 and abs(lengths[i] - lengths[j]) > 12:
                continue
            ratio = SequenceMatcher(None, norms[i], norms[j]).ratio()
            if ratio >= thresholds.sequence_ratio:
                uf.union(i, j)

    roots = [uf.find(i) for i in range(n)]
    remap: dict[int, int] = {}
    group_ids: list[int] = []
    next_id = 0
    for root in roots:
        if root not in remap:
            remap[root] = next_id
            next_id += 1
        group_ids.append(remap[root])
    return group_ids
