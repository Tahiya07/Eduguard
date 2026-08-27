"""Offline topic/semantic preservation and Bloom example validation."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from paths import TOPIC_SIMILARITY_THRESHOLD

NORMALIZE_RE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_RE = re.compile(r"\s+")

BLOOM_LEVELS = (
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
)

META_PATTERNS = (
    r"\bas an ai\b",
    r"\bi (?:cannot|can't|won't)\b",
    r"\brewrit(?:e|ten|ing)\b",
    r"\bbloom(?:'s)?\s+level\b",
    r"\btarget\s+level\b",
    r"\boriginal\s+question\b",
    r"\bhere is\b",
    r"\bhere'?s the\b",
)

ANSWER_PATTERNS = (
    r"\bthe answer is\b",
    r"\binsufficient information\b",
    r"\bas follows:\b",
)

STOPWORDS = frozenset(
    """
    a an the of in on for to with and or is are be this that it its than more
    under what how why which who when where would should could can you your
    based understanding support answer show working provide relevant example
    following given determine please briefly clearly
    """.split()
)


def _normalize(text: str) -> str:
    s = str(text).lower().strip()
    s = NORMALIZE_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s)
    return s.strip()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        t for t in _normalize(text).split() if len(t) > 2 and t not in STOPWORDS
    )


def semantic_similarity(a: str, b: str) -> float:
    """Jaccard overlap on content tokens (offline, no extra model required)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def topic_preserved(a: str, b: str, threshold: float = TOPIC_SIMILARITY_THRESHOLD) -> bool:
    return semantic_similarity(a, b) >= threshold


def _looks_like_question(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if "?" in t:
        return True
    starters = (
        "what ",
        "why ",
        "how ",
        "which ",
        "when ",
        "where ",
        "who ",
        "whom ",
        "whose ",
        "define ",
        "explain ",
        "describe ",
        "list ",
        "name ",
        "identify ",
        "compare ",
        "contrast ",
        "evaluate ",
        "assess ",
        "design ",
        "propose ",
        "create ",
        "develop ",
        "analyze ",
        "analyse ",
        "discuss ",
        "calculate ",
        "apply ",
        "justify ",
        "critique ",
    )
    return _normalize(t).startswith(starters)


def _has_meta(text: str) -> bool:
    n = _normalize(text)
    return any(re.search(p, n) for p in META_PATTERNS)


def _looks_like_answer(text: str) -> bool:
    n = _normalize(text)
    if any(re.search(p, n) for p in ANSWER_PATTERNS):
        return True
    # Declarative-only without question cue
    if "?" not in text and not _looks_like_question(text):
        return True
    return False


def _trivial_verb_swap(source: str, rewrite: str) -> bool:
    """Heuristic: same content tokens after stripping common Bloom verbs."""
    bloom_verbs = frozenset(
        """
        define explain describe discuss list name state identify recall
        summarize interpret classify apply calculate compute solve analyze
        analyse compare contrast examine evaluate assess justify critique
        design develop create propose formulate construct suggest recommend
        """.split()
    )
    src = _tokens(source) - bloom_verbs
    dst = _tokens(rewrite) - bloom_verbs
    if not src or not dst:
        return False
    overlap = len(src & dst) / len(src | dst)
    # Nearly identical content with only verb-ish change
    src_words = _normalize(source).split()
    dst_words = _normalize(rewrite).split()
    if abs(len(src_words) - len(dst_words)) <= 2 and overlap >= 0.92:
        return True
    return False


COGNITIVE_HINTS: dict[str, tuple[str, ...]] = {
    "Remember": ("list", "name", "define", "state", "identify", "recall", "what is", "which of"),
    "Understand": ("explain", "describe", "summarize", "interpret", "classify", "discuss", "why"),
    "Apply": ("apply", "calculate", "solve", "use", "demonstrate", "scenario", "case", "given"),
    "Analyze": ("analyze", "analyse", "compare", "contrast", "differentiate", "examine", "relationship"),
    "Evaluate": ("evaluate", "assess", "justify", "critique", "judge", "strength", "weakness", "criteria"),
    "Create": ("design", "develop", "create", "propose", "formulate", "construct", "plan", "invent"),
}


def _cognitive_valid(rewrite: str, target_level: str) -> bool:
    level = target_level.strip().title()
    if level not in COGNITIVE_HINTS:
        return False
    n = _normalize(rewrite)
    hints = COGNITIVE_HINTS[level]
    # Presence of a preferred cue helps but is not required; absence alone is not rejection.
    # Require student-facing question form and non-empty content, then soft cue OR structural length.
    if not _looks_like_question(rewrite):
        return False
    if any(h in n for h in hints):
        return True
    # Soft accept: enough content words for a non-trivial task statement
    return len(_tokens(rewrite)) >= 4


@dataclass
class BloomValidationResult:
    format_valid: bool
    semantic_valid: bool
    cognitive_valid: bool
    trivial_transform: bool
    topic_preserved: bool
    accepted: bool
    rejection_reason: str | None
    semantic_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_bloom_example(
    source_question: str,
    target_level: str,
    target_rewrite: str,
    *,
    threshold: float = TOPIC_SIMILARITY_THRESHOLD,
) -> BloomValidationResult:
    reasons: list[str] = []
    rewrite = (target_rewrite or "").strip()
    format_ok = bool(rewrite) and _looks_like_question(rewrite) and not _has_meta(rewrite)
    if not rewrite:
        reasons.append("empty_output")
    elif _has_meta(rewrite):
        reasons.append("meta_response")
    elif not _looks_like_question(rewrite):
        reasons.append("invalid_question_format")
    if _looks_like_answer(rewrite) and "?" not in rewrite:
        reasons.append("answer_or_declarative")
        format_ok = False

    sim = semantic_similarity(source_question, rewrite)
    topic_ok = sim >= threshold
    semantic_ok = topic_ok and not bool(reasons and "empty_output" in reasons)
    if not topic_ok:
        reasons.append("topic_drift")

    trivial = _trivial_verb_swap(source_question, rewrite)
    if trivial:
        reasons.append("trivial_verb_swap")

    cog_ok = _cognitive_valid(rewrite, target_level)
    if not cog_ok:
        reasons.append("cognitive_invalid")

    accepted = format_ok and semantic_ok and cog_ok and not trivial and topic_ok
    return BloomValidationResult(
        format_valid=format_ok,
        semantic_valid=semantic_ok,
        cognitive_valid=cog_ok,
        trivial_transform=trivial,
        topic_preserved=topic_ok,
        accepted=accepted,
        rejection_reason=None if accepted else ";".join(reasons) or "rejected",
        semantic_similarity=round(sim, 4),
    )
