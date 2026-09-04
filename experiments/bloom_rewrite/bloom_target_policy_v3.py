"""Bloom target-level rewrite policy v3 (EduGuard experiment only).

Improvements over bloom_target_policy_v1 / synth_v2:
- More structurally diverse templates per level (especially Remember/Understand)
- Prefer clean noun-phrase topics; fall back to short topic spans
- Prefer interrogative or clear exam-imperative forms
- Question-only targets (no answers/explanations)
- Policy version tagged for manifests

Does NOT overwrite v1/v2 policies.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

# Reuse canonical helpers from v1 where safe.
from bloom_target_policy import (  # noqa: E402
    BLOOM_LEVELS,
    CANONICAL_LABELS,
    LEVEL_CODES,
    TOPIC_OVERLAP_THRESHOLD,
    TOPIC_STOPWORDS,
    ValidationResult,
    all_source_target_pairs,
    canonical_level,
    source_is_usable,
)

POLICY_VERSION = "bloom_target_policy_v3"

# Expanded exam-form starters used by validators and QC.
EXAM_IMPERATIVE_STARTERS = (
    "define ",
    "explain ",
    "describe ",
    "list ",
    "name ",
    "identify ",
    "state ",
    "recall ",
    "recognize ",
    "summarize ",
    "interpret ",
    "illustrate ",
    "distinguish ",
    "compare ",
    "contrast ",
    "analyze ",
    "analyse ",
    "examine ",
    "evaluate ",
    "assess ",
    "critique ",
    "justify ",
    "design ",
    "propose ",
    "formulate ",
    "construct ",
    "develop ",
    "create ",
    "calculate ",
    "compute ",
    "determine ",
    "apply ",
    "use ",
    "solve ",
    "given ",
    "using ",
    "for the following ",
    "in the following ",
    "consider ",
)

REWRITE_TEMPLATES_V3: dict[str, tuple[str, ...]] = {
    "Remember": (
        "What is the definition of {topic}?",
        "Name the main components of {topic}.",
        "List the key facts associated with {topic}.",
        "Identify the primary terms used when discussing {topic}.",
        "State the standard definition of {topic}.",
        "Which facts must a student recall about {topic}?",
        "Recognize the essential elements of {topic}.",
        "What are the basic characteristics of {topic}?",
    ),
    "Understand": (
        "Explain why {topic} is used in this academic context.",
        "Describe how {topic} works in principle.",
        "How would you interpret the meaning of {topic}?",
        "Summarize what {topic} means for a learner.",
        "Distinguish the main idea behind {topic} from a related misconception.",
        "Illustrate the purpose of {topic} with a clear conceptual account.",
        "Why does {topic} matter academically?",
        "Describe the relationship between the core ideas in {topic}.",
    ),
    "Apply": (
        "Given a concrete case involving {topic}, how would you apply the relevant procedure?",
        "Calculate the result for a stated problem that requires using {topic}.",
        "Use {topic} to solve the following concrete academic scenario.",
        "Determine the outcome when {topic} is applied to a specific stated case.",
        "In the following problem about {topic}, which procedure should be applied and why?",
        "Apply {topic} to produce a result for a clearly specified practical case.",
    ),
    "Analyze": (
        "Compare the main components of {topic} and explain how they relate.",
        "Analyze how the parts of {topic} interact to produce the overall effect.",
        "What relationships among components explain the structure of {topic}?",
        "Examine the causes and patterns that structure {topic}.",
        "Differentiate the internal elements of {topic} and identify dependencies among them.",
        "Break {topic} into components and analyze the relationships among those parts.",
    ),
    "Evaluate": (
        "Evaluate {topic} using explicit criteria of quality, validity, and suitability, and justify your judgment.",
        "Assess the strengths and limitations of {topic} against stated academic criteria.",
        "Which criteria best support a critical judgment of {topic}, and why?",
        "Critique {topic} by weighing trade-offs and defending a justified conclusion.",
        "Judge the effectiveness of {topic} with evidence and explicit standards.",
        "How well does {topic} meet stated academic criteria, and what justifies that judgment?",
    ),
    "Create": (
        "Design a new plan that uses {topic} under stated constraints to solve a specified problem.",
        "Propose an original strategy that incorporates {topic} to address a defined need.",
        "Formulate a structured approach for constructing a new solution involving {topic}.",
        "Create an original academic artifact that applies {topic} under given constraints.",
        "How would you construct a novel solution that relies on {topic}?",
        "Develop a new procedure based on {topic} to meet stated requirements.",
    ),
}

LEADING_VERB_RE = re.compile(
    r"^(?:please\s+|briefly\s+|clearly\s+)?"
    r"(?:can you |do you |how (?:would|can|do) you |what (?:is|are) |what do you )?"
    r"(?:define|explain|describe|discuss|list|name|state|identify|recall|recognize|"
    r"summarize|interpret|classify|apply|calculate|compute|solve|demonstrate|"
    r"analyze|analyse|compare|contrast|differentiate|examine|evaluate|assess|"
    r"justify|critique|appraise|design|develop|create|propose|formulate|construct|"
    r"suggest|recommend|distinguish|outline|determine|give|show|illustrate|"
    r"mention|specify|write|draw|revise|derive|predict|choose|select|use|implement)\b"
    r"(?:\s+on|\s+the|\s+how|\s+why|\s+what|\s+a|\s+an)?[\s:,-]*",
    re.I,
)

META_PATTERNS = (
    "as an ai",
    "rewritten question",
    "bloom level",
    "target level",
    "original question",
    "the answer is",
    "here is the",
    "this question asks",
)


@dataclass
class TopicInfo:
    topic: str
    content_tokens: frozenset[str]
    ok: bool
    reason: str = ""


def stable_variant_index(key: str, n: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())
        if len(tok) > 2 and tok not in TOPIC_STOPWORDS
    )


def extract_topic_v3(question: str) -> TopicInfo:
    """Prefer a compact noun-phrase topic for template insertion."""
    text = re.sub(r"\s+", " ", str(question).strip().strip('"').strip("'"))
    if not text:
        return TopicInfo("", frozenset(), False, "empty_question")
    remainder = LEADING_VERB_RE.sub("", text).strip()
    remainder = re.sub(r"^(?:the|a|an)\s+", "", remainder, flags=re.I).strip(" .,:;?")
    # Drop dangling clause glue that produced garbled v2 topics.
    remainder = re.sub(
        r"\b(?:can take to|was developed by|name one|other material that can be used as)\b.*",
        "",
        remainder,
        flags=re.I,
    ).strip(" .,:;")
    words = remainder.split()
    if len(words) > 12:
        remainder = " ".join(words[:12])
    if len(remainder) < 4:
        # Fall back to first contentful span from original.
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
        keep = [t for t in tokens if t.lower() not in TOPIC_STOPWORDS][:8]
        remainder = " ".join(keep)
    tokens = _content_tokens(remainder)
    if len(tokens) < 2:
        return TopicInfo(remainder, tokens, False, "insufficient_content_tokens")
    if remainder and remainder[0].islower():
        remainder = remainder[0].upper() + remainder[1:]
    return TopicInfo(remainder, tokens, True, "ok")


def synthesize_rewrite_v3(
    source_question: str,
    source_level: str,
    target_level: str,
    source_id: str,
) -> tuple[str, dict[str, Any]]:
    topic = extract_topic_v3(source_question)
    if not topic.ok:
        raise ValueError(f"Cannot synthesize rewrite: {topic.reason}")
    templates = REWRITE_TEMPLATES_V3[target_level]
    idx = stable_variant_index(f"v3|{source_id}|{source_level}|{target_level}", len(templates))
    rewrite = templates[idx].format(topic=topic.topic)
    meta = {
        "policy_version": POLICY_VERSION,
        "template_index": idx,
        "template_bank_size": len(templates),
        "topic": topic.topic,
        "topic_token_count": len(topic.content_tokens),
    }
    return rewrite, meta


def looks_like_exam_question(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) < 8:
        return False
    if "?" in t:
        return True
    n = re.sub(r"\s+", " ", t.lower())
    return n.startswith(EXAM_IMPERATIVE_STARTERS)


def looks_like_answer_or_explanation(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    n = re.sub(r"\s+", " ", t.lower())
    if any(p in n for p in META_PATTERNS):
        return True
    if re.search(r"\b(the answer is|in conclusion|therefore,|as follows:)\b", n):
        return True
    # Pure declarative sentence without exam cue
    if "?" not in t and not looks_like_exam_question(t):
        if re.match(r"^(it|this|that|there|these|those|virtual|memory|the)\b", n):
            return True
        if re.search(r"\b(is|are|was|were|means|refers to|allows|enables)\b", n) and len(t.split()) > 6:
            return True
    return False


def validate_rewrite_v3(
    source_question: str,
    target_level: str,
    rewrite: str,
    *,
    threshold: float = TOPIC_OVERLAP_THRESHOLD,
) -> ValidationResult:
    reasons: list[str] = []
    rewrite = (rewrite or "").strip()
    if not rewrite:
        return ValidationResult(False, "fail", "EMPTY_OUTPUT", ["empty"])
    if looks_like_answer_or_explanation(rewrite) and not looks_like_exam_question(rewrite):
        return ValidationResult(False, "fail", "ANSWER_OR_DECLARATIVE", ["answer_or_declarative"])
    if not looks_like_exam_question(rewrite):
        return ValidationResult(False, "fail", "INVALID_QUESTION", ["invalid_question_format"])
    src_toks = extract_topic_v3(source_question).content_tokens
    dst_toks = _content_tokens(rewrite)
    overlap = (len(src_toks & dst_toks) / len(src_toks)) if src_toks else 0.0
    if overlap < threshold:
        reasons.append("topic_drift")
    # Soft cognitive cue check by level family
    n = rewrite.lower()
    level = canonical_level(target_level) or target_level
    cue_ok = True
    if level == "Remember" and not any(x in n for x in ("define", "name", "list", "identify", "state", "recall", "what is", "which", "recognize", "facts", "characteristics")):
        cue_ok = False
    if level == "Understand" and not any(x in n for x in ("explain", "describe", "summarize", "interpret", "why", "how", "distinguish", "illustrate", "meaning", "purpose")):
        cue_ok = False
    if level == "Apply" and not any(x in n for x in ("given", "calculate", "apply", "use ", "determine", "solve", "scenario", "case", "procedure", "problem")):
        cue_ok = False
    if level == "Analyze" and not any(x in n for x in ("analyze", "compare", "examine", "relationship", "components", "differentiate", "patterns", "causes", "structure")):
        cue_ok = False
    if level == "Evaluate" and not any(x in n for x in ("evaluate", "assess", "critique", "justify", "criteria", "judgment", "strengths", "limitations", "trade")):
        cue_ok = False
    if level == "Create" and not any(x in n for x in ("design", "propose", "formulate", "construct", "create", "develop", "original", "novel", "new ")):
        cue_ok = False
    if not cue_ok:
        reasons.append("cognitive_mismatch")
    ok = not reasons
    return ValidationResult(
        ok=ok,
        quality_status="pass" if ok else "fail",
        failure_category="" if ok else (reasons[0].upper()),
        reasons=reasons,
        topic_overlap=round(overlap, 4),
    )


def template_inventory() -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "levels": BLOOM_LEVELS,
        "templates_per_level": {k: len(v) for k, v in REWRITE_TEMPLATES_V3.items()},
        "total_templates": sum(len(v) for v in REWRITE_TEMPLATES_V3.values()),
    }


__all__ = [
    "POLICY_VERSION",
    "REWRITE_TEMPLATES_V3",
    "BLOOM_LEVELS",
    "LEVEL_CODES",
    "canonical_level",
    "source_is_usable",
    "all_source_target_pairs",
    "synthesize_rewrite_v3",
    "validate_rewrite_v3",
    "extract_topic_v3",
    "looks_like_exam_question",
    "looks_like_answer_or_explanation",
    "template_inventory",
    "EXAM_IMPERATIVE_STARTERS",
]
