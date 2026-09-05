"""Bloom target-level rewrite policy v3.1 (EduGuard experiment only).

Fixes over synth_v3.0:
- Noun-phrase topic extraction (strip phrasal verbs, secondary clauses, finite tails)
- Template frames that attach with of/concerning/involving (never "How well does {topic}")
- Semantic quality gate rejects dangling fragments and verb leftovers in insertions
- Optional forced template index for balanced usage during corpus build

Does NOT overwrite v1/v2 policies. Does not touch production rewrite code.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

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

POLICY_VERSION = "bloom_target_policy_v3.1"

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
    "break ",
    "break down ",
    "differentiate ",
    "judge ",
)

# All frames attach {topic} as a noun phrase via of / concerning / involving / related to.
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
        "Break down the components of {topic} and analyze the relationships among those parts.",
    ),
    "Evaluate": (
        "Evaluate the effectiveness of {topic} using explicit criteria of quality, validity, and suitability, and justify your judgment.",
        "Assess the strengths and limitations of {topic} against stated academic criteria.",
        "Which criteria best support a critical judgment of {topic}, and why?",
        "Critique {topic} by weighing trade-offs and defending a justified conclusion.",
        "Judge the effectiveness of {topic} with evidence and explicit standards.",
        "Assess how well {topic} meets stated academic criteria, and justify that judgment.",
    ),
    "Create": (
        "Design a new plan that uses {topic} under stated constraints to solve a specified problem.",
        "Propose an original strategy that incorporates {topic} to address a defined need.",
        "Formulate a structured approach for constructing a new solution involving {topic}.",
        "Create an original academic artifact that applies {topic} under given constraints.",
        "How would you construct a novel solution that relies on {topic}?",
        "Develop an original procedure related to {topic} that meets stated requirements.",
    ),
}

LEADING_INSTRUCTION_RE = re.compile(
    r"^(?:please\s+|briefly\s+|clearly\s+|using (?:an )?appropriate (?:diagrams?|examples?|graphs?|illustrations?)[,\s]+|"
    r"based on your understanding[,\s]+|with (?:the )?aid of[^\s]*[,\s]+)?",
    re.I,
)

LEADING_VERB_RE = re.compile(
    r"^(?:can you |do you |would you |could you |how (?:would|can|do) you |what (?:is|are) |what do you )?"
    r"(?:break\s+down|point\s+out|carry\s+out|set\s+out|"
    r"define|explain|describe|discuss|list|name|state|identify|recall|recognize|"
    r"summarize|interpret|classify|apply|calculate|compute|solve|demonstrate|"
    r"analyze|analyse|compare|contrast|differentiate|examine|evaluate|assess|"
    r"justify|critique|appraise|design|develop|create|propose|formulate|construct|"
    r"suggest|recommend|distinguish|outline|determine|give|show|illustrate|"
    r"mention|specify|write|draw|sketch|plot|revise|derive|predict|choose|select|"
    r"use|implement|break|sketch)\b"
    r"(?:\s+on|\s+the|\s+how|\s+why|\s+what|\s+a|\s+an)?[\s:,-]*",
    re.I,
)

SECONDARY_CLAUSE_RE = re.compile(
    r"\s+and\s+(?:then\s+)?"
    r"(?:explain|describe|discuss|analyze|analyse|show|state|identify|evaluate|assess|"
    r"justify|compare|contrast|illustrate|outline|how|what|why)\b.*$",
    re.I,
)

COMPONENT_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:components?|parts?|elements?|aspects?|features?|characteristics?|role|roles)\s+of\s+",
    re.I,
)

WH_PREFIX_RE = re.compile(
    r"^(?:why|how|what|which|when|where)\s+(?:is|are|does|do|did|can|could|would|should)?\s*",
    re.I,
)

# "How many years does this animal live?" → ("years", "this animal live")
HOW_MANY_RE = re.compile(
    r"^how\s+many\s+(\w+)\s+(?:does|do|did|is|are)\s+(.+?)(?:\?|$)",
    re.I,
)
TRAILING_PRED_VERB_RE = re.compile(
    r"\s+(?:live|lives|work|works|mean|means|occur|occurs|happen|happens|exist|exists|"
    r"do|does|did|go|goes|come|comes)\s*$",
    re.I,
)
WEAK_TOPICS = frozenset(
    {
        "many years",
        "years",
        "one",
        "two",
        "three",
        "this",
        "that",
        "these",
        "those",
        "it",
        "something",
        "anything",
        "everything",
    }
)

FINITE_VERB_SPLIT_RE = re.compile(
    r"\s+(?:is|are|was|were|does|do|did|can|could|will|would|may|might|has|have|had)\s+",
    re.I,
)

DANGLING_TAILS = frozenset(
    "the a an of in on for to with and or how what why when where which that "
    "this these those their its such".split()
)

IMPERATIVE_LEFTOVERS = frozenset(
    "break sketch define explain describe discuss list name state identify "
    "analyze analyse evaluate assess design develop create propose draw plot "
    "compare contrast examine calculate apply use solve provide give show "
    "outline suggest recommend make tell write find mention".split()
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

# Patterns that indicate the topic was a clause/fragment stuffed into a frame.
# Do NOT flag legitimate template wording such as "explain how they relate".
MALFORMED_REWRITE_RE = re.compile(
    r"(?:"
    r"\b(?:of|uses|involving|related to|concerning|about|incorporates|relies on|associated with)\s+"
    r"(?:Break|Sketch|Define|Explain|Describe|Discuss|How|What|Which|List|Name|State|Identify|Would)\b"
    r"|\b(?:does|of)\s+How\b"
    r"|\binterest in the\b"
    r"|\bexplain what a\b"
    r"|\bin the(?:\s+[.?]|\s*$)"
    r"|\bhow meet\b"
    r"|\bwhat a meet\b"
    r")",
    re.I,
)


@dataclass
class TopicInfo:
    topic: str
    content_tokens: frozenset[str]
    ok: bool
    reason: str = ""
    is_clause: bool = False


def stable_variant_index(key: str, n: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())
        if len(tok) > 2 and tok not in TOPIC_STOPWORDS
    )


def _strip_dangling(words: list[str]) -> list[str]:
    while words and words[-1].lower().strip(".,:;?!'\"") in DANGLING_TAILS:
        words.pop()
    return words


def _looks_like_imperative_leftover(topic: str) -> bool:
    first = topic.split()[0].lower().strip(".,:;?") if topic.split() else ""
    return first in IMPERATIVE_LEFTOVERS


def topic_is_insertable(topic: str) -> tuple[bool, str]:
    """Reject fragments that destroy grammar when inserted into of/{topic} frames."""
    t = (topic or "").strip()
    if len(t) < 4:
        return False, "topic_too_short"
    words = t.split()
    if len(words) < 2:
        return False, "topic_too_short"
    if t.lower() in WEAK_TOPICS:
        return False, "weak_topic"
    if words[-1].lower().strip(".,:;?") in DANGLING_TAILS:
        return False, "dangling_topic_tail"
    if _looks_like_imperative_leftover(t):
        return False, "imperative_leftover_in_topic"
    if t.lower().startswith(("how ", "what ", "which ", "why ", "when ", "where ")):
        return False, "wh_clause_topic"
    if re.match(r"^(?:would|could|can|should|whether|if|do|does|did)\b", t, flags=re.I):
        return False, "interrogative_residue"
    if re.match(
        r"^(?:based on|using|to |provide|propose|design and|with the|with an|given|your understanding|nd\b|and |in |on |make|tell)\b",
        t,
        flags=re.I,
    ):
        return False, "instruction_residue"
    if "?" in t or re.search(
        r"\bscenarios below\b|\([ivx]+\)|\b(?:TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b|\bas highlighted\b",
        t,
        flags=re.I,
    ):
        return False, "exam_meta_noise"
    if re.search(r"\(\s*\d+\s*\)|\blevel\s*\d|\b(?:five|four|three|two|one)\s*\(\s*\d+", t, flags=re.I):
        return False, "exam_meta_noise"
    if re.search(
        r"[\"']|justify your|provide a relevant|show your working|,\s*(?:propose|design|sketch|justify|provide|describe)\b",
        t,
        flags=re.I,
    ):
        return False, "instruction_residue"
    # Finite clause leftovers (subject + verb) are unsafe in "of {topic}" frames.
    if FINITE_VERB_SPLIT_RE.search(t) and len(words) > 6:
        return False, "finite_clause_topic"
    tokens = _content_tokens(t)
    if len(tokens) < 2:
        return False, "insufficient_content_tokens"
    return True, "ok"


def extract_topic_v3(question: str) -> TopicInfo:
    """Extract a compact noun-phrase topic safe for template insertion."""
    text = re.sub(r"\s+", " ", str(question).strip().strip('"').strip("'"))
    if not text:
        return TopicInfo("", frozenset(), False, "empty_question")

    how_many = HOW_MANY_RE.match(text)
    if how_many:
        unit = how_many.group(1).strip().lower()
        subject = TRAILING_PRED_VERB_RE.sub("", how_many.group(2)).strip(" .,:;?")
        subject = re.sub(r"^(?:the|a|an)\s+", "", subject, flags=re.I).strip()
        if unit in {"years", "months", "days", "hours", "minutes"} and subject:
            candidate = f"lifespan of {subject}"
        else:
            candidate = subject
        ok, reason = topic_is_insertable(candidate)
        if ok:
            tokens = _content_tokens(candidate)
            return TopicInfo(candidate, tokens, True, "ok_how_many")
        return TopicInfo(candidate, _content_tokens(candidate), False, reason or "weak_topic")

    remainder = LEADING_INSTRUCTION_RE.sub("", text).strip()
    for _ in range(4):
        nxt = LEADING_VERB_RE.sub("", remainder).strip()
        if nxt == remainder:
            break
        remainder = nxt

    remainder = SECONDARY_CLAUSE_RE.sub("", remainder).strip(" .,:;?")
    remainder = WH_PREFIX_RE.sub("", remainder).strip(" .,:;?")
    remainder = COMPONENT_PREFIX_RE.sub("", remainder).strip(" .,:;?")
    remainder = re.sub(r"^(?:the|a|an)\s+(?:following|concept of|idea of)\s+", "", remainder, flags=re.I)
    remainder = remainder.strip(" .,:;?")

    # Prefer the subject NP before a finite verb when the remainder is a clause.
    match = FINITE_VERB_SPLIT_RE.search(remainder)
    if match and match.start() >= 8:
        subject = remainder[: match.start()].strip(" .,:;")
        subject_words = _strip_dangling(subject.split())
        if len(subject_words) >= 2 and len(_content_tokens(" ".join(subject_words))) >= 2:
            remainder = " ".join(subject_words)

    # Soft length cap at a phrase boundary (never mid-article).
    words = _strip_dangling(remainder.split())
    if len(words) > 12:
        cut = words[:12]
        for i in range(len(cut) - 1, 4, -1):
            if cut[i].lower() in {"and", "that", "which", "who", "whom", "whose"}:
                cut = cut[:i]
                break
        words = _strip_dangling(cut)
    remainder = " ".join(words).strip(" .,:;?")

    # Last-chance cleanup if an imperative verb survived (e.g. unknown phrasal).
    if _looks_like_imperative_leftover(remainder):
        words = remainder.split()[1:]
        if words and words[0].lower() in {"the", "a", "an", "on", "how", "why", "what"}:
            words = words[1:] if words[0].lower() in {"on", "how", "why", "what"} else words
        remainder = " ".join(_strip_dangling(words)).strip(" .,:;?")

    if len(remainder) < 4:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
        keep = [t for t in tokens if t.lower() not in TOPIC_STOPWORDS][:8]
        remainder = " ".join(keep)

    ok, reason = topic_is_insertable(remainder)
    if not ok:
        return TopicInfo(remainder, _content_tokens(remainder), False, reason)

    # Mid-sentence insertion: prefer lowercase lead unless acronym (DNA, CPU).
    first = remainder.split()[0]
    if first[:1].isupper() and not (first.isupper() and 2 <= len(first) <= 4):
        remainder = remainder[0].lower() + remainder[1:]

    tokens = _content_tokens(remainder)
    is_clause = len(remainder.split()) > 10
    return TopicInfo(remainder, tokens, True, "ok", is_clause=is_clause)


def synthesize_rewrite_v3(
    source_question: str,
    source_level: str,
    target_level: str,
    source_id: str,
    *,
    template_index: int | None = None,
) -> tuple[str, dict[str, Any]]:
    topic = extract_topic_v3(source_question)
    if not topic.ok:
        raise ValueError(f"Cannot synthesize rewrite: {topic.reason}")
    templates = REWRITE_TEMPLATES_V3[target_level]
    if template_index is None:
        idx = stable_variant_index(f"v3.1|{source_id}|{source_level}|{target_level}", len(templates))
    else:
        idx = int(template_index) % len(templates)
    rewrite = templates[idx].format(topic=topic.topic)
    meta = {
        "policy_version": POLICY_VERSION,
        "template_index": idx,
        "template_bank_size": len(templates),
        "topic": topic.topic,
        "topic_token_count": len(topic.content_tokens),
        "topic_is_clause": topic.is_clause,
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
    topic: str | None = None,
) -> ValidationResult:
    reasons: list[str] = []
    rewrite = (rewrite or "").strip()
    if not rewrite:
        return ValidationResult(False, "fail", "EMPTY_OUTPUT", ["empty"])

    topic_info = extract_topic_v3(source_question)
    topic_text = topic if topic is not None else topic_info.topic
    insertable, insert_reason = topic_is_insertable(topic_text)
    if not insertable:
        return ValidationResult(
            False,
            "fail",
            "MALFORMED_TOPIC",
            [insert_reason],
            topic_overlap=0.0,
        )

    if MALFORMED_REWRITE_RE.search(rewrite):
        return ValidationResult(
            False,
            "fail",
            "MALFORMED_REWRITE",
            ["malformed_insertion"],
            topic_overlap=0.0,
        )

    # Reject if the embedded topic itself still looks like a truncated clause.
    if re.search(r"\b(?:in the|what a|explain how)\b", topic_text, flags=re.I):
        return ValidationResult(
            False,
            "fail",
            "MALFORMED_TOPIC",
            ["truncated_topic_phrase"],
            topic_overlap=0.0,
        )

    if looks_like_answer_or_explanation(rewrite) and not looks_like_exam_question(rewrite):
        return ValidationResult(False, "fail", "ANSWER_OR_DECLARATIVE", ["answer_or_declarative"])
    if not looks_like_exam_question(rewrite):
        return ValidationResult(False, "fail", "INVALID_QUESTION", ["invalid_question_format"])

    src_toks = topic_info.content_tokens if topic_info.content_tokens else _content_tokens(source_question)
    dst_toks = _content_tokens(rewrite)
    overlap = (len(src_toks & dst_toks) / len(src_toks)) if src_toks else 0.0
    if overlap < threshold:
        reasons.append("topic_drift")

    n = rewrite.lower()
    level = canonical_level(target_level) or target_level
    cue_ok = True
    if level == "Remember" and not any(
        x in n
        for x in (
            "define",
            "name",
            "list",
            "identify",
            "state",
            "recall",
            "what is",
            "which",
            "recognize",
            "facts",
            "characteristics",
        )
    ):
        cue_ok = False
    if level == "Understand" and not any(
        x in n
        for x in (
            "explain",
            "describe",
            "summarize",
            "interpret",
            "why",
            "how",
            "distinguish",
            "illustrate",
            "meaning",
            "purpose",
        )
    ):
        cue_ok = False
    if level == "Apply" and not any(
        x in n
        for x in (
            "given",
            "calculate",
            "apply",
            "use ",
            "determine",
            "solve",
            "scenario",
            "case",
            "procedure",
            "problem",
        )
    ):
        cue_ok = False
    if level == "Analyze" and not any(
        x in n
        for x in (
            "analyze",
            "compare",
            "examine",
            "relationship",
            "components",
            "differentiate",
            "patterns",
            "causes",
            "structure",
        )
    ):
        cue_ok = False
    if level == "Evaluate" and not any(
        x in n
        for x in (
            "evaluate",
            "assess",
            "critique",
            "justify",
            "criteria",
            "judgment",
            "strengths",
            "limitations",
            "trade",
            "judge",
        )
    ):
        cue_ok = False
    if level == "Create" and not any(
        x in n
        for x in (
            "design",
            "propose",
            "formulate",
            "construct",
            "create",
            "develop",
            "original",
            "novel",
            "new ",
        )
    ):
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
    "CANONICAL_LABELS",
    "canonical_level",
    "source_is_usable",
    "all_source_target_pairs",
    "synthesize_rewrite_v3",
    "validate_rewrite_v3",
    "extract_topic_v3",
    "topic_is_insertable",
    "looks_like_exam_question",
    "looks_like_answer_or_explanation",
    "template_inventory",
    "EXAM_IMPERATIVE_STARTERS",
]
