"""Central Bloom target-level rewrite policy for the EduGuard experiment.

This module is independent of the production rewrite pipeline. It must not be
imported by production code until the experiment is complete and a model is
selected from measured evidence.

Policies reason about the STUDENT TASK, not merely action verbs.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

POLICY_VERSION = "bloom_target_policy_v1"
TOPIC_OVERLAP_THRESHOLD = float(os.environ.get("BLOOM_SEMANTIC_SIMILARITY_THRESHOLD", "0.20"))

BLOOM_LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

LEVEL_CODES = {
    "Remember": "C1",
    "Understand": "C2",
    "Apply": "C3",
    "Analyze": "C4",
    "Evaluate": "C5",
    "Create": "C6",
}

CANONICAL_LABELS = {
    "remember": "Remember",
    "remembering": "Remember",
    "knowledge": "Remember",
    "recall": "Remember",
    "understand": "Understand",
    "understanding": "Understand",
    "comprehension": "Understand",
    "apply": "Apply",
    "applying": "Apply",
    "application": "Apply",
    "analyze": "Analyze",
    "analyse": "Analyze",
    "analyzing": "Analyze",
    "analysing": "Analyze",
    "analysis": "Analyze",
    "evaluate": "Evaluate",
    "evaluating": "Evaluate",
    "evaluation": "Evaluate",
    "create": "Create",
    "creating": "Create",
    "synthesis": "Create",
}

TARGET_POLICIES: dict[str, dict[str, object]] = {
    "Remember": {
        "code": "C1",
        "operation": "recall",
        "student_task": (
            "The student recalls, identifies, lists, names, defines, states, "
            "or recognizes facts, terms, or components without explaining "
            "relationships, applying a procedure, judging quality, or designing "
            "a new artifact."
        ),
        "allowed_operations": ["identify", "list", "name", "define", "state", "recognize", "recall"],
        "forbidden_operations": [
            "explain how",
            "apply in a scenario",
            "analyze relationships",
            "evaluate using criteria",
            "design a new artifact",
        ],
        "required_structure": "recall_without_elaboration",
    },
    "Understand": {
        "code": "C2",
        "operation": "comprehension",
        "student_task": (
            "The student explains, describes, summarizes, interprets, classifies, "
            "or discusses an existing concept. The task is comprehension of what "
            "already exists, not a new case solution, judgment, or design."
        ),
        "allowed_operations": ["explain", "describe", "summarize", "interpret", "classify", "discuss"],
        "forbidden_operations": [
            "solve a concrete case",
            "judge with criteria",
            "design a new artifact",
        ],
        "required_structure": "explain_existing_concept",
    },
    "Apply": {
        "code": "C3",
        "operation": "application",
        "student_task": (
            "The student uses knowledge or a procedure in a concrete scenario, "
            "case, calculation, or problem and determines a result."
        ),
        "allowed_operations": ["apply", "use", "demonstrate", "implement", "solve", "execute", "determine"],
        "forbidden_operations": [
            "merely define",
            "merely explain",
            "evaluate using criteria",
            "design a new artifact",
        ],
        "required_structure": "concrete_scenario_or_procedure",
    },
    "Analyze": {
        "code": "C4",
        "operation": "analysis",
        "student_task": (
            "The student breaks information into components and examines "
            "relationships, comparisons, causes, patterns, or structure."
        ),
        "allowed_operations": ["analyze", "compare", "differentiate", "examine", "investigate", "categorize"],
        "forbidden_operations": [
            "merely define",
            "judge overall quality without examining parts",
            "design a new artifact",
        ],
        "required_structure": "parts_relationships_or_comparison",
    },
    "Evaluate": {
        "code": "C5",
        "operation": "judgment",
        "student_task": (
            "The student judges effectiveness, validity, quality, or suitability "
            "using explicit criteria or evidence and justifies the judgment."
        ),
        "allowed_operations": ["evaluate", "assess", "justify", "critique", "defend", "judge"],
        "forbidden_operations": [
            "only describe",
            "design a new artifact without judgment",
        ],
        "required_structure": "judgment_with_criteria",
    },
    "Create": {
        "code": "C6",
        "operation": "creation",
        "student_task": (
            "The student designs, formulates, develops, proposes, constructs, "
            "or produces a novel artifact, solution, or plan under constraints."
        ),
        "allowed_operations": ["design", "develop", "construct", "formulate", "propose", "create"],
        "forbidden_operations": [
            "only recall",
            "only explain",
            "only judge without producing something new",
        ],
        "required_structure": "novel_artifact_plan_or_solution",
    },
}

META_PATTERNS = [
    "this question asks",
    "the rewritten question",
    "the student should",
    "the student must",
    "to understand",
    "this requires",
    "this rewrite",
    "the answer is",
    "bloom level",
    "this is a remember",
    "this is an understand",
    "this is an apply",
    "this is an analyze",
    "this is an evaluate",
    "this is a create",
]

ANSWER_PATTERNS = [
    "the answer is",
    "the correct answer",
    "therefore the result",
    "in conclusion,",
]

INCOMPLETE_RE = re.compile(
    r"\.\s*\.\s*\.|_{3,}|\.{3}|fill in the blank|how was this similar to|what do you recall\s*\.",
    re.I,
)

LEADING_INSTRUCTION_RE = re.compile(
    r"^(?:please\s+|briefly\s+|clearly\s+|using (?:an )?appropriate (?:diagrams?|examples?|graphs?|illustrations?)[,\s]+|"
    r"based on your understanding[,\s]+|with (?:the )?aid of[^\s]*[,\s]+)?",
    re.I,
)

LEADING_VERB_RE = re.compile(
    r"^(?:can you |do you |how (?:would|can|do) you |what (?:is|are) |what do you )?"
    r"(?:define|explain|describe|discuss|list|name|state|identify|recall|recognize|"
    r"summarize|interpret|classify|apply|calculate|compute|solve|demonstrate|"
    r"analyze|analyse|compare|contrast|differentiate|examine|evaluate|assess|"
    r"justify|critique|appraise|design|develop|create|propose|formulate|construct|"
    r"suggest|recommend|distinguish|highlight|outline|determine|give|show|comment|"
    r"illustrate|mention|specify|write|draw|revise|participate|derive|predict|"
    r"choose|select|use|implement|elaborate|compute)\b(?:\s+on|\s+the|\s+how|\s+why|\s+what|\s+a|\s+an)?[\s:,-]*",
    re.I,
)

TRAILING_INSTRUCTION_RE = re.compile(
    r"(?:show your working.*|justify your answer.*|support your answer.*|"
    r"provide (?:a )?relevant example.*|include (?:an? )?example.*)$",
    re.I,
)

COUNT_RE = re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*\(\s*\d+\s*\)\s*", re.I)
COUNT_NOUN_PREFIX_RE = re.compile(
    r"^(?:methods?|ways?|reasons?|differences?|benefits?|factors?|types?|examples?|"
    r"suggestions?|approaches?|steps?|characteristics?|roles?|functions?)\s+(?:that\s+|which\s+)?",
    re.I,
)
CLAUSE_MARKERS = frozenset("can could should would will may might that how why when where who whom whose".split())

TOPIC_STOPWORDS = frozenset(
    "a an the of in on for to with and or is are be this that it its than more under using".split()
)

APPLY_MARKERS = (
    "given ",
    "scenario",
    "concrete",
    "case ",
    "calculate",
    "determine how",
    "use ",
    "apply ",
    "solve ",
    "procedure",
    "stated problem",
)
ANALYZE_MARKERS = (
    "analyze",
    "compare",
    "differentiate",
    "examine",
    "relationship",
    "components",
    "interact",
    "causes",
    "patterns",
    "structure",
)
EVALUATE_MARKERS = (
    "evaluate",
    "assess",
    "critique",
    "justify",
    "criteria",
    "effectiveness",
    "validity",
    "suitability",
    "strengths",
    "limitations",
    "trade-off",
    "judgment",
)
CREATE_MARKERS = (
    "design",
    "propose",
    "formulate",
    "construct",
    "develop a",
    "create a",
    "original",
    "new plan",
    "artifact",
    "under stated constraints",
)
UNDERSTAND_MARKERS = (
    "explain",
    "describe",
    "summarize",
    "interpret",
    "classify",
    "discuss",
    "why it is used",
    "how ",
    "purpose",
)
REMEMBER_MARKERS = (
    "define",
    "identify",
    "list ",
    "name ",
    "state ",
    "recall",
    "recognize",
)

REWRITE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "Remember": (
        "Define {topic}.",
        "Identify the key components of {topic}.",
        "Name the primary characteristics of {topic}.",
    ),
    "Understand": (
        "Explain how {topic} works and why it is used.",
        "Describe the purpose and main characteristics of {topic}.",
        "Summarize the role of {topic} in this academic context.",
    ),
    "Apply": (
        "Given a concrete academic scenario involving {topic}, determine how {topic} would be used to produce a result.",
        "Use {topic} to solve a specific stated case that requires applying the relevant procedure.",
        "Apply {topic} to a concrete problem and determine the outcome.",
    ),
    "Analyze": (
        "Analyze how the components of {topic} interact and identify the relationships among them.",
        "Compare the internal parts of {topic} and examine the causes or patterns that structure it.",
        "Examine the structure of {topic} by breaking it into components and relating those parts.",
    ),
    "Evaluate": (
        "Evaluate the effectiveness of {topic} using explicit criteria of quality, validity, and suitability, and justify your judgment.",
        "Assess the strengths and limitations of {topic} against stated academic criteria and defend your conclusion.",
        "Critique {topic} considering validity, suitability, and trade-offs, and justify your judgment with evidence.",
    ),
    "Create": (
        "Design a new plan that uses {topic} under stated constraints to produce an original solution.",
        "Propose an original academic artifact or strategy that incorporates {topic} to address a specified problem.",
        "Formulate a structured approach that constructs a new solution based on {topic} under given constraints.",
    ),
}

REWRITE_TEMPLATES_CLAUSE: dict[str, tuple[str, ...]] = {
    "Remember": (
        "Identify the key facts about {topic}.",
        "State the main points concerning {topic}.",
        "Name the essential elements involved in {topic}.",
    ),
    "Understand": (
        "Explain the meaning of {topic} and why it matters academically.",
        "Describe what is meant by {topic}.",
        "Summarize the academic idea behind {topic}.",
    ),
    "Apply": (
        "Given a concrete academic scenario about {topic}, determine the result of applying the relevant procedure.",
        "Use the relevant procedure for {topic} to solve a specific stated case.",
        "Apply the appropriate method to a concrete problem concerning {topic} and determine the outcome.",
    ),
    "Analyze": (
        "Analyze the components involved in {topic} and identify the relationships among them.",
        "Compare the parts of {topic} and examine the causes or patterns that structure it.",
        "Examine the structure of {topic} by breaking it into components and relating those parts.",
    ),
    "Evaluate": (
        "Evaluate the effectiveness of {topic} using explicit criteria of quality, validity, and suitability, and justify your judgment.",
        "Assess the strengths and limitations of {topic} against stated academic criteria and defend your conclusion.",
        "Critique {topic} considering validity, suitability, and trade-offs, and justify your judgment with evidence.",
    ),
    "Create": (
        "Design a new plan addressing {topic} under stated constraints to produce an original solution.",
        "Propose an original academic artifact or strategy related to {topic} to address a specified problem.",
        "Formulate a structured approach that constructs a new solution for {topic} under given constraints.",
    ),
}


@dataclass
class TopicInfo:
    topic: str
    content_tokens: frozenset[str]
    ok: bool
    reason: str = ""
    is_clause: bool = False


@dataclass
class ValidationResult:
    ok: bool
    quality_status: str
    failure_category: str = ""
    reasons: list[str] = field(default_factory=list)
    topic_overlap: float = 0.0
    trivial: bool = False
    meta: bool = False
    invalid_question: bool = False
    forbidden: bool = False


def canonical_level(raw: str) -> str | None:
    if raw is None:
        return None
    token = str(raw).strip()
    if token in BLOOM_LEVELS:
        return token
    return CANONICAL_LABELS.get(token.lower())


def stable_variant_index(key: str, n: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n


def extract_topic(question: str) -> TopicInfo:
    text = str(question).strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return TopicInfo("", frozenset(), False, "empty_question")
    remainder = LEADING_INSTRUCTION_RE.sub("", text).strip()
    remainder = COUNT_RE.sub("", remainder).strip()
    remainder = LEADING_VERB_RE.sub("", remainder).strip()
    remainder = COUNT_NOUN_PREFIX_RE.sub("", remainder).strip()
    remainder = TRAILING_INSTRUCTION_RE.sub("", remainder).strip()
    remainder = remainder.strip(" .,:;")
    remainder = re.sub(r"^(?:the|a|an)\s+(?:following|concept of|idea of)\s+", "", remainder, flags=re.I)
    remainder = remainder.strip(" .,:;")
    if remainder.lower().startswith(("how ", "why ", "what ", "which ", "when ", "where ")):
        remainder = re.sub(
            r"^(?:how|why|what|which|when|where)\s+(?:is|are|does|do|would|can|did)\s+",
            "",
            remainder,
            flags=re.I,
        ).strip()
    if len(remainder) < 4:
        return TopicInfo(remainder, frozenset(), False, "topic_too_short")
    tokens = frozenset(
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9_-]*", remainder.lower())
        if len(tok) > 2 and tok not in TOPIC_STOPWORDS
    )
    if len(tokens) < 2:
        return TopicInfo(remainder, tokens, False, "insufficient_content_tokens")
    remainder = remainder.rstrip("?.")
    words = remainder.split()
    if len(words) > 24:
        remainder = " ".join(words[:24])
        words = remainder.split()
    is_clause = bool(CLAUSE_MARKERS & {w.lower() for w in words}) or len(words) > 12
    if remainder:
        if is_clause:
            remainder = remainder[0].lower() + remainder[1:]
        elif remainder[0].islower():
            remainder = remainder[0].upper() + remainder[1:]
    return TopicInfo(remainder, tokens, True, "ok", is_clause=is_clause)


def source_is_usable(question: str, level: str | None) -> tuple[bool, str]:
    if canonical_level(level) is None:
        return False, "invalid_bloom_label"
    q = str(question).strip()
    words = q.split()
    if len(words) < 6:
        return False, "too_short"
    if len(words) > 70:
        return False, "too_long"
    if INCOMPLETE_RE.search(q):
        return False, "incomplete_or_template"
    if q.count("?") > 3:
        return False, "malformed"
    topic = extract_topic(q)
    if not topic.ok:
        return False, topic.reason
    return True, "ok"


def synthesize_rewrite(source_question: str, source_level: str, target_level: str, source_id: str) -> str:
    topic = extract_topic(source_question)
    if not topic.ok:
        raise ValueError(f"Cannot synthesize rewrite: {topic.reason}")
    bank = REWRITE_TEMPLATES_CLAUSE if topic.is_clause else REWRITE_TEMPLATES
    templates = bank[target_level]
    idx = stable_variant_index(f"{source_id}|{source_level}|{target_level}", len(templates))
    return templates[idx].format(topic=topic.topic)


def _topic_overlap(source: str, rewrite: str) -> float:
    src = extract_topic(source).content_tokens
    cand = frozenset(
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9_-]*", rewrite.lower())
        if len(tok) > 2 and tok not in TOPIC_STOPWORDS
    )
    if not src or not cand:
        return 0.0
    return len(src & cand) / len(src)


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _dominant_level_signals(rewrite: str) -> dict[str, bool]:
    return {
        "Remember": _has_any(rewrite, REMEMBER_MARKERS) and not _has_any(
            rewrite, APPLY_MARKERS + EVALUATE_MARKERS + CREATE_MARKERS + ANALYZE_MARKERS
        ),
        "Understand": _has_any(rewrite, UNDERSTAND_MARKERS)
        and not _has_any(rewrite, APPLY_MARKERS[0:4] + EVALUATE_MARKERS + CREATE_MARKERS),
        "Apply": _has_any(rewrite, APPLY_MARKERS) and not _has_any(rewrite, CREATE_MARKERS[:6] + EVALUATE_MARKERS[:4]),
        "Analyze": _has_any(rewrite, ANALYZE_MARKERS) and not _has_any(rewrite, CREATE_MARKERS[:6] + EVALUATE_MARKERS[:3]),
        "Evaluate": _has_any(rewrite, EVALUATE_MARKERS) and not _has_any(rewrite, CREATE_MARKERS[:6]),
        "Create": _has_any(rewrite, CREATE_MARKERS),
    }


def _task_structure_valid(rewrite: str, target_level: str) -> tuple[bool, str]:
    lowered = rewrite.lower()
    if target_level == "Remember":
        if _has_any(lowered, ("design a", "evaluate the", "analyze how", "given a concrete", "justify your judgment")):
            return False, "FORBIDDEN_COGNITIVE_OPERATION"
        if not _has_any(rewrite, REMEMBER_MARKERS):
            return False, "WRONG_TARGET_LEVEL"
        return True, "ok"
    if target_level == "Understand":
        if _has_any(lowered, ("design a", "evaluate the", "given a concrete academic scenario", "under stated constraints")):
            return False, "FORBIDDEN_COGNITIVE_OPERATION"
        if not _has_any(rewrite, UNDERSTAND_MARKERS):
            return False, "WRONG_TARGET_LEVEL"
        return True, "ok"
    if target_level == "Apply":
        if not _has_any(rewrite, APPLY_MARKERS):
            return False, "WRONG_TARGET_LEVEL"
        if not any(token in lowered for token in ("given", "scenario", "case", "concrete", "problem", "procedure", "determine", "apply", "use ")):
            return False, "WRONG_TARGET_LEVEL"
        return True, "ok"
    if target_level == "Analyze":
        if not any(token in lowered for token in ("analyze", "compare", "examine", "components", "relationship", "structure", "patterns", "causes")):
            return False, "WRONG_TARGET_LEVEL"
        return True, "ok"
    if target_level == "Evaluate":
        has_judgment = any(token in lowered for token in ("evaluate", "assess", "critique", "justify", "judge"))
        has_criteria = any(token in lowered for token in ("criteria", "effectiveness", "validity", "suitability", "strengths", "limitations", "trade-off", "evidence"))
        if not (has_judgment and has_criteria):
            return False, "WRONG_TARGET_LEVEL"
        return True, "ok"
    if target_level == "Create":
        has_create = any(token in lowered for token in ("design", "propose", "formulate", "construct", "develop", "create"))
        has_novelty = any(token in lowered for token in ("new", "original", "artifact", "plan", "strategy", "solution", "constraints"))
        if not (has_create and has_novelty):
            return False, "WRONG_TARGET_LEVEL"
        return True, "ok"
    return False, "OTHER"


def _is_trivial_verb_substitution(source: str, rewrite: str) -> bool:
    def strip_first_verb(text: str) -> str:
        cleaned = LEADING_INSTRUCTION_RE.sub("", text.strip())
        cleaned = LEADING_VERB_RE.sub("", cleaned).strip().lower()
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    src_core = strip_first_verb(source)
    rw_core = strip_first_verb(rewrite)
    if not src_core or not rw_core:
        return False
    if src_core == rw_core:
        return True
    if SequenceMatcher := __import__("difflib").SequenceMatcher:
        return SequenceMatcher(None, src_core, rw_core).ratio() >= 0.92
    return False


def _looks_like_question(text: str) -> bool:
    words = text.lower().split()
    if not words:
        return False
    first = words[0].rstrip(":,.")
    starters = {
        "define", "identify", "list", "name", "state", "explain", "describe",
        "summarize", "interpret", "classify", "discuss", "use", "apply", "solve",
        "determine", "calculate", "analyze", "compare", "examine", "evaluate",
        "assess", "justify", "critique", "design", "develop", "construct",
        "formulate", "propose", "create", "given", "what", "how", "why",
        "which", "who", "when", "where",
    }
    return text.endswith("?") or first in starters


def validate_rewrite(source_question: str, rewrite: str, target_level: str) -> ValidationResult:
    reasons: list[str] = []
    category = ""
    text = (rewrite or "").strip()
    if not text:
        return ValidationResult(False, "fail", "INVALID_QUESTION", ["empty_generation"], invalid_question=True)

    lowered = text.lower()
    meta = any(pat in lowered for pat in META_PATTERNS)
    if meta:
        reasons.append("meta_language")
        category = "META_RESPONSE"

    if any(pat in lowered for pat in ANSWER_PATTERNS):
        reasons.append("declarative_answer")
        category = category or "DECLARATIVE_ANSWER"

    if not _looks_like_question(text):
        reasons.append("not_student_facing_question")
        category = category or "INVALID_QUESTION"

    if len(text.split()) > 80:
        reasons.append("too_long")
        category = category or "INVALID_QUESTION"

    overlap = _topic_overlap(source_question, text)
    if overlap < TOPIC_OVERLAP_THRESHOLD:
        reasons.append(f"topic_overlap_{overlap:.2f}")
        category = category or "TOPIC_DRIFT"

    trivial = _is_trivial_verb_substitution(source_question, text)
    if trivial and canonical_level(target_level) != extract_topic(source_question).reason:
        # Trivial only matters when the student task did not actually change.
        src_core_same = trivial
        if src_core_same:
            reasons.append("trivial_verb_substitution")
            category = category or "TRIVIAL_VERB_SUBSTITUTION"

    task_ok, task_cat = _task_structure_valid(text, target_level)
    if not task_ok:
        reasons.append("task_structure_mismatch")
        category = category or task_cat

    multi = bool(
        re.search(
            r"\b(explain and design|define and evaluate|analyze and create|evaluate and design)\b",
            lowered,
        )
    )
    if multi:
        reasons.append("multi_level_task")
        category = category or "MULTI_LEVEL_TASK"

    ok = not reasons
    return ValidationResult(
        ok=ok,
        quality_status="pass" if ok else "fail",
        failure_category="" if ok else (category or "OTHER"),
        reasons=reasons,
        topic_overlap=overlap,
        trivial=trivial,
        meta=meta,
        invalid_question="not_student_facing_question" in reasons or "empty_generation" in reasons,
        forbidden=category == "FORBIDDEN_COGNITIVE_OPERATION",
    )


def all_source_target_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for source in BLOOM_LEVELS:
        for target in BLOOM_LEVELS:
            if source != target:
                pairs.append((source, target))
    return pairs
