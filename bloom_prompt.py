# ============================================================
# Teacher-side Bloom moderation (generative)
# Qwen2.5-1.5B-Instruct GGUF via llama.cpp
#
# Bloom *labels* come from the trained LoRA classifier (predict_bloom.py).
# GGUF generates only the higher-level rewrite; rationale is LoRA-aligned.
# ============================================================

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from multi_slm import resolve_slm_model_path
from predict_bloom import BLOOM_LABELS, build_prompt as build_classifier_prompt

IM_START = "<|im_start|>"
IM_END = "<|" + "im_end" + "|>"

# Deterministic transformation policies for each Bloom level
TARGET_TRANSFORMATION_POLICY = {
    "Remember": {
        "operation": "recall",
        "allowed": ["identify", "list", "name", "define", "recognize", "recall", "state"],
        "forbidden": ["design a", "evaluate the", "analyze how"],
        "template": "Recall a fact, term, component, function, or procedure.",
        "task": "The student recalls or identifies facts without explaining relationships or using a scenario.",
        "inappropriate": "Do not require explanation, a scenario-based solution, judgment, or design.",
        "examples": ["List the main functions of virtual memory."],
    },
    "Understand": {
        "operation": "comprehension",
        "allowed": ["explain", "describe", "summarize", "interpret", "classify", "discuss"],
        "forbidden": ["design a", "evaluate the", "solve the"],
        "template": "Interpret, summarize, classify, or describe an existing concept.",
        "task": "The student demonstrates comprehension or interpretation of the existing concept.",
        "inappropriate": "Do not require creating a solution, judging with criteria, or solving a concrete case.",
        "examples": ["Describe how virtual memory enables processes to use more memory than is physically available."],
    },
    "Apply": {
        "operation": "application",
        "allowed": ["apply", "use", "demonstrate", "implement", "solve", "execute"],
        "forbidden": ["define ", "list the", "evaluate the"],
        "template": "Use knowledge or a procedure in a concrete situation.",
        "task": "The student uses the concept to determine, solve, predict, or handle a specific case.",
        "inappropriate": "Do not merely request a definition or explanation; include a concrete situation or result to determine.",
        "examples": ["Given a page-reference scenario, determine how virtual memory handles a page not in physical memory."],
    },
    "Analyze": {
        "operation": "analysis",
        "allowed": ["analyze", "compare", "differentiate", "examine", "investigate", "categorize"],
        "forbidden": ["design a", "create a", "evaluate the"],
        "template": "Examine parts, relationships, differences, causes, or structure.",
        "task": "The student breaks existing information into parts and examines their relationships.",
        "inappropriate": "Do not merely define the concept, judge it by criteria, or design a new artifact.",
        "examples": ["Analyze how paging and page faults interact in a virtual memory system."],
    },
    "Evaluate": {
        "operation": "judgment",
        "allowed": ["evaluate", "assess", "justify", "critique", "defend", "judge"],
        "forbidden": ["design a", "create a"],
        "template": "Make and justify a judgment using criteria, evidence, or trade-offs.",
        "task": "The student makes a judgment using explicit criteria, evidence, or trade-offs.",
        "inappropriate": "Do not only describe the concept or ask the student to build a new artifact.",
        "examples": ["Evaluate virtual memory considering performance, memory utilization, and page-fault overhead."],
    },
    "Create": {
        "operation": "creation",
        "allowed": ["design", "develop", "construct", "formulate", "propose", "create"],
        "forbidden": [],
        "template": "Produce a novel artifact, plan, model, strategy, or solution with constraints.",
        "task": "The student designs or produces a new artifact, plan, model, or solution.",
        "inappropriate": "Do not only ask for recall, explanation, or a judgment without producing something new.",
        "examples": ["Design a virtual memory strategy that minimizes page-fault overhead under stated constraints."],
    },
}

# Special transformation mappings for common cross-level transformations
SPECIAL_TRANSFORMATIONS = {
    ("Create", "Understand"): {
        "remove_patterns": ["design", "develop", "construct", "build", "create", "formulate", "implement", "propose", "produce"],
        "replace_with": ["explain", "describe", "summarize", "interpret", "classify"],
        "object_change": "Change from 'how to create' to 'existing components/functions'",
        "examples": [
            ("Design X", "Explain the main components and functions of X"),
            ("Create X", "Describe the key features and purpose of X"),
            ("Develop X", "Explain how X works and its main functions"),
        ],
    },
}

BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

_LEVEL_GUIDANCE: dict[str, dict[str, str]] = {
    "Remember": {
        "depth": "recall of facts, terms, or procedures without explaining relationships",
        "rewrite": "Require explanation, comparison, or use of the recalled knowledge",
    },
    "Understand": {
        "depth": "comprehension, interpretation, or summarization of ideas",
        "rewrite": "Require application to a scenario, prediction, or worked example",
    },
    "Apply": {
        "depth": "using knowledge or procedures in a concrete situation",
        "rewrite": "Require analysis of trade-offs, breakdown of components, or diagnosis",
    },
    "Analyze": {
        "depth": "breaking a problem into parts and examining relationships or causes",
        "rewrite": "Require evaluation with criteria, justification, or critique",
    },
    "Evaluate": {
        "depth": "judging quality, validity, or trade-offs using explicit criteria",
        "rewrite": "Require designing, proposing, or synthesizing a novel solution",
    },
    "Create": {
        "depth": "designing or producing a new artifact, plan, or argument",
        "rewrite": "Extend the task with constraints, audience, or integration across topics",
    },
}

_LLM = None

_LONG_TO_SHORT = {
    "Remembering": "Remember",
    "Understanding": "Understand",
    "Applying": "Apply",
    "Analyzing": "Analyze",
    "Evaluating": "Evaluate",
    "Creating": "Create",
    "Knowledge": "Remember",
    "Remember": "Remember",
    "Recall": "Remember",
    "Comprehension": "Understand",
    "Understand": "Understand",
    "Application": "Apply",
    "Apply": "Apply",
    "Analysis": "Analyze",
    "Analyze": "Analyze",
    "Evaluation": "Evaluate",
    "Evaluate": "Evaluate",
    "Synthesis": "Create",
    "Create": "Create",
}


@dataclass
class BloomModerationResult:
    question: str
    lora_level: str
    lora_confidence: float
    target_higher_level: str
    bloom_level: str
    reason: str
    higher_level_rewrite: str
    raw: str = ""
    raw_output: str = ""
    cleaned_output: str = ""
    backend: str = ""
    latency_s: Optional[float] = None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "lora_level": self.lora_level,
            "lora_confidence": self.lora_confidence,
            "target_higher_level": self.target_higher_level,
            "bloom_level": self.bloom_level,
            "reason": self.reason,
            "higher_level_rewrite": self.higher_level_rewrite,
            "raw": self.raw,
            "raw_output": self.raw_output or self.raw,
            "cleaned_output": self.cleaned_output or self.higher_level_rewrite,
            "backend": self.backend,
            "latency_s": self.latency_s,
            "error": self.error,
        }


def _canonical_bloom_label(raw: str, *, fallback: str = "Understand") -> str:
    token = (raw or "").strip().split("\n")[0].strip().rstrip(".")
    if ":" in token:
        token = token.split(":", 1)[-1].strip()
    short = _LONG_TO_SHORT.get(token, token)
    if short in BLOOM_LABELS:
        return short
    for label in BLOOM_LABELS:
        if label.lower() in token.lower():
            return label
    return fallback


def _next_bloom_level(level: str) -> str:
    short = _canonical_bloom_label(level, fallback="Understand")
    if short not in BLOOM_ORDER:
        return "Analyze"
    idx = BLOOM_ORDER.index(short)
    return BLOOM_ORDER[min(idx + 1, len(BLOOM_ORDER) - 1)]


def next_bloom_level(level: str) -> str:
    return _next_bloom_level(level)


def build_classifier_aligned_reason(
    lora_level: str,
    *,
    confidence: float,
    probabilities: dict[str, float] | None = None,
) -> str:
    """Deterministic rationale aligned with train_qwen_bloom / predict_bloom.py."""
    level = _canonical_bloom_label(lora_level)
    guide = _LEVEL_GUIDANCE.get(level, _LEVEL_GUIDANCE["Understand"])
    runner_up = ""
    if probabilities:
        ordered = sorted(
            ((label, float(probabilities.get(label, 0.0))) for label in BLOOM_LABELS),
            key=lambda item: item[1],
            reverse=True,
        )
        if len(ordered) >= 2 and ordered[1][1] >= 0.12:
            runner_up = (
                f" The next most likely level was {ordered[1][0]} "
                f"({ordered[1][1]:.0%}), but depth of reasoning favors {level}."
            )
    return (
        f"The LoRA classifier ({confidence:.0%} confidence) placed this item at **{level}** "
        f"because it primarily requires {guide['depth']}. "
        f"{runner_up}"
    )


def build_rewrite_prompt(
    question: str,
    *,
    lora_level: str,
    target_level: str,
) -> str:
    """GGUF prompt for higher-order rewrite only (label is fixed by LoRA)."""
    level = _canonical_bloom_label(lora_level)
    guide = _LEVEL_GUIDANCE.get(level, _LEVEL_GUIDANCE["Understand"])
    classifier_block = build_classifier_prompt(question)
    return f"""{IM_START}system
You are an expert exam-question editor using Bloom's Taxonomy.
The Bloom level is already decided by a trained classifier: {level}.
Do NOT re-classify. Do NOT change the topic.

Task: rewrite the question so it requires **{target_level}**-level thinking
(one stage higher than {level}). {guide['rewrite']}.
Focus on reasoning depth, not verb swapping (avoid only changing "define" to "explain").

Output ONLY the rewritten question as a single exam-style sentence or short paragraph.
No labels, no preamble, no bullet list.
{IM_END}
{IM_START}user
Classifier context (for alignment only):
{classifier_block}

Rewrite this question for {target_level}-level cognition:
{question.strip()}
{IM_END}
{IM_START}assistant
""".strip()


def _is_trivial_transformation(original: str, rewrite: str, target_level: str) -> tuple[bool, str]:
    """
    Check if the rewrite is a trivial transformation that doesn't meaningfully change cognitive demand.
    
    Rejects:
    - Identical questions after normalization
    - Surface-level changes (punctuation, capitalization, filler words)
    - "Explain" → "How would you explain" wrapping without elaboration
    - Cognitive-demand-preserving transformations
    
    Relies on semantic and task validators for cognitive validation to avoid false positives.
    Only catches obvious trivial wrapping cases here.
    
    Args:
        original: The original question
        rewrite: The generated rewrite
        target_level: The target Bloom level
    
    Returns:
        tuple: (is_trivial, reason)
    """
    import re
    
    # Normalize both questions for comparison
    def normalize(q: str) -> str:
        q = q.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)  # Remove punctuation
        q = re.sub(r'\s+', ' ', q)  # Normalize whitespace
        return q
    
    norm_original = normalize(original)
    norm_rewrite = normalize(rewrite)
    
    # Check for identity
    if norm_original == norm_rewrite:
        return True, "Rewrite is identical to original after normalization"
    
    # Check for trivial "How would you explain" wrapping for Understand
    if target_level == "Understand":
        # Pattern: "explain X" → "how would you explain X"
        if norm_original.startswith("explain ") and norm_rewrite.startswith("how would you explain "):
            original_content = norm_original.replace("explain ", "", 1)
            rewrite_content = norm_rewrite.replace("how would you explain ", "", 1)
            if original_content == rewrite_content:
                return True, "Trivial transformation: 'explain' to 'how would you explain' without elaboration"
        
        # Pattern: "what is X" → "how would you explain what X is"
        if norm_original.startswith("what is ") and norm_rewrite.startswith("how would you explain what"):
            return True, "Trivial transformation: wrapping 'what is' with 'how would you explain'"
        
        # Check for elaboration beyond simple wrapping
        if norm_rewrite.startswith("how would you explain "):
            rewrite_content = norm_rewrite.replace("how would you explain ", "", 1)
            original_content = norm_original.replace("explain ", "", 1).replace("what is ", "", 1)
            
            # If the content is essentially the same, it's trivial
            if rewrite_content == original_content or rewrite_content.startswith(original_content):
                return True, "Understand transformation lacks meaningful elaboration"
    
    # Check for trivial verb wrapping for other levels
    trivial_wrappings = {
        "Apply": [("apply ", "how would you apply ")],
        "Analyze": [("analyze ", "how would you analyze ")],
        "Evaluate": [("evaluate ", "how would you evaluate ")],
        "Create": [("design ", "how would you design ")],
    }
    
    if target_level in trivial_wrappings:
        for original_prefix, rewrite_prefix in trivial_wrappings[target_level]:
            if norm_original.startswith(original_prefix) and norm_rewrite.startswith(rewrite_prefix):
                original_content = norm_original.replace(original_prefix, "", 1)
                rewrite_content = norm_rewrite.replace(rewrite_prefix, "", 1)
                if original_content == rewrite_content:
                    return True, f"Trivial transformation: '{original_prefix}' to '{rewrite_prefix}' without elaboration"
    
    # For other cognitive validation, rely on semantic and task validators
    # to avoid false positives on valid transformations that don't use expected keywords
    
    return False, ""


def _clean_rewrite(text: str) -> str:
    cleaned = (text or "").strip()
    # Remove common prefixes that aren't part of the question
    cleaned = re.sub(r"(?im)^(bloom level|reason|higher[- ]level rewrite|rewrite|question|answer|the question is|the rewritten question is)\s*:\s*", "", cleaned)
    # Remove explanatory prefixes
    cleaned = re.sub(r"(?im)^(the|a|an)\s+(rewritten|new|modified|revised)\s+(question|version|answer)\s*(is|:|says|states)\s*", "", cleaned)
    # Remove sentence fragments that are explanations
    cleaned = re.sub(r"(?im)^(this|that|the)\s+(analysis|explanation|description|rewrite|version)\s+(involves|requires|means|includes)\s*", "", cleaned)
    # Remove introductory phrases
    cleaned = re.sub(r"(?im)^(to|in order to|for the purpose of)\s+", "", cleaned)
    cleaned = cleaned.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    
    # Ensure it ends with appropriate punctuation
    if cleaned:
        if not cleaned.endswith('?') and not cleaned.endswith('.'):
            # If it contains question words, add question mark
            if any(word in cleaned.lower() for word in ['what', 'how', 'why', 'which', 'who', 'when', 'where', 'explain', 'describe', 'analyze', 'evaluate', 'design', 'create', 'identify', 'list', 'compare']):
                cleaned += '?'
            else:
                cleaned += '.'
    
    return cleaned


def _get_llm():
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama

        model_path = resolve_slm_model_path("bloom_moderation")
        _LLM = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
    return _LLM


def _generate_rewrite(prompt: str) -> tuple[str, str, str, float]:
    try:
        from qwen_gguf_cli import QwenGgufCliGenerator

        gen = QwenGgufCliGenerator.for_task(
            "bloom_moderation",
            max_tokens=180,
            ctx_size=2048,
            threads=4,
        )
        out = gen.generate_prompt(prompt)
        return out.answer, _clean_rewrite(out.answer), out.backend, float(out.elapsed_s)
    except Exception:
        llm = _get_llm()
        output = llm(
            prompt,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            max_tokens=180,
            stop=[IM_END, IM_START],
        )
        text = output["choices"][0]["text"].strip()
        return text, _clean_rewrite(text), "llama-cpp-python", 0.0


def moderate_bloom_question(
    question: str,
    *,
    lora_level: str,
    lora_confidence: float = 0.0,
    probabilities: dict[str, float] | None = None,
    rewrite_generator=None,
) -> BloomModerationResult:
    """LoRA label + aligned rationale + GGUF higher-order rewrite."""
    short_level = _canonical_bloom_label(lora_level)
    target_level = _next_bloom_level(short_level)
    reason = build_classifier_aligned_reason(
        short_level,
        confidence=lora_confidence,
        probabilities=probabilities,
    )
    rewrite = ""
    raw = ""
    backend = ""
    latency_s: Optional[float] = None
    error = ""

    try:
        prompt = build_rewrite_prompt(
            question,
            lora_level=short_level,
            target_level=target_level,
        )
        if rewrite_generator is None:
            raw, rewrite, backend, latency_s = _generate_rewrite(prompt)
        else:
            raw, backend, latency_s = rewrite_generator(prompt)
            rewrite = _clean_rewrite(raw)
        if not rewrite or len(rewrite.split()) < 6:
            raise RuntimeError("rewrite_too_short")
    except Exception as exc:
        error = str(exc)
        guide = _LEVEL_GUIDANCE.get(short_level, _LEVEL_GUIDANCE["Understand"])
        rewrite = (
            f"[Auto-rewrite unavailable] Elevate to {target_level}: {guide['rewrite']}. "
            f"Original: {question.strip()}"
        )

    return BloomModerationResult(
        question=question,
        lora_level=short_level,
        lora_confidence=float(lora_confidence),
        target_higher_level=target_level,
        bloom_level=short_level,
        reason=reason,
        higher_level_rewrite=rewrite,
        raw=raw,
        raw_output=raw,
        cleaned_output=rewrite,
        backend=backend,
        latency_s=latency_s,
        error=error,
    )


def build_moderation_prompt(question: str) -> str:
    """Prompt for linguistic quality improvement without changing Bloom level."""
    return f"""{IM_START}system
You are an expert academic editor. Improve the linguistic and formal quality of exam questions.

Task: rewrite the question to improve:
- grammar
- spelling
- punctuation
- sentence structure
- academic/formal wording
- readability

CRITICAL: Do NOT change the cognitive level or difficulty. Preserve the original meaning, subject, topic, and academic intent exactly.

Output ONLY the improved question as a single exam-style sentence or short paragraph.
No labels, no preamble, no bullet list, no explanation of changes.
{IM_END}
{IM_START}user
Improve the linguistic quality of this exam question (do not change its cognitive level):
{question.strip()}
{IM_END}
{IM_START}assistant
""".strip()

def build_targeted_rewrite_prompt(
    question: str,
    *,
    target_level: str,
    previous_failure: str = "",
) -> str:
    """Build a concise target-specific prompt for the small local model."""
    policy = TARGET_TRANSFORMATION_POLICY.get(target_level, TARGET_TRANSFORMATION_POLICY["Understand"])
    examples = "\n".join(f"- {example}" for example in policy["examples"])
    retry_instruction = f"\nCorrection: {previous_failure}\n" if previous_failure else ""
    return f"""{IM_START}system
You edit one exam question for Bloom's Taxonomy.

TARGET BLOOM LEVEL: {target_level}
TASK THE QUESTION MUST REQUIRE: {policy["task"]}

Write ONE student-facing exam question, not an answer or explanation. Preserve
the original topic and technical entities; change the required cognitive action.
Do not mention Bloom, the rewrite, or these instructions. 10-40 words.

GOOD OUTPUT EXAMPLE FOR THIS TARGET:
{examples}
{retry_instruction}
{IM_END}
{IM_START}user
{question.strip()}
{IM_END}
{IM_START}assistant
""".strip()

def _validate_output_format(question: str) -> tuple[bool, str]:
    """
    Validate that output is a student-facing question, not meta-explanation.
    
    Returns:
        tuple: (is_valid, reason)
    """
    question_lower = question.lower().strip()
    
    # Reject meta-language patterns
    meta_patterns = [
        "this question asks",
        "the rewritten question",
        "to understand",
        "this requires",
        "the analysis involves",
        "the student must",
        "this rewrite",
        "the answer is",
        "this is a",
        "the purpose is",
    ]
    
    for pattern in meta_patterns:
        if pattern in question_lower:
            return False, f"Contains meta-language: '{pattern}'"
    
    words = question_lower.split()
    if words and words[0] in {"this", "to"}:
        return False, "Begins with explanatory meta-language"
    
    # Reject if it's too long (likely an explanation)
    if len(question.split()) > 40:
        return False, "Too long - likely an explanation not a question"
    
    # Reject if it contains multiple sentences (likely explanation)
    if question.count('.') > 1 or question.count('!') > 0:
        return False, "Multiple sentences - likely an explanation"
    
    # Imperative exam prompts are valid questions even without a question mark.
    # This checks only grammatical prompt form, never Bloom classification.
    question_words = {"what", "how", "why", "which", "who", "when", "where", "given", "determine", "calculate", "show", "predict", "outline", "distinguish", "recommend", "propose"}
    imperative_starters = {"define", "identify", "list", "name", "state", "explain", "describe", "summarize", "interpret", "classify", "use", "solve", "demonstrate", "analyze", "compare", "examine", "evaluate", "assess", "justify", "critique", "design", "develop", "construct", "create", "formulate"}
    has_question_structure = (
        question.endswith("?")
        or bool(words and words[0].rstrip(":,.") in question_words)
        or bool(words and words[0].rstrip(":,.") in imperative_starters)
    )
    
    if not has_question_structure:
        return False, "Lacks question structure"
    
    return True, "Output format valid"

def _semantic_cognitive_check_deterministic(question: str, target_level: str) -> tuple[bool, str]:
    """
    Perform semantic check using deterministic transformation policy.
    
    Returns:
        tuple: (is_valid, reason)
    """
    policy = TARGET_TRANSFORMATION_POLICY.get(target_level)
    if not policy:
        return True, "No policy defined - skipping check"
    
    question_lower = question.lower()
    
    # This is a contradiction guardrail, not a Bloom classifier. A valid task
    # need not use one of the policy's preferred verbs.
    for forbidden in policy["forbidden"]:
        # Check for "how to [forbidden]" patterns
        if f"how to {forbidden}" in question_lower:
            return False, f"Contains forbidden pattern: 'how to {forbidden}'"
        if forbidden in question_lower:
            return False, f"Contains an obvious contradictory task: '{forbidden}'"
    
    return True, "Semantic cognitive check passed"

def moderate_question_linguistic(question: str) -> str:
    """Improve linguistic quality without changing cognitive level."""
    prompt = build_moderation_prompt(question)
    try:
        from qwen_gguf_cli import QwenGgufCliGenerator

        gen = QwenGgufCliGenerator.for_task(
            "bloom_moderation",
            max_tokens=180,
            ctx_size=2048,
            threads=4,
        )
        out = gen.generate_prompt(prompt)
        return _clean_rewrite(out.answer)
    except Exception:
        llm = _get_llm()
        output = llm(
            prompt,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            max_tokens=180,
            stop=[IM_END, IM_START],
        )
        text = output["choices"][0]["text"].strip()
        return _clean_rewrite(text)

def _analyze_cognitive_task_structure(question: str, target_level: str) -> tuple[bool, str]:
    """
    Analyze the cognitive task structure to determine if the dominant operation matches the target level.
    
    This function examines what the STUDENT must DO, not just which words appear in the sentence.
    
    Returns:
        tuple: (is_valid, reason)
    """
    question_lower = question.lower()
    
    # Define dominant cognitive patterns for each level
    # These patterns focus on the TASK demanded from the student, not subject matter
    dominant_patterns = {
        "Remember": [
            r"\b(identify|list|name|define|recognize|recall|state|label)\b.*\s+(the|a|an|what|which|who)",
            r"\b(identify|list|name|define|recognize|recall|state|label)\b.*\s+(components|parts|elements|steps|stages|types|kinds|examples)",
        ],
        "Understand": [
            r"\b(explain|describe|summarize|interpret|classify|discuss)\b.*\s+(the|a|an|how|why|what)",
            r"\b(explain|describe|summarize|interpret|classify|discuss)\b.*\s+(principles|concepts|ideas|meaning|relationship|function)",
        ],
        "Apply": [
            r"\b(apply|use|demonstrate|implement|solve|execute)\b.*\s+(to|for|in|with)",
            r"\b(apply|use|demonstrate|implement|solve|execute)\b.*\s+(problem|situation|case|scenario|example)",
        ],
        "Analyze": [
            r"\b(analyze|compare|differentiate|examine|investigate|categorize)\b.*\s+(the|a|an|and|between|with)",
            r"\b(analyze|compare|differentiate|examine|investigate|categorize)\b.*\s+(relationship|differences|similarities|patterns|structure|components)",
        ],
        "Evaluate": [
            r"\b(evaluate|assess|justify|critique|defend|judge)\b.*\s+(the|a|an|using|based on|according to)",
            r"\b(evaluate|assess|justify|critique|defend|judge)\b.*\s+(effectiveness|quality|validity|merits|strengths|weaknesses)",
        ],
        "Create": [
            r"\b(design|develop|construct|formulate|propose|create)\b.*\s+(a|an|the)",
            r"\b(design|develop|construct|formulate|propose|create)\b.*\s+(system|model|plan|solution|product|application|structure)",
        ],
    }
    
    # Define problematic task structures where the dominant operation doesn't match
    # These focus on multi-verb constructions where the actual task is higher than the apparent level
    problematic_structures = {
        "Understand": [
            r"\b(explain)\b.*\s+(how to design|how to develop|how to create|how to construct)",
            r"\b(describe)\b.*\s+(how to design|how to develop|how to create|how to construct)",
            r"\b(explain and design|explain and develop|explain and create|explain and construct)\b",
            r"\b(design and explain|develop and explain|create and explain|construct and explain)\b",
        ],
        "Analyze": [
            r"\b(analyze and design|analyze and develop|analyze and create|analyze and construct)\b",
            r"\b(compare and design|compare and develop|compare and create)\b",
            r"\b(design and analyze|develop and analyze|create and analyze)\b",
        ],
        "Evaluate": [
            r"\b(explain and evaluate|describe and evaluate)\b",
            r"\b(evaluate and explain|evaluate and describe)\b",
        ],
        "Remember": [
            r"\b(define and design|define and create|define and develop|define and construct)\b",
            r"\b(design and define|create and define|develop and define)\b",
        ],
        "Apply": [
            r"\b(apply and design|apply and create|apply and develop)\b",
            r"\b(design and apply|create and apply|develop and apply)\b",
        ],
    }
    
    import re
    
    # Check for problematic structures first
    if target_level in problematic_structures:
        for pattern in problematic_structures[target_level]:
            if re.search(pattern, question_lower):
                return False, f"Question contains problematic construction that doesn't match {target_level} cognitive task."
    
    # Check if the question contains dominant patterns for the target level
    if target_level in dominant_patterns:
        has_dominant_pattern = False
        for pattern in dominant_patterns[target_level]:
            if re.search(pattern, question_lower):
                has_dominant_pattern = True
                break
        
        if not has_dominant_pattern:
            return False, f"Question lacks clear dominant cognitive pattern for {target_level}."
    
    return True, "Task structure analysis passed."

_TOPIC_STOPWORDS = frozenset("a an the what how why which who when where explain describe list name define identify state recall main purpose role concept concepts question given determine would should could of in on for to with and or is are be this that it its than more under using".split())
SEMANTIC_SIMILARITY_THRESHOLD = float(os.environ.get("BLOOM_SEMANTIC_SIMILARITY_THRESHOLD", "0.20"))


def _topic_terms(text: str) -> set[str]:
    """Extract likely subject terms; this is intentionally small and offline."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in _TOPIC_STOPWORDS}


def semantic_preservation_check(original: str, rewrite: str, *, threshold: float = SEMANTIC_SIMILARITY_THRESHOLD) -> tuple[bool, float, str]:
    """Guard against topic drift without demanding verb or full-string overlap."""
    source = _topic_terms(original)
    candidate = _topic_terms(rewrite)
    if not source or not candidate:
        return False, 0.0, "Could not identify topic terms"
    overlap = source & candidate
    # Source terms are a better denominator: rewrites legitimately add scenarios.
    score = len(overlap) / len(source)
    if not overlap:
        return False, score, f"Topic drift: none of {sorted(source)} was retained"
    if score < threshold:
        return False, score, f"Topic overlap {score:.0%} is below configured threshold {threshold:.0%}"
    return True, score, f"Topic overlap {score:.0%}: {', '.join(sorted(overlap))}"


def _retry_instruction(target_level: str, failure: str) -> str:
    if failure.startswith("Classified as"):
        instructions = {
            "Understand": "Require comprehension or interpretation of the existing concept; do not ask for design, solution, or judgment.",
            "Apply": "Require the student to use the concept in a concrete situation and determine a result; do not merely ask for an explanation.",
            "Analyze": "Require examination of relationships, differences, causes, components, or structure; do not merely ask for a description.",
        }
        return instructions.get(target_level, f"Make the student's task clearly {target_level}-level.")
    return failure


def _validate_generated_candidate(question: str, rewrite: str, target_level: str, predictor) -> dict:
    record = {"cleaned_output": rewrite, "format_valid": False, "semantic_similarity": 0.0,
              "deterministic_check": "not run", "classifier_prediction": "", "classifier_confidence": 0.0,
              "final_validation": False, "failure_reason": ""}
    valid, reason = _validate_output_format(rewrite)
    record["format_valid"] = valid
    if not valid:
        record["failure_reason"] = reason
        return record
    valid, similarity, reason = semantic_preservation_check(question, rewrite)
    record["semantic_similarity"] = similarity
    if not valid:
        record["failure_reason"] = reason
        return record
    valid, reason = _semantic_cognitive_check_deterministic(rewrite, target_level)
    record["deterministic_check"] = reason
    if not valid:
        record["failure_reason"] = reason
        return record
    try:
        result = predictor.predict(rewrite)
        predicted = _canonical_bloom_label(result["prediction"])
        confidence = float(result.get("confidence") or 0.0)
    except Exception as exc:
        record["failure_reason"] = f"Classifier error: {exc}"
        return record
    record["classifier_prediction"] = predicted
    record["classifier_confidence"] = confidence
    if predicted != _canonical_bloom_label(target_level):
        record["failure_reason"] = f"Classified as {predicted}; generate a {target_level} task."
    elif confidence < 0.6:
        record["failure_reason"] = f"Classifier confidence {confidence:.0%} is below 60%."
    else:
        record["final_validation"] = True
        record["failure_reason"] = ""
    return record


def rewrite_to_target_level(question: str, target_level: str) -> tuple[str, bool, str]:
    """Generate at most three candidates; only a classifier-validated one succeeds."""
    target_level = _canonical_bloom_label(target_level)
    from predict_bloom import QwenBloomPredictor
    predictor = QwenBloomPredictor()
    previous_failure = ""
    for _attempt in range(1, 4):
        prompt = build_targeted_rewrite_prompt(question, target_level=target_level, previous_failure=previous_failure)
        try:
            raw_output, rewrite, _backend, _latency = _generate_rewrite(prompt)
        except Exception as exc:
            previous_failure = f"Generation error: {exc}"
            continue
        record = _validate_generated_candidate(question, rewrite, target_level, predictor)
        if record["final_validation"]:
            return rewrite, True, ""
        previous_failure = _retry_instruction(target_level, record["failure_reason"])
    return "", False, f"Could not generate a validated {target_level} rewrite after 3 attempts: {previous_failure}"


def run_target_rewrite_diagnostic(question: str = "Explain what virtual memory is.", *, samples_per_target: int = 5) -> list[dict]:
    """Capture raw evidence for every target without stopping after failures."""
    from predict_bloom import QwenBloomPredictor
    predictor = QwenBloomPredictor()
    records: list[dict] = []
    for target_level in BLOOM_ORDER:
        for attempt in range(1, samples_per_target + 1):
            prompt = build_targeted_rewrite_prompt(question, target_level=target_level)
            record = {"target_level": target_level, "attempt": attempt, "raw_output": "", "cleaned_output": ""}
            try:
                raw_output, rewrite, _backend, _latency = _generate_rewrite(prompt)
                record["raw_output"] = raw_output
                record.update(_validate_generated_candidate(question, rewrite, target_level, predictor))
            except Exception as exc:
                record.update({"format_valid": False, "semantic_similarity": 0.0, "deterministic_check": "not run", "classifier_prediction": "", "classifier_confidence": 0.0, "final_validation": False, "failure_reason": f"Generation error: {exc}"})
            records.append(record)
    return records


def write_target_rewrite_diagnostic(path: str | Path, question: str = "Explain what virtual memory is.", *, samples_per_target: int = 5) -> list[dict]:
    """Run the diagnostic and persist raw evidence as UTF-8 JSON for review."""
    import json
    records = run_target_rewrite_diagnostic(question, samples_per_target=samples_per_target)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records

if __name__ == "__main__":
    print("Teacher Bloom moderation — label: predict_bloom.py; rewrite: GGUF")
    
    # Specific test cases from requirements
    specific_tests = [
        ("Explain the design principles of a banking system.", "Understand", "VALID Understand - design in subject matter"),
        ("Design a basic banking system.", "Understand", "INVALID Understand - design as student task"),
        ("Design and explain a banking system.", "Understand", "INVALID Understand - multi-verb with design"),
        ("Analyze the design of a banking system and identify its strengths and weaknesses.", "Analyze", "VALID Analyze - design in subject matter"),
        ("Evaluate the effectiveness of the banking system using criteria such as security, scalability, and cost.", "Evaluate", "VALID Evaluate - with criteria"),
        ("Design a banking system that incorporates secure authentication and transaction processing.", "Create", "VALID Create - design as student task"),
        ("Explain and justify whether the banking system is effective using explicit criteria.", "Evaluate", "VALID Evaluate - explain and justify with criteria"),
    ]
    
    print("\n=== SPECIFIC VALIDATION TESTS ===\n")
    
    for question, target_level, test_description in specific_tests:
        print(f"Test: {test_description}")
        print(f"Input: {question}")
        print(f"Target: {target_level}")
        
        # Test cognitive task structure analysis
        try:
            task_valid, task_reason = _analyze_cognitive_task_structure(question, target_level)
            print(f"Task Structure: {'PASS' if task_valid else 'FAIL'} - {task_reason}")
        except Exception as e:
            print(f"Task Structure Error: {str(e)}")
        
        print()
    
    # Cross-level transformation tests
    cross_level_tests = [
        ("Design a level one banking system.", "Understand", "C6 → C2"),
        ("Design a level one banking system.", "Remember", "C6 → C1"),
        ("Design a level one banking system.", "Apply", "C6 → C3"),
        ("Explain the basic components of a banking system.", "Evaluate", "C2 → C5"),
        ("Apply the banking concept to solve a problem.", "Analyze", "C3 → C4"),
        ("Analyze the banking system architecture.", "Evaluate", "C4 → C5"),
        ("Evaluate the effectiveness of the banking system.", "Create", "C5 → C6"),
        ("Evaluate the effectiveness of the banking system.", "Remember", "C5 → C1"),
    ]
    
    print("\n=== CROSS-LEVEL TRANSFORMATION TESTS ===\n")
    
    for question, target_level, test_name in cross_level_tests:
        print(f"Test: {test_name}")
        print(f"Input: {question}")
        print(f"Target: {target_level}")
        
        try:
            rewrite, success, error = rewrite_to_target_level(question, target_level)
            
            if success:
                print(f"✓ SUCCESS: {rewrite}")
                
                # Verify with classifier
                from predict_bloom import QwenBloomPredictor
                predictor = QwenBloomPredictor()
                validation = predictor.predict(rewrite)
                predicted = _canonical_bloom_label(validation["prediction"])
                confidence = validation.get("confidence", 0.0)
                
                # Verify task structure
                task_valid, task_reason = _analyze_cognitive_task_structure(rewrite, target_level)
                
                if predicted == _canonical_bloom_label(target_level) and task_valid:
                    print(f"  Full Validation: PASS (predicted: {predicted}, confidence: {confidence:.0%}, task: valid)")
                else:
                    print(f"  Validation: PARTIAL (predicted: {predicted}, target: {target_level}, confidence: {confidence:.0%}, task: {'valid' if task_valid else 'invalid'})")
            else:
                print(f"✗ FAILED: {error}")
                
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
        
        print()
    
    # Interactive mode
    print("=== INTERACTIVE MODE ===")
    while True:
        q = input("\nEnter academic question (or 'exit'): ")
        if q.lower() == "exit":
            break
        from predict_bloom import QwenBloomPredictor

        pred = QwenBloomPredictor().predict(q)
        mod = moderate_bloom_question(
            q,
            lora_level=pred["prediction"],
            lora_confidence=pred["confidence"],
            probabilities=pred.get("probabilities"),
        )
        print("\nLoRA:", pred["prediction"], f"(conf={pred['confidence']})")
        print("Reason:", mod.reason)
        print("Rewrite:", mod.higher_level_rewrite)
        if mod.error:
            print("Rewrite warning:", mod.error)
