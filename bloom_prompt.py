# ============================================================
# Teacher-side Bloom moderation (generative)
# Qwen2.5-1.5B-Instruct GGUF via llama.cpp
#
# Bloom *labels* come from the trained LoRA classifier (predict_bloom.py).
# GGUF generates only the higher-level rewrite; rationale is LoRA-aligned.
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass
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
        "forbidden": ["explain", "describe", "apply", "analyze", "evaluate", "design", "create", "develop", "construct", "implement"],
        "template": "[Identify/List/Name/Define] + [topic/components/concepts]",
        "task": "Recall or identify facts without explanation or application",
    },
    "Understand": {
        "operation": "comprehension",
        "allowed": ["explain", "describe", "summarize", "interpret", "classify", "discuss"],
        "forbidden": ["design", "develop", "construct", "implement", "create", "build", "solve", "evaluate", "justify", "analyze"],
        "template": "[Explain/Describe/Summarize] + [topic/components/functions/purpose]",
        "task": "Explain or describe existing concepts without creating/evaluating",
    },
    "Apply": {
        "operation": "application",
        "allowed": ["apply", "use", "demonstrate", "implement", "solve", "execute"],
        "forbidden": ["explain", "describe", "identify", "list", "name", "define"],
        "template": "[Apply/Use/Demonstrate] + [concept] + [specific scenario/problem]",
        "task": "Apply knowledge to a specific situation or problem",
    },
    "Analyze": {
        "operation": "analysis",
        "allowed": ["analyze", "compare", "differentiate", "examine", "investigate", "categorize"],
        "forbidden": ["design", "develop", "construct", "implement", "create", "formulate", "propose"],
        "template": "[Analyze/Compare/Examine] + [components/relationships/causes/differences]",
        "task": "Examine structure, relationships, or components of existing information",
    },
    "Evaluate": {
        "operation": "judgment",
        "allowed": ["evaluate", "assess", "justify", "critique", "defend", "judge"],
        "forbidden": ["explain", "describe", "list", "identify", "name", "define"],
        "template": "[Evaluate/Assess/Justify] + [subject] + using [explicit criteria/evidence]",
        "task": "Make judgment using explicit criteria or evidence",
    },
    "Create": {
        "operation": "creation",
        "allowed": ["design", "develop", "construct", "formulate", "propose", "create"],
        "forbidden": ["explain", "describe", "identify", "list", "name", "define", "summarize"],
        "template": "[Design/Develop/Construct/Formulate] + [new artifact/system/solution]",
        "task": "Produce or design something new",
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
    
    # If the result is too long (explanation), try to extract the question part
    if len(cleaned) > 100:
        # Split by common sentence delimiters and take the first meaningful part
        parts = re.split(r'[.!?]', cleaned)
        if len(parts) > 1:
            # Take the first complete sentence that's not too short
            for part in parts:
                part = part.strip()
                if len(part.split()) >= 4 and len(part) <= 50:
                    cleaned = part
                    break
    
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


def _generate_rewrite(prompt: str) -> tuple[str, str, float]:
    try:
        from qwen_gguf_cli import QwenGgufCliGenerator

        gen = QwenGgufCliGenerator.for_task(
            "bloom_moderation",
            max_tokens=180,
            ctx_size=2048,
            threads=4,
        )
        out = gen.generate_prompt(prompt)
        return _clean_rewrite(out.answer), out.backend, float(out.elapsed_s)
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
        return _clean_rewrite(text), "llama-cpp-python", 0.0


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
            rewrite, backend, latency_s = _generate_rewrite(prompt)
        else:
            rewrite, backend, latency_s = rewrite_generator(prompt)
            rewrite = _clean_rewrite(rewrite)
        raw = rewrite
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
    """Build compact, structured prompt for deterministic cognitive transformation."""
    policy = TARGET_TRANSFORMATION_POLICY.get(target_level, TARGET_TRANSFORMATION_POLICY["Understand"])
    
    # Build compact prompt sections
    allowed_str = ", ".join(policy["allowed"])
    forbidden_str = ", ".join(policy["forbidden"])
    
    # Add special transformation guidance if applicable
    special_guidance = ""
    if previous_failure:
        special_guidance = f"\nPREVIOUS OUTPUT REJECTED.\nReason: {previous_failure}\nGenerate a new question.\n"
    
    # For Understand, add specific guidance about changing object of task
    object_guidance = ""
    example_guidance = ""
    if target_level == "Understand":
        object_guidance = "\nFocus on existing components, functions, or purpose.\nDo not ask how to create/design/develop.\n"
        example_guidance = "\nExample: Design X → Explain the components of X\n"
    elif target_level == "Remember":
        example_guidance = "\nExample: Design X → List the components of X\n"
    elif target_level == "Apply":
        example_guidance = "\nExample: Design X → Apply design principles to solve Y\n"
    elif target_level == "Analyze":
        example_guidance = "\nExample: Design X → Analyze the structure of X\n"
    elif target_level == "Evaluate":
        example_guidance = "\nExample: Design X → Evaluate X using criteria A, B, C\n"
    elif target_level == "Create":
        example_guidance = "\nExample: Explain X → Design a new X with feature Y\n"
    
    # Keep it compact but provide enough guidance for 1.5B model
    prompt = f"""{IM_START}system
You rewrite assessment questions to match specific cognitive levels.

TARGET LEVEL: {target_level.upper()}
COGNITIVE OPERATION: {policy["operation"].upper()}

TASK:
{policy["task"]}

ALLOW:
{allowed_str}

REMOVE:
{forbidden_str}

RULE:
Preserve the original topic.
Change the student's required cognitive action.
Do not preserve the original high-level task.
Output 10-30 words maximum.

{object_guidance}
{example_guidance}
{special_guidance}
OUTPUT ONLY ONE STUDENT-FACING QUESTION.
No explanation. No answer. No meta language.
{IM_END}
{IM_START}user
{question.strip()}
{IM_END}
{IM_START}assistant
""".strip()
    
    return prompt

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
    
    # Reject explanatory beginnings
    explanatory_beginnings = [
        "this",
        "the",
        "an",
        "a",  # when followed by explanation
        "to",
        "in order to",
    ]
    
    words = question_lower.split()
    if words and words[0] in explanatory_beginnings[:5]:
        return False, "Begins with explanatory meta-language"
    
    # Reject if it's too long (likely an explanation)
    if len(question.split()) > 40:
        return False, "Too long - likely an explanation not a question"
    
    # Reject if it contains multiple sentences (likely explanation)
    if question.count('.') > 1 or question.count('!') > 0:
        return False, "Multiple sentences - likely an explanation"
    
    # Check if it's actually a question (ends with ? or has question words)
    question_words = ["what", "how", "why", "which", "who", "when", "where", "explain", "describe", "analyze", "evaluate", "design", "create", "identify", "list", "compare"]
    has_question_structure = any(qw in question_lower for qw in question_words)
    
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
    
    # Check for forbidden operations
    for forbidden in policy["forbidden"]:
        # Check for "how to [forbidden]" patterns
        if f"how to {forbidden}" in question_lower:
            return False, f"Contains forbidden pattern: 'how to {forbidden}'"
        # Check for direct [forbidden] task
        forbidden_patterns = [
            f"{forbidden} a",
            f"{forbidden} the",
            f"{forbidden} an",
            f"to {forbidden}",
        ]
        for pattern in forbidden_patterns:
            if pattern in question_lower:
                return False, f"Contains forbidden operation: '{forbidden}'"
    
    # Check for required operations
    has_required = any(req in question_lower for req in policy["allowed"])
    if not has_required:
        return False, f"Lacks required cognitive operation for {target_level}"
    
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

def rewrite_to_target_level(question: str, target_level: str) -> tuple[str, bool, str]:
    """Rewrite question to target specific Bloom level with deterministic transformation pipeline.
    
    Pipeline:
    Generate → Output Format Check → Cognitive Task Check → Bloom Classifier → Confidence Check → PASS/REGENERATE
    
    Returns:
        tuple: (rewrite_text, validation_success, error_message)
        - rewrite_text: The generated rewrite or empty string if failed
        - validation_success: True if rewrite matched target level, False otherwise
        - error_message: Error details if validation failed completely
    """
    max_attempts = 3
    best_rewrite = ""
    best_confidence = 0.0
    previous_failure = ""
    
    for attempt in range(max_attempts):
        prompt = build_targeted_rewrite_prompt(question, target_level=target_level, previous_failure=previous_failure)
        try:
            from qwen_gguf_cli import QwenGgufCliGenerator

            gen = QwenGgufCliGenerator.for_task(
                "bloom_moderation",
                max_tokens=60,  # Increased for valid question generation
                ctx_size=2048,
                threads=4,
            )
            out = gen.generate_prompt(prompt)
            rewrite = _clean_rewrite(out.answer)
        except Exception:
            llm = _get_llm()
            output = llm(
                prompt,
                temperature=0.1,  # Lower temperature for more deterministic output
                top_p=0.8,
                top_k=30,
                repeat_penalty=1.1,
                max_tokens=60,  # Increased for valid question generation
                stop=[IM_END, IM_START],
            )
            text = output["choices"][0]["text"].strip()
            rewrite = _clean_rewrite(text)
        
        # Validate the rewrite using multi-layer approach
        if rewrite and len(rewrite.split()) >= 3:  # Minimum word count for a valid question
            # Layer 0: Output format validation - is this a student-facing question?
            format_valid, format_reason = _validate_output_format(rewrite)
            if not format_valid:
                previous_failure = format_reason
                continue  # Skip to next attempt
            
            # Layer 1: Semantic cognitive check (pre-filter using deterministic policy)
            semantic_valid, semantic_reason = _semantic_cognitive_check_deterministic(rewrite, target_level)
            if not semantic_valid:
                previous_failure = semantic_reason
                continue  # Skip to next attempt if semantic check fails
            
            # Layer 2: Classifier prediction with confidence threshold
            try:
                from predict_bloom import QwenBloomPredictor
                predictor = QwenBloomPredictor()
                validation = predictor.predict(rewrite)
                predicted_level = _canonical_bloom_label(validation["prediction"])
                confidence = validation.get("confidence", 0.0)
                
                # Handle NaN/None confidence
                if confidence is None or not isinstance(confidence, (int, float)):
                    confidence = 0.0
                
                # Layer 3: Cognitive task structure analysis
                task_valid, task_reason = _analyze_cognitive_task_structure(rewrite, target_level)
                
                # Only proceed if all checks pass
                if predicted_level == _canonical_bloom_label(target_level) and confidence >= 0.6 and task_valid:
                    return rewrite, True, ""
                else:
                    # Keep the best attempt if validation fails
                    if confidence > best_confidence:
                        best_rewrite = rewrite
                        best_confidence = confidence
                        # Generate failure reason for next attempt
                        if not task_valid:
                            previous_failure = task_reason
                        elif predicted_level != _canonical_bloom_label(target_level):
                            previous_failure = f"Classified as {predicted_level} instead of {target_level}"
                        else:
                            previous_failure = f"Low confidence ({confidence:.0%})"
            except Exception as e:
                # If validation fails, continue to next attempt
                previous_failure = f"Validation error: {str(e)}"
                continue
    
    # All attempts failed - return best attempt with error
    if best_rewrite:
        return best_rewrite, False, "Could not generate a validated rewrite matching the selected Bloom level."
    return "", False, "Failed to generate a valid rewrite after multiple attempts."
    
    # All attempts failed validation - do not return incorrect rewrite
    if best_rewrite:
        return "", False, f"Could not generate a validated rewrite matching {target_level}. The best attempt did not pass cognitive task structure validation."
    return "", False, f"Could not generate a valid rewrite for {target_level} after {max_attempts} attempts."

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
