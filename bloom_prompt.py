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


def _clean_rewrite(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"(?im)^(bloom level|reason|higher[- ]level rewrite|rewrite)\s*:\s*", "", cleaned)
    cleaned = cleaned.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
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
) -> str:
    """Prompt for rewriting to a specific target Bloom level with cognitive transformation."""
    guide = _LEVEL_GUIDANCE.get(target_level, _LEVEL_GUIDANCE["Understand"])
    
    # Cognitive transformation guidance for each level
    cognitive_transforms = {
        "Remember": "CHANGE REQUIREMENT: The student must only recall, identify, list, name, define, or recognize facts. Remove any requirement to explain, apply, analyze, evaluate, or create. Focus on pure recall and recognition.",
        "Understand": "CHANGE REQUIREMENT: The student must explain, describe, summarize, interpret, classify, or discuss concepts. Remove any requirement to create, design, or build something. Focus on comprehension and explanation.",
        "Apply": "CHANGE REQUIREMENT: The student must use knowledge in a new situation, apply concepts, demonstrate, implement, solve, or execute. Remove any requirement to merely recall or explain. Focus on practical application.",
        "Analyze": "CHANGE REQUIREMENT: The student must analyze, compare, differentiate, examine, investigate, or categorize components and relationships. Remove any requirement to create something new. Focus on breaking down and examining structure.",
        "Evaluate": "CHANGE REQUIREMENT: The student must evaluate, assess, justify, critique, defend, or judge using criteria. Remove any requirement to merely describe or apply. Focus on making judgments with evidence.",
        "Create": "CHANGE REQUIREMENT: The student must design, develop, construct, formulate, propose, or create something new. Remove any requirement to merely recall, explain, or analyze. Focus on synthesis and production.",
    }
    
    # Clear task structure guidance to avoid ambiguous multi-verb questions
    task_structure_guidance = {
        "Remember": "Use simple, direct verbs: identify, list, name, define, recognize, recall. Avoid complex multi-verb constructions.",
        "Understand": "Use single cognitive verbs: explain, describe, summarize, interpret, classify, discuss. Avoid mixing with create/evaluate verbs.",
        "Apply": "Use action verbs: apply, use, demonstrate, implement, solve, execute. Focus on using knowledge in a specific situation.",
        "Analyze": "Use analytical verbs: analyze, compare, differentiate, examine, investigate, categorize. Focus on examining relationships and structure.",
        "Evaluate": "Use judgment verbs: evaluate, assess, justify, critique, defend, judge. Always include criteria or evidence requirements.",
        "Create": "Use creative verbs: design, develop, construct, formulate, propose, create. Focus on producing something new.",
    }
    
    # Validation criteria for each level
    validation_criteria = {
        "Remember": "VALIDATION CHECK: Does this require ONLY recall/recognition/identification/listing/naming/definition? Does it avoid explanation/application/analysis/evaluation/creation?",
        "Understand": "VALIDATION CHECK: Does this require explaining/describing/summarizing/interpreting/classifying/discussing? Does it avoid designing/constructing/evaluating/solving novel problems? Is the dominant cognitive operation comprehension?",
        "Apply": "VALIDATION CHECK: Does this require using knowledge/procedures in a specific situation or applying a known method? Does it go beyond merely explaining the concept? Is the dominant task application?",
        "Analyze": "VALIDATION CHECK: Does this require breaking information into parts and examining relationships/differences/causes/patterns/evidence? Does it involve analytical comparison/examination rather than simple description? Is the dominant task analysis?",
        "Evaluate": "VALIDATION CHECK: Does this require making a judgment using criteria/evidence/standards/justification? Does it avoid merely asking for opinion without criteria? Is the dominant task evaluation?",
        "Create": "VALIDATION CHECK: Does this require producing/designing/constructing/formulating/developing/proposing something new? Does it avoid merely explaining an existing design? Is the dominant task creation?",
    }
    
    cognitive_guidance = cognitive_transforms.get(target_level, "")
    structure_guidance = task_structure_guidance.get(target_level, "")
    validation_guidance = validation_criteria.get(target_level, "")
    
    return f"""{IM_START}system
You are an expert exam-question editor using Bloom's Taxonomy.

CRITICAL TASK: Transform the cognitive demand of this question to match **{target_level}**-level thinking.

Target cognitive requirement: {guide['depth']}

{cognitive_guidance}

{structure_guidance}

{validation_guidance}

COMMON MISTAKES TO AVOID:
- Do NOT simply change vocabulary (e.g., "level one" to "elementary") while keeping the same cognitive task
- Do NOT only add a cognitive verb if the task structure remains unchanged
- Avoid ambiguous multi-verb questions (e.g., "Explain and design" - choose one clear cognitive operation)
- "Design" as the student's task means Create-level, but "explain the design" can be Understand-level
- Focus on what the STUDENT must DO, not on words that appear in the subject matter
- Your rewrite must fundamentally change the cognitive operation the student performs

PREFERRED QUESTION STRUCTURES:
- Remember: "Identify the components of..." "List the steps for..." "Define the term..."
- Understand: "Explain the principles of..." "Describe how..." "Summarize the process of..."
- Apply: "Apply [concept] to solve..." "Use [method] to address..." "Demonstrate how..."
- Analyze: "Analyze the relationship between..." "Compare the differences in..." "Examine the causes of..."
- Evaluate: "Evaluate the effectiveness of... using [criteria]" "Assess whether... based on..."
- Create: "Design a [system] that includes..." "Develop a [solution] for..." "Construct a [model] of..."

Preserve the original topic and educational intent wherever possible, but the cognitive operation MUST change.

Output ONLY the rewritten question as a single exam-style sentence or short paragraph.
No labels, no preamble, no bullet list, no explanation.
{IM_END}
{IM_START}user
Rewrite this question for {target_level}-level cognition (fundamentally change the cognitive task):
{question.strip()}
{IM_END}
{IM_START}assistant
""".strip()

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
    """Rewrite question to target specific Bloom level with cognitive task structure validation.
    
    Returns:
        tuple: (rewrite_text, validation_success, error_message)
        - rewrite_text: The generated rewrite or empty string if failed
        - validation_success: True if rewrite matched target level, False otherwise
        - error_message: Error details if validation failed completely
    """
    max_attempts = 3
    best_rewrite = ""
    best_confidence = 0.0
    
    for attempt in range(max_attempts):
        prompt = build_targeted_rewrite_prompt(question, target_level=target_level)
        try:
            from qwen_gguf_cli import QwenGgufCliGenerator

            gen = QwenGgufCliGenerator.for_task(
                "bloom_moderation",
                max_tokens=180,
                ctx_size=2048,
                threads=4,
            )
            out = gen.generate_prompt(prompt)
            rewrite = _clean_rewrite(out.answer)
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
            rewrite = _clean_rewrite(text)
        
        # Validate the rewrite using multi-layer approach
        if rewrite and len(rewrite.split()) >= 6:
            try:
                from predict_bloom import QwenBloomPredictor
                predictor = QwenBloomPredictor()
                validation = predictor.predict(rewrite)
                predicted_level = _canonical_bloom_label(validation["prediction"])
                confidence = validation.get("confidence", 0.0)
                
                # Layer 1: Classifier prediction with confidence threshold
                if predicted_level == _canonical_bloom_label(target_level) and confidence >= 0.6:
                    # Layer 2: Cognitive task structure analysis
                    task_valid, task_reason = _analyze_cognitive_task_structure(rewrite, target_level)
                    if task_valid:
                        return rewrite, True, ""
                    else:
                        # Task structure failed, continue trying
                        if confidence > best_confidence:
                            best_rewrite = rewrite
                            best_confidence = confidence
                else:
                    # Classifier prediction failed
                    if confidence > best_confidence:
                        best_rewrite = rewrite
                        best_confidence = confidence
            except Exception as e:
                # If validation fails, continue to next attempt
                continue
    
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
