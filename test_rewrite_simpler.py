"""
Test with the actual rewrite prompt but using the simple format that worked.
"""

import sys
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python not available")
    sys.exit(1)

from bloom_prompt import (
    _clean_rewrite,
    _validate_output_format,
    _semantic_cognitive_check_deterministic,
    _analyze_cognitive_task_structure,
)


def test_rewrite_simple_format():
    """Test rewrite with simple format."""
    print("Testing rewrite with simple prompt format...")
    
    original = "Define virtual memory."
    target_level = "Understand"
    
    print(f"Original: {original}")
    print(f"Target: {target_level}")
    
    model_path = Path("C:/Users/tahiy/PycharmProjects/Eduguard/models/qwen.gguf")
    
    # Build a simpler prompt using the same content
    simple_prompt = f"""<|im_start|>system
You rewrite assessment questions to match specific cognitive levels.

TARGET LEVEL: {target_level.upper()}
COGNITIVE OPERATION: COMPREHENSION

TASK:
Explain or describe existing concepts without creating/evaluating

ALLOW:
explain, describe, summarize, interpret, classify, discuss

REMOVE:
design, develop, construct, implement, create, build, solve, evaluate, justify, analyze

RULE:
Preserve the original topic.
Change the student's required cognitive action.
Do not preserve the original high-level task.
Output 10-30 words maximum.

Focus on existing components, functions, or purpose.
Do not ask how to create/design/develop.
Example: Design X -> Explain the components of X

OUTPUT ONLY ONE STUDENT-FACING QUESTION.
No explanation. No answer. No meta language.
<|im_end|>
<|im_start|>user
{original.strip()}
<|im_end|>
<|im_start|>assistant
"""
    
    print(f"Prompt length: {len(simple_prompt)}")
    print(f"Prompt preview (first 300 chars):\n{simple_prompt[:300]}...")
    
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        print(f"Model loaded successfully")
        
        output = llm(
            simple_prompt,
            temperature=0.1,
            top_p=0.8,
            top_k=30,
            repeat_penalty=1.1,
            max_tokens=96,
        )
        
        text = output["choices"][0]["text"].strip()
        print(f"\nRaw output: {text}")
        print(f"Output length: {len(text)}")
        
        cleaned = _clean_rewrite(text)
        print(f"Cleaned output: {cleaned}")
        
        if cleaned:
            format_valid, format_reason = _validate_output_format(cleaned)
            semantic_valid, semantic_reason = _semantic_cognitive_check_deterministic(cleaned, target_level)
            task_valid, task_reason = _analyze_cognitive_task_structure(cleaned, target_level)
            
            print(f"\nFormat Valid: {format_valid} - {format_reason}")
            print(f"Semantic Valid: {semantic_valid} - {semantic_reason}")
            print(f"Task Valid: {task_valid} - {task_reason}")
        else:
            print("Cleaned output is empty")
            
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_rewrite_simple_format()
