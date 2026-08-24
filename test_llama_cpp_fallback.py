"""
Test the llama-cpp-python fallback generation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bloom_prompt import (
    build_targeted_rewrite_prompt,
    _clean_rewrite,
    _validate_output_format,
    _semantic_cognitive_check_deterministic,
    _analyze_cognitive_task_structure,
    _get_llm,
)


def test_llama_cpp_fallback():
    """Test the llama-cpp-python fallback generation."""
    print("Testing llama-cpp-python fallback...")
    
    original = "Define virtual memory."
    target_level = "Understand"
    
    print(f"Original: {original}")
    print(f"Target: {target_level}")
    
    # Build prompt
    prompt = build_targeted_rewrite_prompt(original, target_level=target_level, previous_failure="")
    print(f"\nPrompt preview (first 300 chars):\n{prompt[:300]}...")
    
    # Try llama-cpp-python fallback
    try:
        llm = _get_llm()
        print(f"LLM loaded: {llm}")
        
        output = llm(
            prompt,
            temperature=0.1,
            top_p=0.8,
            top_k=30,
            repeat_penalty=1.1,
            max_tokens=96,
            stop=["<|im_end|>", "<|im_start|>"],
        )
        
        text = output["choices"][0]["text"].strip()
        cleaned = _clean_rewrite(text)
        
        print(f"\nRaw output: {text}")
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
    test_llama_cpp_fallback()
