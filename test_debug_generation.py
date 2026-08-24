"""
Debug generation to see what the model actually produces.
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
)


def test_generation_debug():
    """Debug what the model actually generates."""
    print("Debugging generation...")
    
    original = "Define virtual memory."
    target_level = "Understand"
    
    print(f"Original: {original}")
    print(f"Target: {target_level}")
    
    # Build prompt
    prompt = build_targeted_rewrite_prompt(original, target_level=target_level, previous_failure="")
    print(f"\nPrompt preview (first 500 chars):\n{prompt[:500]}...")
    
    # Try to generate
    try:
        from qwen_gguf_cli import QwenGgufCliGenerator
        
        gen = QwenGgufCliGenerator.for_task(
            "bloom_moderation",
            max_tokens=96,
            ctx_size=2048,
            threads=4,
        )
        out = gen.generate_prompt(prompt)
        raw_answer = out.answer
        cleaned = _clean_rewrite(raw_answer)
        
        print(f"\nRaw answer: {raw_answer}")
        print(f"Cleaned answer: {cleaned}")
        
        if cleaned:
            format_valid, format_reason = _validate_output_format(cleaned)
            semantic_valid, semantic_reason = _semantic_cognitive_check_deterministic(cleaned, target_level)
            task_valid, task_reason = _analyze_cognitive_task_structure(cleaned, target_level)
            
            print(f"\nFormat Valid: {format_valid} - {format_reason}")
            print(f"Semantic Valid: {semantic_valid} - {semantic_reason}")
            print(f"Task Valid: {task_valid} - {task_reason}")
        else:
            print("Cleaned answer is empty")
            
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_generation_debug()
