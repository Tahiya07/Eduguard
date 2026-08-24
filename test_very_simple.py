"""
Test with a much simpler, more direct prompt.
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


def test_very_simple_prompt():
    """Test with very simple prompt."""
    print("Testing with very simple prompt...")
    
    original = "Define virtual memory."
    target_level = "Understand"
    
    print(f"Original: {original}")
    print(f"Target: {target_level}")
    
    model_path = Path("C:/Users/tahiy/PycharmProjects/Eduguard/models/qwen.gguf")
    
    # Much simpler prompt
    simple_prompt = f"""<|im_start|>system
Rewrite the question to use the word "explain" instead of "define". Keep the same topic. Output only the question.<|im_end|>
<|im_start|>user
{original}
<|im_end|>
<|im_start|>assistant
"""
    
    print(f"Prompt: {simple_prompt}")
    
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
            max_tokens=50,
        )
        
        text = output["choices"][0]["text"].strip()
        print(f"\nRaw output: {text}")
        
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
    test_very_simple_prompt()
