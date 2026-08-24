"""
Debug what the model generates for Apply and Analyze with the new compact prompt.
"""

import sys
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python not available")
    sys.exit(1)

from bloom_prompt import (
    build_targeted_rewrite_prompt,
    _clean_rewrite,
    _validate_output_format,
    _semantic_cognitive_check_deterministic,
    _analyze_cognitive_task_structure,
)


def test_apply_analyze_direct():
    """Test Apply and Analyze directly."""
    print("Testing Apply and Analyze with compact prompt...")
    
    original = "Explain what virtual memory is."
    test_levels = ["Apply", "Analyze"]
    
    model_path = Path("C:/Users/tahiy/PycharmProjects/Eduguard/models/qwen.gguf")
    
    for target_level in test_levels:
        print(f"\n{'=' * 70}")
        print(f"Original: {original}")
        print(f"Target: {target_level}")
        print(f"{'=' * 70}")
        
        prompt = build_targeted_rewrite_prompt(original, target_level=target_level)
        print(f"Prompt:\n{prompt}")
        
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
            
            output = llm(
                prompt,
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
    test_apply_analyze_direct()
