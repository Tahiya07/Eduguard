"""
Test using the actual qwen.gguf model directly.
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

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python not available")
    sys.exit(1)


def test_with_qwen_gguf():
    """Test using the qwen.gguf model directly."""
    print("Testing with qwen.gguf model...")
    
    original = "Define virtual memory."
    target_level = "Understand"
    
    print(f"Original: {original}")
    print(f"Target: {target_level}")
    
    # Model path
    model_path = Path("C:/Users/tahiy/PycharmProjects/Eduguard/models/qwen.gguf")
    print(f"Model path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    
    if not model_path.exists():
        print("Model file not found!")
        return
    
    # Build prompt
    prompt = build_targeted_rewrite_prompt(original, target_level=target_level, previous_failure="")
    print(f"\nPrompt preview (first 300 chars):\n{prompt[:300]}...")
    
    # Load model
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        print(f"Model loaded successfully")
        
        # Generate
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
    test_with_qwen_gguf()
