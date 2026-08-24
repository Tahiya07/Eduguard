"""
Test that existing Apply and Analyze examples still pass with trivial transformation check.
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
    _is_trivial_transformation,
)

test_cases = [
    ("Explain what virtual memory is.", "Apply"),
    ("Explain what virtual memory is.", "Analyze"),
]

model_path = Path("C:/Users/tahiy/PycharmProjects/Eduguard/models/qwen.gguf")

llm = Llama(
    model_path=str(model_path),
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    use_mmap=True,
    use_mlock=False,
    verbose=False,
)

for original, target in test_cases:
    print(f"\n{'=' * 70}")
    print(f"Original: {original}")
    print(f"Target: {target}")
    print(f"{'=' * 70}")
    
    prompt = build_targeted_rewrite_prompt(original, target_level=target)
    
    output = llm(
        prompt,
        temperature=0.1,
        top_p=0.8,
        top_k=30,
        repeat_penalty=1.1,
        max_tokens=96,
    )
    
    text = output["choices"][0]["text"].strip()
    cleaned = _clean_rewrite(text)
    
    print(f"Generated: {cleaned}")
    
    # Check trivial transformation
    is_trivial, trivial_reason = _is_trivial_transformation(original, cleaned, target)
    print(f"Trivial Transformation: {is_trivial}")
    if trivial_reason:
        print(f"Reason: {trivial_reason}")
    
    if cleaned and not is_trivial:
        format_valid, format_reason = _validate_output_format(cleaned)
        semantic_valid, semantic_reason = _semantic_cognitive_check_deterministic(cleaned, target)
        task_valid, task_reason = _analyze_cognitive_task_structure(cleaned, target)
        
        print(f"Format Valid: {format_valid} - {format_reason}")
        print(f"Semantic Valid: {semantic_valid} - {semantic_reason}")
        print(f"Task Valid: {task_valid} - {task_reason}")
        
        if format_valid and semantic_valid and task_valid:
            print("[PASS] - All validations passed")
        else:
            print("[FAIL] - Validation failed")
    elif is_trivial:
        print("[FAIL] - Trivial transformation rejected")
