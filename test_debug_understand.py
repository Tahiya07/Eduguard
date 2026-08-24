"""
Debug why Understand is failing.
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

original = "Explain what virtual memory is."
target_level = "Understand"

model_path = Path("C:/Users/tahiy/PycharmProjects/Eduguard/models/qwen.gguf")

prompt = build_targeted_rewrite_prompt(original, target_level=target_level)
print(f"Prompt:\n{prompt}")

llm = Llama(
    model_path=str(model_path),
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    use_mmap=True,
    use_mlock=False,
    verbose=False,
)

for i in range(4):
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
    
    print(f"\nAttempt {i+1}:")
    print(f"Raw: {text}")
    print(f"Cleaned: {cleaned}")
    
    if cleaned:
        format_valid, format_reason = _validate_output_format(cleaned)
        semantic_valid, semantic_reason = _semantic_cognitive_check_deterministic(cleaned, target_level)
        task_valid, task_reason = _analyze_cognitive_task_structure(cleaned, target_level)
        
        print(f"Format: {format_valid} - {format_reason}")
        print(f"Semantic: {semantic_valid} - {semantic_reason}")
        print(f"Task: {task_valid} - {task_reason}")
