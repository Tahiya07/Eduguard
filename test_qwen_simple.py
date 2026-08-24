"""
Test the qwen.gguf model with a simple prompt to verify it works.
"""

import sys
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python not available")
    sys.exit(1)


def test_qwen_simple():
    """Test qwen.gguf with a simple prompt."""
    print("Testing qwen.gguf with simple prompt...")
    
    model_path = Path("C:/Users/tahiy/PycharmProjects/Eduguard/models/qwen.gguf")
    print(f"Model path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    
    if not model_path.exists():
        print("Model file not found!")
        return
    
    # Simple test prompt
    simple_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\nWhat is 2+2?<|im_end|><|im_start|>assistant\n"
    
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=True,  # Enable verbose to see what's happening
        )
        print(f"Model loaded successfully")
        
        # Generate
        output = llm(
            simple_prompt,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=50,
            stop=["<|im_end|>", "<|im_start|>"],
        )
        
        text = output["choices"][0]["text"].strip()
        print(f"\nRaw output: {text}")
        print(f"Output length: {len(text)}")
        
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_qwen_simple()
