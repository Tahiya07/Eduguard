"""
Test with a very simple prompt format to see if the model generates anything.
"""

import sys
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python not available")
    sys.exit(1)


def test_simple_format():
    """Test with simple prompt format."""
    print("Testing with simple prompt format...")
    
    model_path = Path("C:/Users/tahiy/PycharmProjects/Eduguard/models/qwen.gguf")
    
    # Very simple test prompt in Qwen format
    simple_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\nRewrite 'Define virtual memory' to use the word 'explain' instead.<|im_end|><|im_start|>assistant\n"
    
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
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=50,
        )
        
        text = output["choices"][0]["text"].strip()
        print(f"\nRaw output: {text}")
        print(f"Output length: {len(text)}")
        
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_format()
