"""
Test single generation for the problematic case: Create -> Understand.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_single_generation():
    """Test single generation for Create -> Understand transformation."""
    try:
        from bloom_prompt import rewrite_to_target_level
    except Exception as e:
        print(f"[ERROR] Failed to import bloom_prompt: {e}")
        import traceback
        traceback.print_exc()
        return
    
    test_question = "Design a level zero restaurant management system."
    target_level = "Understand"
    
    print("=" * 80)
    print("SINGLE GENERATION TEST: Create -> Understand")
    print("=" * 80)
    print()
    print(f"Original Question: {test_question}")
    print(f"Target Level: {target_level}")
    print()
    print("Generating...")
    print()
    
    try:
        rewritten, validation_success, error_message = rewrite_to_target_level(test_question, target_level)
        
        print("-" * 80)
        print("RESULT:")
        print("-" * 80)
        print()
        
        if validation_success:
            print(f"[SUCCESS] Validation passed")
            print(f"Rewritten: {rewritten}")
            print()
            print("This is a valid Understand-level question.")
        else:
            print(f"[FAILED] Validation failed")
            print(f"Best attempt: {rewritten}")
            print(f"Error: {error_message}")
            print()
            print("The system could not generate a validated rewrite.")
        
        print()
        print("=" * 80)
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_generation()
