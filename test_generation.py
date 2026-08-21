"""
Test the target level rewrite generation with the specific test case.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_understand_generation():
    """Test generation of Understand-level question from Create-level input."""
    try:
        from bloom_prompt import rewrite_to_target_level
    except Exception as e:
        print(f"Failed to import bloom_prompt: {e}")
        import traceback
        traceback.print_exc()
        return
    
    test_cases = [
        ("Design a level zero restaurant management system.", "Understand"),
        ("Design a banking system.", "Understand"),
        ("Develop a student management application.", "Understand"),
        ("Construct a database system.", "Understand"),
        ("Create a restaurant management system.", "Understand"),
        ("Design a network architecture.", "Understand"),
    ]
    
    print("=" * 80)
    print("TARGET LEVEL REWRITE GENERATION TEST")
    print("=" * 80)
    print()
    
    for question, target_level in test_cases:
        print(f"Input: {question}")
        print(f"Target: {target_level}")
        print("-" * 80)
        
        try:
            rewritten, validation_success, error_message = rewrite_to_target_level(question, target_level)
            
            if validation_success:
                print(f"✅ SUCCESS")
                print(f"Rewritten: {rewritten}")
            else:
                print(f"❌ VALIDATION FAILED")
                print(f"Best attempt: {rewritten}")
                print(f"Error: {error_message}")
            
            print()
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 80)

if __name__ == "__main__":
    test_understand_generation()
