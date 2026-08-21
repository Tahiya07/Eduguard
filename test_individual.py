"""
Test each Bloom level individually.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from bloom_prompt import rewrite_to_target_level
    
    test_question = "Design a level zero restaurant management system."
    target_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    
    print("=" * 80)
    print("INDIVIDUAL BLOOM LEVEL TESTS")
    print("=" * 80)
    print()
    print(f"Original: {test_question}")
    print()
    
    for target_level in target_levels:
        print(f"Target: {target_level}")
        print("-" * 40)
        
        try:
            rewritten, validation_success, error_message = rewrite_to_target_level(test_question, target_level)
            
            if validation_success:
                print(f"[PASS] {rewritten}")
            else:
                print(f"[FAIL] {rewritten}")
                print(f"Error: {error_message}")
            
        except Exception as e:
            print(f"[ERROR] {e}")
        
        print()
    
    print("=" * 80)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
