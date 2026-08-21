"""
Quick test of single Understand generation.
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
    target_level = "Understand"
    
    print("Testing Create -> Understand transformation...")
    print(f"Original: {test_question}")
    print(f"Target: {target_level}")
    print()
    
    rewritten, validation_success, error_message = rewrite_to_target_level(test_question, target_level)
    
    print(f"Result: {rewritten}")
    print(f"Validation: {validation_success}")
    print(f"Error: {error_message}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
