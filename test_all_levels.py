"""
Test all six Bloom levels for the problematic test case.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_all_levels():
    """Test all six Bloom levels."""
    try:
        from bloom_prompt import rewrite_to_target_level
    except Exception as e:
        print(f"[ERROR] Failed to import bloom_prompt: {e}")
        import traceback
        traceback.print_exc()
        return
    
    test_question = "Design a level zero restaurant management system."
    target_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    
    print("=" * 80)
    print("ALL BLOOM LEVELS TEST")
    print("=" * 80)
    print()
    print(f"Original Question: {test_question}")
    print()
    
    results = []
    
    for target_level in target_levels:
        print(f"Target: {target_level}")
        print("-" * 80)
        
        try:
            rewritten, validation_success, error_message = rewrite_to_target_level(test_question, target_level)
            
            if validation_success:
                print(f"[SUCCESS] Validation passed")
                print(f"Rewritten: {rewritten}")
                results.append({
                    "target": target_level,
                    "rewritten": rewritten,
                    "validation": "PASS",
                    "error": None
                })
            else:
                print(f"[FAILED] Validation failed")
                print(f"Best attempt: {rewritten}")
                print(f"Error: {error_message}")
                results.append({
                    "target": target_level,
                    "rewritten": rewritten,
                    "validation": "FAIL",
                    "error": error_message
                })
            
            print()
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            print()
            results.append({
                "target": target_level,
                "rewritten": None,
                "validation": "ERROR",
                "error": str(e)
            })
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    passed = sum(1 for r in results if r["validation"] == "PASS")
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print()
    
    for result in results:
        status = "[PASS]" if result["validation"] == "PASS" else "[FAIL]"
        print(f"{status} {result['target']}: {result['rewritten'] or 'No output'}")
        if result["error"]:
            print(f"       Error: {result['error']}")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    test_all_levels()
