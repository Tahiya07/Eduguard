"""
Test script to verify Bloom validation logic without requiring full model loading.
This tests the cognitive task structure analysis function independently.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_cognitive_task_structure():
    """Test the cognitive task structure analysis function."""
    print("Testing cognitive task structure analysis...")
    
    # Import the function
    try:
        from bloom_prompt import _analyze_cognitive_task_structure
        print("[OK] Successfully imported _analyze_cognitive_task_structure")
    except ImportError as e:
        print(f"[FAIL] Failed to import: {e}")
        return False
    
    # Test cases from requirements
    test_cases = [
        ("Explain the design principles of a banking system.", "Understand", True, "VALID Understand - design in subject matter"),
        ("Design a basic banking system.", "Understand", False, "INVALID Understand - design as student task"),
        ("Design and explain a banking system.", "Understand", False, "INVALID Understand - multi-verb with design"),
        ("Analyze the design of a banking system and identify its strengths and weaknesses.", "Analyze", True, "VALID Analyze - design in subject matter"),
        ("Evaluate the effectiveness of the banking system using criteria such as security, scalability, and cost.", "Evaluate", True, "VALID Evaluate - with criteria"),
        ("Design a banking system that incorporates secure authentication and transaction processing.", "Create", True, "VALID Create - design as student task"),
        ("Explain and justify whether the banking system is effective using explicit criteria.", "Evaluate", True, "VALID Evaluate - explain and justify with criteria"),
    ]
    
    print("\nRunning specific validation tests:\n")
    
    passed = 0
    failed = 0
    
    for question, target_level, expected_valid, description in test_cases:
        try:
            is_valid, reason = _analyze_cognitive_task_structure(question, target_level)
            
            if is_valid == expected_valid:
                print(f"[PASS] {description}")
                print(f"  Result: {reason}")
                passed += 1
            else:
                print(f"[FAIL] {description}")
                print(f"  Expected: {expected_valid}, Got: {is_valid}")
                print(f"  Reason: {reason}")
                failed += 1
        except Exception as e:
            print(f"[ERROR] {description}")
            print(f"  Exception: {str(e)}")
            failed += 1
        
        print()
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0

def test_import_chain():
    """Test that the import chain works without model loading."""
    print("Testing import chain...")
    
    try:
        # Test basic imports
        import re
        print("[OK] Python stdlib imports work")
        
        # Test project imports
        from bloom_prompt import _canonical_bloom_label, BLOOM_ORDER
        print("[OK] bloom_prompt basic imports work")
        
        # Test that constants are defined
        assert len(BLOOM_ORDER) == 6, "BLOOM_ORDER should have 6 levels"
        print("[OK] Bloom levels are correctly defined")
        
        return True
    except Exception as e:
        print(f"[FAIL] Import chain failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("BLOOM VALIDATION LOGIC TEST")
    print("=" * 60)
    print()
    
    # Test import chain first
    import_ok = test_import_chain()
    print()
    
    # Test cognitive task structure analysis
    if import_ok:
        analysis_ok = test_cognitive_task_structure()
    else:
        analysis_ok = False
    
    print()
    print("=" * 60)
    if import_ok and analysis_ok:
        print("ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)