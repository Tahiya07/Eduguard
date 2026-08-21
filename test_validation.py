"""
Test validation functions without running the model.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_output_format_validation():
    """Test output format validation."""
    try:
        from bloom_prompt import _validate_output_format
    except Exception as e:
        print(f"[ERROR] Failed to import: {e}")
        return
    
    print("=" * 80)
    print("OUTPUT FORMAT VALIDATION TEST")
    print("=" * 80)
    print()
    
    test_cases = [
        ("Explain the main components of a restaurant management system.", True, "Valid question"),
        ("This question asks the student to explain components.", False, "Meta-language"),
        ("Describe how to design a restaurant management system.", True, "Valid question structure"),
        ("To understand the system, one must analyze its parts.", False, "Explanatory beginning"),
        ("Explain the components. Then analyze the relationships.", False, "Multiple sentences"),
        ("What are the main components of the system?", True, "Valid question"),
    ]
    
    for question, expected_valid, description in test_cases:
        is_valid, reason = _validate_output_format(question)
        status = "[PASS]" if is_valid == expected_valid else "[FAIL]"
        print(f"{status} {description}")
        print(f"     Question: {question}")
        print(f"     Valid: {is_valid}, Reason: {reason}")
        print()

def test_semantic_validation():
    """Test semantic cognitive validation."""
    try:
        from bloom_prompt import _semantic_cognitive_check_deterministic
    except Exception as e:
        print(f"[ERROR] Failed to import: {e}")
        return
    
    print("=" * 80)
    print("SEMANTIC COGNITIVE VALIDATION TEST")
    print("=" * 80)
    print()
    
    test_cases = [
        ("Explain the main components of a restaurant management system.", "Understand", True, "Valid Understand"),
        ("Describe how to design a restaurant management system.", "Understand", False, "Contains 'how to design'"),
        ("Explain how to implement a restaurant management system.", "Understand", False, "Contains 'how to implement'"),
        ("Design a restaurant management system.", "Understand", False, "Contains forbidden 'design'"),
        ("Identify the components of a restaurant management system.", "Remember", True, "Valid Remember"),
        ("Analyze the components of a restaurant management system.", "Analyze", True, "Valid Analyze"),
        ("Design a restaurant management system.", "Create", True, "Valid Create"),
    ]
    
    for question, target_level, expected_valid, description in test_cases:
        is_valid, reason = _semantic_cognitive_check_deterministic(question, target_level)
        status = "[PASS]" if is_valid == expected_valid else "[FAIL]"
        print(f"{status} {description}")
        print(f"     Question: {question}")
        print(f"     Target: {target_level}")
        print(f"     Valid: {is_valid}, Reason: {reason}")
        print()

if __name__ == "__main__":
    test_output_format_validation()
    test_semantic_validation()
