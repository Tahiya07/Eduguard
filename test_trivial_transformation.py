"""
Regression tests for trivial transformation detection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bloom_prompt import _is_trivial_transformation


def test_trivial_transformation_rejection():
    """Test that trivial transformations are correctly rejected."""
    print("=" * 70)
    print("TRIVIAL TRANSFORMATION REJECTION TESTS")
    print("=" * 70)
    print()
    
    # Test 1: Identity
    print("Test 1: Identity rejection")
    original = "Explain what virtual memory is."
    rewrite = "Explain what virtual memory is."
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Understand")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert is_trivial, "Should reject identical questions"
    print("[PASS]")
    print()
    
    # Test 2: Trivial "How would you explain" wrapping
    print("Test 2: Trivial 'How would you explain' wrapping")
    original = "Explain what virtual memory is."
    rewrite = "How would you explain what virtual memory is?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Understand")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    # The arrow character in the actual reason might cause encoding issues
    # Just check that it's trivial
    assert is_trivial, "Should reject trivial 'how would you explain' wrapping"
    print("[PASS]")
    print()
    
    # Test 3: "What is" to "How would you explain what is" wrapping
    print("Test 3: 'What is' to 'How would you explain what is' wrapping")
    original = "What is virtual memory?"
    rewrite = "How would you explain what virtual memory is?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Understand")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert is_trivial, "Should reject trivial 'what is' wrapping"
    print("[PASS]")
    print()
    
    # Test 4: Trivial "How would you apply" wrapping
    print("Test 4: Trivial 'How would you apply' wrapping")
    original = "Apply virtual memory concepts."
    rewrite = "How would you apply virtual memory concepts?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Apply")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert is_trivial, "Should reject trivial 'how would you apply' wrapping"
    print("[PASS]")
    print()
    
    # Test 5: Trivial "How would you analyze" wrapping
    print("Test 5: Trivial 'How would you analyze' wrapping")
    original = "Analyze the system architecture."
    rewrite = "How would you analyze the system architecture?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Analyze")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert is_trivial, "Should reject trivial 'how would you analyze' wrapping"
    print("[PASS]")
    print()
    
    # Test 6: Meaningful Understand transformation
    print("Test 6: Meaningful Understand transformation")
    original = "Explain what virtual memory is."
    rewrite = "How would you explain the purpose and function of virtual memory in managing system resources?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Understand")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert not is_trivial, "Should accept meaningful Understand transformation with elaboration"
    print("[PASS]")
    print()
    
    # Test 7: Apply with scenario (should pass)
    print("Test 7: Apply with scenario")
    original = "Explain what virtual memory is."
    rewrite = "How would you apply virtual memory in a specific scenario?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Apply")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert not is_trivial, "Should accept Apply with scenario"
    print("[PASS]")
    print()
    
    # Test 8: Analyze with components (should pass)
    print("Test 8: Analyze with components")
    original = "Explain what virtual memory is."
    rewrite = "How would you analyze the components of virtual memory?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Analyze")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert not is_trivial, "Should accept Analyze with components"
    print("[PASS]")
    print()
    
    # Test 9: Analyze without analytical keyword (should reject)
    print("Test 9: Analyze without analytical keyword")
    original = "Explain what virtual memory is."
    rewrite = "How would you describe virtual memory?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Analyze")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert is_trivial, "Should reject Analyze without analytical keyword"
    print("[PASS]")
    print()
    
    # Test 10: Evaluate without criteria (should reject)
    print("Test 10: Evaluate without criteria")
    original = "Describe the algorithm."
    rewrite = "How would you describe the algorithm?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Evaluate")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert is_trivial, "Should reject Evaluate without criteria"
    print("[PASS]")
    print()
    
    # Test 11: Evaluate with criteria (should pass)
    print("Test 11: Evaluate with criteria")
    original = "Describe the algorithm."
    rewrite = "How would you evaluate the algorithm using time complexity criteria?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Evaluate")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert not is_trivial, "Should accept Evaluate with criteria"
    print("[PASS]")
    print()
    
    # Test 12: Create without design keyword (should reject)
    print("Test 12: Create without design keyword")
    original = "Analyze the system."
    rewrite = "How would you improve the system?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Create")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert is_trivial, "Should reject Create without design keyword"
    print("[PASS]")
    print()
    
    # Test 13: Create with design (should pass)
    print("Test 13: Create with design")
    original = "Analyze the system."
    rewrite = "How would you design a new version of the system?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Create")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert not is_trivial, "Should accept Create with design"
    print("[PASS]")
    print()
    
    # Test 14: Apply without scenario (should reject)
    print("Test 14: Apply without scenario")
    original = "Explain sorting."
    rewrite = "How would you apply sorting?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Apply")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert is_trivial, "Should reject Apply without scenario"
    print("[PASS]")
    print()
    
    # Test 15: Remember without recall keyword (should reject)
    print("Test 15: Remember without recall keyword")
    original = "Describe the system."
    rewrite = "What is the system?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Remember")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert is_trivial, "Should reject Remember without recall keyword"
    print("[PASS]")
    print()
    
    # Test 16: Remember with recall keyword (should pass)
    print("Test 16: Remember with recall keyword")
    original = "Describe the system."
    rewrite = "What are the main components of the system?"
    is_trivial, reason = _is_trivial_transformation(original, rewrite, "Remember")
    print(f"Original: {original}")
    print(f"Rewrite: {rewrite}")
    print(f"Is Trivial: {is_trivial}")
    print(f"Reason: {reason}")
    assert not is_trivial, "Should accept Remember with recall keyword"
    print("[PASS]")
    print()
    
    print("=" * 70)
    print("ALL TRIVIAL TRANSFORMATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_trivial_transformation_rejection()
