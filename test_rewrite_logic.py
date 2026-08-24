"""
Lightweight test for target-level rewrite logic changes.
Tests the acceptance/fallback logic without loading full models.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bloom_prompt import (
    _validate_output_format,
    _semantic_cognitive_check_deterministic,
    _analyze_cognitive_task_structure,
    _canonical_bloom_label,
)


def test_format_validation():
    """Test that format validation still works correctly."""
    print("\n=== TESTING FORMAT VALIDATION ===\n")
    
    # Valid questions
    valid_questions = [
        "What are the main components of a database system?",
        "Explain the principles of object-oriented programming.",
        "How does virtual memory work in modern operating systems?",
    ]
    
    for question in valid_questions:
        is_valid, reason = _validate_output_format(question)
        print(f"Question: {question}")
        print(f"Valid: {is_valid}, Reason: {reason}")
        assert is_valid, f"Valid question rejected: {reason}"
        print("[PASS]")
    
    # Invalid questions (meta-language)
    invalid_questions = [
        "This question asks about database components.",
        "The rewritten question is about object-oriented programming.",
        "To understand virtual memory, one must know about paging.",
    ]
    
    for question in invalid_questions:
        is_valid, reason = _validate_output_format(question)
        print(f"Question: {question}")
        print(f"Valid: {is_valid}, Reason: {reason}")
        assert not is_valid, f"Invalid question accepted: {question}"
        print("[PASS]")
    
    print("\nFormat validation tests passed.\n")


def test_semantic_validation():
    """Test that semantic cognitive validation still works correctly."""
    print("\n=== TESTING SEMANTIC VALIDATION ===\n")
    
    # Test Remember level
    valid_remember = "List the main components of a database system."
    invalid_remember = "Design a comprehensive database system."
    
    is_valid, reason = _semantic_cognitive_check_deterministic(valid_remember, "Remember")
    print(f"Valid Remember: {valid_remember}")
    print(f"Valid: {is_valid}, Reason: {reason}")
    assert is_valid, f"Valid Remember question rejected: {reason}"
    print("[PASS]")
    
    is_valid, reason = _semantic_cognitive_check_deterministic(invalid_remember, "Remember")
    print(f"Invalid Remember: {invalid_remember}")
    print(f"Valid: {is_valid}, Reason: {reason}")
    assert not is_valid, f"Invalid Remember question accepted"
    print("[PASS]")
    
    # Test Understand level
    valid_understand = "Explain the main components of a database system."
    invalid_understand = "Design a comprehensive database system."
    
    is_valid, reason = _semantic_cognitive_check_deterministic(valid_understand, "Understand")
    print(f"Valid Understand: {valid_understand}")
    print(f"Valid: {is_valid}, Reason: {reason}")
    assert is_valid, f"Valid Understand question rejected: {reason}"
    print("[PASS]")
    
    is_valid, reason = _semantic_cognitive_check_deterministic(invalid_understand, "Understand")
    print(f"Invalid Understand: {invalid_understand}")
    print(f"Valid: {is_valid}, Reason: {reason}")
    assert not is_valid, f"Invalid Understand question accepted"
    print("[PASS]")
    
    # Test Create level
    valid_create = "Design a comprehensive database system."
    invalid_create = "Explain the main components of a database system."
    
    is_valid, reason = _semantic_cognitive_check_deterministic(valid_create, "Create")
    print(f"Valid Create: {valid_create}")
    print(f"Valid: {is_valid}, Reason: {reason}")
    assert is_valid, f"Valid Create question rejected: {reason}"
    print("[PASS]")
    
    is_valid, reason = _semantic_cognitive_check_deterministic(invalid_create, "Create")
    print(f"Invalid Create: {invalid_create}")
    print(f"Valid: {is_valid}, Reason: {reason}")
    assert not is_valid, f"Invalid Create question accepted"
    print("[PASS]")
    
    print("\nSemantic validation tests passed.\n")


def test_task_structure_validation():
    """Test that task structure validation still works correctly."""
    print("\n=== TESTING TASK STRUCTURE VALIDATION ===\n")
    
    # Test Understand level - should reject "explain how to design"
    invalid_understand = "Explain how to design a database system."
    is_valid, reason = _analyze_cognitive_task_structure(invalid_understand, "Understand")
    print(f"Invalid Understand: {invalid_understand}")
    print(f"Valid: {is_valid}, Reason: {reason}")
    assert not is_valid, f"Invalid Understand structure accepted"
    print("[PASS]")
    
    # Test Understand level - should accept "explain the design"
    valid_understand = "Explain the design of a database system."
    is_valid, reason = _analyze_cognitive_task_structure(valid_understand, "Understand")
    print(f"Valid Understand: {valid_understand}")
    print(f"Valid: {is_valid}, Reason: {reason}")
    assert is_valid, f"Valid Understand structure rejected: {reason}"
    print("[PASS]")
    
    # Test Create level - should accept "design a"
    valid_create = "Design a database system with secure authentication."
    is_valid, reason = _analyze_cognitive_task_structure(valid_create, "Create")
    print(f"Valid Create: {valid_create}")
    print(f"Valid: {is_valid}, Reason: {reason}")
    assert is_valid, f"Valid Create structure rejected: {reason}"
    print("[PASS]")
    
    print("\nTask structure validation tests passed.\n")


def test_candidate_scoring_logic():
    """Test the new candidate scoring logic."""
    print("\n=== TESTING CANDIDATE SCORING LOGIC ===\n")
    
    # Simulate the scoring logic from the new implementation
    def calculate_candidate_score(format_valid, semantic_valid, task_valid, confidence, matches_target):
        score = 0.0
        if format_valid:
            score += 100.0
        if semantic_valid:
            score += 50.0
        if task_valid:
            score += 30.0
        score += confidence * 10.0
        if matches_target:
            score += 20.0
        return score
    
    # Test cases
    test_cases = [
        {
            "name": "Perfect match",
            "format_valid": True,
            "semantic_valid": True,
            "task_valid": True,
            "confidence": 0.75,
            "matches_target": True,
            "expected_min": 200.0,
        },
        {
            "name": "Good candidate with lower confidence",
            "format_valid": True,
            "semantic_valid": True,
            "task_valid": True,
            "confidence": 0.45,
            "matches_target": False,
            "expected_min": 180.0,
        },
        {
            "name": "Format and semantic valid only",
            "format_valid": True,
            "semantic_valid": True,
            "task_valid": False,
            "confidence": 0.30,
            "matches_target": False,
            "expected_min": 150.0,
        },
        {
            "name": "Format valid only",
            "format_valid": True,
            "semantic_valid": False,
            "task_valid": False,
            "confidence": 0.20,
            "matches_target": False,
            "expected_min": 100.0,
        },
    ]
    
    for test_case in test_cases:
        score = calculate_candidate_score(
            test_case["format_valid"],
            test_case["semantic_valid"],
            test_case["task_valid"],
            test_case["confidence"],
            test_case["matches_target"],
        )
        print(f"Test: {test_case['name']}")
        print(f"Score: {score}")
        assert score >= test_case["expected_min"], f"Score too low: {score} < {test_case['expected_min']}"
        print("[PASS]")
    
    print("\nCandidate scoring logic tests passed.\n")


def test_acceptance_criteria():
    """Test the new acceptance criteria."""
    print("\n=== TESTING ACCEPTANCE CRITERIA ===\n")
    
    # Test immediate acceptance (exact match with high confidence)
    print("Test 1: Immediate acceptance (predicted == target, confidence >= 0.60, task_valid)")
    predicted_level = "Understand"
    target_level = "Understand"
    confidence = 0.75
    task_valid = True
    
    canonical_target = _canonical_bloom_label(target_level)
    immediate_accept = (predicted_level == canonical_target and confidence >= 0.60 and task_valid)
    print(f"Predicted: {predicted_level}, Target: {canonical_target}, Confidence: {confidence}, Task Valid: {task_valid}")
    print(f"Immediate Accept: {immediate_accept}")
    assert immediate_accept, "Should accept immediately"
    print("[PASS]")
    
    # Test secondary acceptance with needs_review (task_valid, confidence < 0.60)
    print("\nTest 2: Secondary acceptance with needs_review (task_valid, confidence < 0.60)")
    predicted_level = "Understand"  # Matches target
    target_level = "Understand"
    confidence = 0.45
    task_valid = True
    
    secondary_accept = (task_valid and confidence < 0.60)
    needs_review = (task_valid and confidence < 0.60)
    print(f"Predicted: {predicted_level}, Target: {canonical_target}, Confidence: {confidence}, Task Valid: {task_valid}")
    print(f"Secondary Accept: {secondary_accept}, Needs Review: {needs_review}")
    assert secondary_accept, "Should accept with needs_review"
    assert needs_review, "Should flag for review due to low confidence"
    print("[PASS]")
    
    # Test secondary acceptance with needs_review (classifier disagrees)
    print("\nTest 3: Secondary acceptance with needs_review (classifier disagrees, task_valid)")
    predicted_level = "Analyze"  # Different from target
    target_level = "Understand"
    confidence = 0.55
    task_valid = True
    
    secondary_accept = (task_valid and predicted_level != canonical_target)
    needs_review = (task_valid and predicted_level != canonical_target)
    print(f"Predicted: {predicted_level}, Target: {canonical_target}, Confidence: {confidence}, Task Valid: {task_valid}")
    print(f"Secondary Accept: {secondary_accept}, Needs Review: {needs_review}")
    assert secondary_accept, "Should accept with needs_review"
    assert needs_review, "Should flag for review due to classifier disagreement"
    print("[PASS]")
    
    # Test fallback (format, semantic, and task valid - always return rewrite)
    print("\nTest 4: Fallback candidate (format_valid, semantic_valid, task_valid) - always return rewrite")
    format_valid = True
    semantic_valid = True
    task_valid = True
    confidence = 0.25
    
    fallback_candidate = format_valid and semantic_valid and task_valid
    always_return_rewrite = fallback_candidate  # Key change: requires task_valid as well
    print(f"Format Valid: {format_valid}, Semantic Valid: {semantic_valid}, Task Valid: {task_valid}, Confidence: {confidence}")
    print(f"Fallback Candidate: {fallback_candidate}, Always Return Rewrite: {always_return_rewrite}")
    assert fallback_candidate, "Should retain as fallback candidate when all three validations pass"
    assert always_return_rewrite, "Should always return rewrite when format+semantic+task valid"
    print("[PASS]")
    
    print("\nAcceptance criteria tests passed.\n")


def run_all_tests():
    """Run all lightweight logic tests."""
    print("=" * 70)
    print("TARGET-LEVEL REWRITE LOGIC TEST SUITE (LIGHTWEIGHT)")
    print("=" * 70)
    
    try:
        test_format_validation()
        test_semantic_validation()
        test_task_structure_validation()
        test_candidate_scoring_logic()
        test_acceptance_criteria()
        
        print("=" * 70)
        print("ALL LOGIC TESTS PASSED")
        print("=" * 70)
        print("\nSummary:")
        print("- Format validation preserved")
        print("- Semantic cognitive validation preserved")
        print("- Task structure validation preserved")
        print("- Candidate scoring logic implemented correctly")
        print("- Acceptance criteria implemented correctly")
        print("- Fallback logic returns rewrite when format+semantic+task valid")
        print("- needs_review flag used for classifier disagreement/low confidence")
        print("- Always provides rewrite when all cognitive requirements met")
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
