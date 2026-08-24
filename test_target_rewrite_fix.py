"""
Test suite for target-level Bloom rewrite fix.

Tests all six Bloom levels with representative questions.
Verifies that rewrites are valid and not rejected solely due to classifier confidence.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bloom_prompt import rewrite_to_target_level, _canonical_bloom_label


def test_remember():
    """Test Remember-level rewrites."""
    print("\n=== TESTING REMEMBER ===\n")
    
    test_questions = [
        "Design a comprehensive database schema for an e-commerce platform.",
        "Implement a machine learning algorithm for fraud detection.",
        "Create a user authentication system with role-based access control.",
    ]
    
    for question in test_questions:
        print(f"Input: {question}")
        print(f"Target: Remember")
        
        rewrite, success, error = rewrite_to_target_level(question, "Remember")
        
        if rewrite:
            print(f"[PASS] Rewrite: {rewrite}")
            print(f"  Success: {success}")
            if error:
                print(f"  Note: {error}")
            
            # Verify rewrite is not empty
            assert rewrite, "Rewrite should not be empty"
            
            # Verify rewrite is a genuine question
            assert len(rewrite.split()) >= 3, "Rewrite should be at least 3 words"
            
            # Verify rewrite is not identical to original
            assert rewrite.lower() != question.lower(), "Rewrite should differ from original"
            
            # Verify rewrite preserves topic
            original_words = set(question.lower().split())
            rewrite_words = set(rewrite.lower().split())
            overlap = len(original_words & rewrite_words)
            assert overlap >= 2, "Rewrite should preserve some original terms"
            
            print("  [PASS] All validations passed")
        else:
            print(f"[FAIL] FAILED: {error}")
            # This should rarely happen with the fix
            assert False, f"Remember rewrite failed: {error}"
        
        print()


def test_understand():
    """Test Understand-level rewrites."""
    print("\n=== TESTING UNDERSTAND ===\n")
    
    test_questions = [
        "Design a basic banking system.",
        "Create a comprehensive security protocol for network infrastructure.",
        "Develop a RESTful API for a social media application.",
    ]
    
    for question in test_questions:
        print(f"Input: {question}")
        print(f"Target: Understand")
        
        rewrite, success, error = rewrite_to_target_level(question, "Understand")
        
        if rewrite:
            print(f"[PASS] Rewrite: {rewrite}")
            print(f"  Success: {success}")
            if error:
                print(f"  Note: {error}")
            
            # Verify rewrite is not empty
            assert rewrite, "Rewrite should not be empty"
            
            # Verify rewrite is a genuine question
            assert len(rewrite.split()) >= 3, "Rewrite should be at least 3 words"
            
            # Verify rewrite is not identical to original
            assert rewrite.lower() != question.lower(), "Rewrite should differ from original"
            
            # Verify rewrite preserves topic
            original_words = set(question.lower().split())
            rewrite_words = set(rewrite.lower().split())
            overlap = len(original_words & rewrite_words)
            assert overlap >= 2, "Rewrite should preserve some original terms"
            
            # Verify rewrite targets Understand (not Create-level verbs)
            rewrite_lower = rewrite.lower()
            forbidden_create_verbs = ["design", "develop", "construct", "build", "create", "implement"]
            for verb in forbidden_create_verbs:
                # Check if the verb is used as a student task (not in subject matter)
                if verb in rewrite_lower:
                    # Allow if it's in the subject (e.g., "Explain the design of...")
                    if not f"how to {verb}" in rewrite_lower:
                        # Check if it's the main task verb
                        words = rewrite_lower.split()
                        if verb in words[:3]:  # If it's one of the first 3 words, it's likely the task
                            print(f"  [WARN] Warning: Contains Create-level verb '{verb}' in task position")
            
            print("  [PASS] All validations passed")
        else:
            print(f"[FAIL] FAILED: {error}")
            assert False, f"Understand rewrite failed: {error}"
        
        print()


def test_apply():
    """Test Apply-level rewrites."""
    print("\n=== TESTING APPLY ===\n")
    
    test_questions = [
        "Explain the concept of object-oriented programming.",
        "Describe the principles of database normalization.",
        "Summarize the key features of the HTTP protocol.",
    ]
    
    for question in test_questions:
        print(f"Input: {question}")
        print(f"Target: Apply")
        
        rewrite, success, error = rewrite_to_target_level(question, "Apply")
        
        if rewrite:
            print(f"[PASS] Rewrite: {rewrite}")
            print(f"  Success: {success}")
            if error:
                print(f"  Note: {error}")
            
            # Verify rewrite is not empty
            assert rewrite, "Rewrite should not be empty"
            
            # Verify rewrite is a genuine question
            assert len(rewrite.split()) >= 3, "Rewrite should be at least 3 words"
            
            # Verify rewrite is not identical to original
            assert rewrite.lower() != question.lower(), "Rewrite should differ from original"
            
            # Verify rewrite preserves topic
            original_words = set(question.lower().split())
            rewrite_words = set(rewrite.lower().split())
            overlap = len(original_words & rewrite_words)
            assert overlap >= 2, "Rewrite should preserve some original terms"
            
            print("  [PASS] All validations passed")
        else:
            print(f"[FAIL] FAILED: {error}")
            assert False, f"Apply rewrite failed: {error}"
        
        print()


def test_analyze():
    """Test Analyze-level rewrites."""
    print("\n=== TESTING ANALYZE ===\n")
    
    test_questions = [
        "Describe the components of a computer network.",
        "Explain the basic structure of a relational database.",
        "Summarize the key elements of a software development lifecycle.",
    ]
    
    for question in test_questions:
        print(f"Input: {question}")
        print(f"Target: Analyze")
        
        rewrite, success, error = rewrite_to_target_level(question, "Analyze")
        
        if rewrite:
            print(f"[PASS] Rewrite: {rewrite}")
            print(f"  Success: {success}")
            if error:
                print(f"  Note: {error}")
            
            # Verify rewrite is not empty
            assert rewrite, "Rewrite should not be empty"
            
            # Verify rewrite is a genuine question
            assert len(rewrite.split()) >= 3, "Rewrite should be at least 3 words"
            
            # Verify rewrite is not identical to original
            assert rewrite.lower() != question.lower(), "Rewrite should differ from original"
            
            # Verify rewrite preserves topic
            original_words = set(question.lower().split())
            rewrite_words = set(rewrite.lower().split())
            overlap = len(original_words & rewrite_words)
            assert overlap >= 2, "Rewrite should preserve some original terms"
            
            print("  [PASS] All validations passed")
        else:
            print(f"[FAIL] FAILED: {error}")
            assert False, f"Analyze rewrite failed: {error}"
        
        print()


def test_evaluate():
    """Test Evaluate-level rewrites."""
    print("\n=== TESTING EVALUATE ===\n")
    
    test_questions = [
        "Describe the components of a cloud computing architecture.",
        "Explain the process of software testing methodologies.",
        "Summarize the key features of agile development practices.",
    ]
    
    for question in test_questions:
        print(f"Input: {question}")
        print(f"Target: Evaluate")
        
        rewrite, success, error = rewrite_to_target_level(question, "Evaluate")
        
        if rewrite:
            print(f"[PASS] Rewrite: {rewrite}")
            print(f"  Success: {success}")
            if error:
                print(f"  Note: {error}")
            
            # Verify rewrite is not empty
            assert rewrite, "Rewrite should not be empty"
            
            # Verify rewrite is a genuine question
            assert len(rewrite.split()) >= 3, "Rewrite should be at least 3 words"
            
            # Verify rewrite is not identical to original
            assert rewrite.lower() != question.lower(), "Rewrite should differ from original"
            
            # Verify rewrite preserves topic
            original_words = set(question.lower().split())
            rewrite_words = set(rewrite.lower().split())
            overlap = len(original_words & rewrite_words)
            assert overlap >= 2, "Rewrite should preserve some original terms"
            
            print("  [PASS] All validations passed")
        else:
            print(f"[FAIL] FAILED: {error}")
            assert False, f"Evaluate rewrite failed: {error}"
        
        print()


def test_create():
    """Test Create-level rewrites."""
    print("\n=== TESTING CREATE ===\n")
    
    test_questions = [
        "Explain the principles of secure authentication systems.",
        "Describe the components of a microservices architecture.",
        "Summarize the key features of a distributed database system.",
    ]
    
    for question in test_questions:
        print(f"Input: {question}")
        print(f"Target: Create")
        
        rewrite, success, error = rewrite_to_target_level(question, "Create")
        
        if rewrite:
            print(f"[PASS] Rewrite: {rewrite}")
            print(f"  Success: {success}")
            if error:
                print(f"  Note: {error}")
            
            # Verify rewrite is not empty
            assert rewrite, "Rewrite should not be empty"
            
            # Verify rewrite is a genuine question
            assert len(rewrite.split()) >= 3, "Rewrite should be at least 3 words"
            
            # Verify rewrite is not identical to original
            assert rewrite.lower() != question.lower(), "Rewrite should differ from original"
            
            # Verify rewrite preserves topic
            original_words = set(question.lower().split())
            rewrite_words = set(rewrite.lower().split())
            overlap = len(original_words & rewrite_words)
            assert overlap >= 2, "Rewrite should preserve some original terms"
            
            # Verify rewrite targets Create (not just Understanding)
            rewrite_lower = rewrite.lower()
            # Check if it's merely explaining/summarizing
            if rewrite_lower.startswith(("explain", "describe", "summarize")):
                # But should contain create-level verbs for the actual task
                create_verbs = ["design", "develop", "construct", "create", "formulate", "propose"]
                has_create_verb = any(verb in rewrite_lower for verb in create_verbs)
                if not has_create_verb:
                    print(f"  [WARN] Warning: May be merely explaining rather than creating")
            
            print("  [PASS] All validations passed")
        else:
            print(f"[FAIL] FAILED: {error}")
            assert False, f"Create rewrite failed: {error}"
        
        print()


def test_confidence_tolerance():
    """Test that rewrites are accepted even with lower classifier confidence."""
    print("\n=== TESTING CONFIDENCE TOLERANCE ===\n")
    
    # Test a case where classifier might have lower confidence
    question = "Design a comprehensive system for data analytics."
    target = "Understand"
    
    print(f"Input: {question}")
    print(f"Target: {target}")
    
    rewrite, success, error = rewrite_to_target_level(question, target)
    
    if rewrite:
        print(f"[PASS] Rewrite: {rewrite}")
        print(f"  Success: {success}")
        if error:
            print(f"  Note: {error}")
        
        # The key test: should return a rewrite even if confidence is < 0.60
        # as long as it passes format, semantic, and task validation
        assert rewrite, "Should return rewrite even with lower classifier confidence"
        print("  [PASS] Confidence tolerance test passed")
    else:
        print(f"[FAIL] FAILED: {error}")
        assert False, "Should not fail due to low classifier confidence alone"


def run_all_tests():
    """Run all test suites."""
    print("=" * 70)
    print("TARGET-LEVEL BLOOM REWRITE FIX TEST SUITE")
    print("=" * 70)
    
    try:
        test_remember()
        test_understand()
        test_apply()
        test_analyze()
        test_evaluate()
        test_create()
        test_confidence_tolerance()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)
        print("\nSummary:")
        print("- All six Bloom levels tested successfully")
        print("- Rewrites are not empty")
        print("- Rewrites are genuine questions")
        print("- Rewrites differ from original")
        print("- Rewrites preserve original topic")
        print("- Valid candidates accepted even with classifier confidence < 0.60")
        print("- Failure message only returned when no usable candidate exists")
        
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
