"""
Test script to verify cognitive task structure analysis without requiring full dependencies.
This tests the regex-based analysis function independently.
"""

import sys
import re
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Define the function directly to test it without dependencies
def _analyze_cognitive_task_structure(question: str, target_level: str):
    """
    Analyze the cognitive task structure to determine if the dominant operation matches the target level.
    
    This function examines what the STUDENT must DO, not just which words appear in the sentence.
    
    Returns:
        tuple: (is_valid, reason)
    """
    question_lower = question.lower()
    
    # Define dominant cognitive patterns for each level
    # These patterns focus on the TASK demanded from the student, not subject matter
    dominant_patterns = {
        "Remember": [
            r"\b(identify|list|name|define|recognize|recall|state|label)\b.*\s+(the|a|an|what|which|who)",
            r"\b(identify|list|name|define|recognize|recall|state|label)\b.*\s+(components|parts|elements|steps|stages|types|kinds|examples)",
        ],
        "Understand": [
            r"\b(explain|describe|summarize|interpret|classify|discuss)\b.*\s+(the|a|an|how|why|what)",
            r"\b(explain|describe|summarize|interpret|classify|discuss)\b.*\s+(principles|concepts|ideas|meaning|relationship|function)",
        ],
        "Apply": [
            r"\b(apply|use|demonstrate|implement|solve|execute)\b.*\s+(to|for|in|with)",
            r"\b(apply|use|demonstrate|implement|solve|execute)\b.*\s+(problem|situation|case|scenario|example)",
        ],
        "Analyze": [
            r"\b(analyze|compare|differentiate|examine|investigate|categorize)\b.*\s+(the|a|an|and|between|with)",
            r"\b(analyze|compare|differentiate|examine|investigate|categorize)\b.*\s+(relationship|differences|similarities|patterns|structure|components)",
        ],
        "Evaluate": [
            r"\b(evaluate|assess|justify|critique|defend|judge)\b.*\s+(the|a|an|using|based on|according to)",
            r"\b(evaluate|assess|justify|critique|defend|judge)\b.*\s+(effectiveness|quality|validity|merits|strengths|weaknesses)",
        ],
        "Create": [
            r"\b(design|develop|construct|formulate|propose|create)\b.*\s+(a|an|the)",
            r"\b(design|develop|construct|formulate|propose|create)\b.*\s+(system|model|plan|solution|product|application|structure)",
        ],
    }
    
    # Define problematic task structures where the dominant operation doesn't match
    # These focus on multi-verb constructions where the actual task is higher than the apparent level
    problematic_structures = {
        "Understand": [
            r"\b(explain)\b.*\s+(how to design|how to develop|how to create|how to construct)",
            r"\b(describe)\b.*\s+(how to design|how to develop|how to create|how to construct)",
            r"\b(explain and design|explain and develop|explain and create|explain and construct)\b",
            r"\b(design and explain|develop and explain|create and explain|construct and explain)\b",
        ],
        "Analyze": [
            r"\b(analyze and design|analyze and develop|analyze and create|analyze and construct)\b",
            r"\b(compare and design|compare and develop|compare and create)\b",
            r"\b(design and analyze|develop and analyze|create and analyze)\b",
        ],
        "Evaluate": [
            r"\b(explain and evaluate|describe and evaluate)\b",
            r"\b(evaluate and explain|evaluate and describe)\b",
        ],
        "Remember": [
            r"\b(define and design|define and create|define and develop|define and construct)\b",
            r"\b(design and define|create and define|develop and define)\b",
        ],
        "Apply": [
            r"\b(apply and design|apply and create|apply and develop)\b",
            r"\b(design and apply|create and apply|develop and apply)\b",
        ],
    }
    
    # Check for problematic structures first
    if target_level in problematic_structures:
        for pattern in problematic_structures[target_level]:
            if re.search(pattern, question_lower):
                return False, f"Question contains problematic construction that doesn't match {target_level} cognitive task."
    
    # Check if the question contains dominant patterns for the target level
    if target_level in dominant_patterns:
        has_dominant_pattern = False
        for pattern in dominant_patterns[target_level]:
            if re.search(pattern, question_lower):
                has_dominant_pattern = True
                break
        
        if not has_dominant_pattern:
            return False, f"Question lacks clear dominant cognitive pattern for {target_level}."
    
    return True, "Task structure analysis passed."

def test_cognitive_task_structure():
    """Test the cognitive task structure analysis function."""
    print("Testing cognitive task structure analysis...")
    
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

if __name__ == "__main__":
    print("=" * 60)
    print("COGNITIVE TASK STRUCTURE ANALYSIS TEST")
    print("=" * 60)
    print()
    
    # Test cognitive task structure analysis
    analysis_ok = test_cognitive_task_structure()
    
    print()
    print("=" * 60)
    if analysis_ok:
        print("ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)