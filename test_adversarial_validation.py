"""
Comprehensive adversarial tests for the integrated validation pipeline.

Tests:
1. Valid questions without expected keywords are accepted when they genuinely demonstrate target cognitive operation
2. Superficial verb substitution is still rejected
3. All six Bloom levels
4. Five different subject domains (computer science, biology, mathematics, history, literature)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bloom_prompt import (
    _is_trivial_transformation,
    _validate_output_format,
    _semantic_cognitive_check_deterministic,
    _analyze_cognitive_task_structure,
)


def test_all_levels_and_domains():
    """Test all six Bloom levels across five subject domains."""
    print("=" * 70)
    print("COMPREHENSIVE ADVERSARIAL TESTS")
    print("=" * 70)
    print()
    
    # Test cases: (original, target, rewrite, should_accept, description)
    test_cases = [
        # Computer Science domain
        ("Define binary search.", "Remember", "What are the key steps in binary search?", True, "Valid Remember without 'list' keyword"),
        ("Explain binary search.", "Understand", "How does binary search operate on sorted data?", True, "Valid Understand with 'operate' (operation-based)"),
        ("Describe sorting.", "Apply", "How would you use sorting algorithms to organize database records?", True, "Valid Apply with 'use' (operation-based)"),
        ("List sorting algorithms.", "Analyze", "What distinguishes quicksort from mergesort in terms of time complexity?", True, "Valid Analyze with 'distinguishes' (operation-based)"),
        ("What is time complexity?", "Evaluate", "Which sorting algorithm performs best for large datasets and why?", True, "Valid Evaluate with comparison (operation-based)"),
        ("Analyze the algorithm.", "Create", "Propose a new sorting method that minimizes comparisons?", True, "Valid Create with 'propose' (operation-based)"),
        
        # Biology domain
        ("What is photosynthesis?", "Remember", "Identify the main stages of photosynthesis.", True, "Valid Remember with 'identify'"),
        ("Explain photosynthesis.", "Understand", "What is the purpose of chlorophyll in photosynthesis?", True, "Valid Understand with 'purpose' (operation-based)"),
        ("Describe respiration.", "Apply", "How would cellular respiration function during intense exercise?", True, "Valid Apply with 'function during' (operation-based)"),
        ("List cellular organelles.", "Analyze", "How do mitochondria and chloroplasts interact in plant cells?", True, "Valid Analyze with 'interact' (operation-based)"),
        ("What is natural selection?", "Evaluate", "Which adaptation strategy is most effective for survival in arid climates?", True, "Valid Evaluate with comparison (operation-based)"),
        ("Describe evolution.", "Create", "Develop a model showing how a new species might emerge from isolation?", True, "Valid Create with 'develop' (operation-based)"),
        
        # Mathematics domain
        ("Define derivative.", "Remember", "State the definition of a derivative.", True, "Valid Remember with 'state'"),
        ("Explain derivatives.", "Understand", "What is the geometric interpretation of a derivative?", True, "Valid Understand with 'interpretation' (operation-based)"),
        ("Describe integration.", "Apply", "How would you use integration to calculate the area under a curve?", True, "Valid Apply with 'use' (operation-based)"),
        ("List integration methods.", "Analyze", "What relates the fundamental theorem of calculus to integration techniques?", True, "Valid Analyze with 'relates' (operation-based)"),
        ("What is convergence?", "Evaluate", "Which series converges faster: geometric or harmonic?", True, "Valid Evaluate with comparison (operation-based)"),
        ("Describe differentiation.", "Create", "Formulate a new method for approximating derivatives with minimal error?", True, "Valid Create with 'formulate' (operation-based)"),
        
        # History domain
        ("Define industrial revolution.", "Remember", "Name the key inventions of the industrial revolution.", True, "Valid Remember with 'name'"),
        ("Explain industrial revolution.", "Understand", "What characterized the social changes during the industrial revolution?", True, "Valid Understand with 'characterized' (operation-based)"),
        ("Describe reformation.", "Apply", "How would reformation principles apply to modern religious movements?", True, "Valid Apply with 'apply' (operation-based)"),
        ("List world wars.", "Analyze", "What influenced the outcomes of World War I versus World War II?", True, "Valid Analyze with 'influenced' (operation-based)"),
        ("What is the cold war?", "Evaluate", "Which geopolitical strategy was more effective: containment or detente?", True, "Valid Evaluate with comparison (operation-based)"),
        ("Describe democracy.", "Create", "Construct a democratic system that balances individual rights with collective security?", True, "Valid Create with 'construct' (operation-based)"),
        
        # Literature domain
        ("Define metaphor.", "Remember", "Identify examples of metaphors in the text.", True, "Valid Remember with 'identify'"),
        ("Explain symbolism.", "Understand", "What is the author's purpose in using symbolism?", True, "Valid Understand with 'purpose' (operation-based)"),
        ("Describe irony.", "Apply", "How would dramatic irony function in a modern screenplay?", True, "Valid Apply with 'function' (operation-based)"),
        ("List literary devices.", "Analyze", "How do metaphor and simile differ in their effect on readers?", True, "Valid Analyze with 'differ' (operation-based)"),
        ("What is theme?", "Evaluate", "Which theme is more central: love or sacrifice?", True, "Valid Evaluate with comparison (operation-based)"),
        ("Describe narrative structure.", "Create", "Devise a narrative technique that reveals character through action rather than dialogue?", True, "Valid Create with 'devise' (operation-based)"),
        
        # Trivial transformations (should be rejected)
        ("Explain binary search.", "Understand", "How would you explain binary search?", False, "Trivial: 'explain' to 'how would you explain'"),
        ("What is binary search?", "Understand", "How would you explain what binary search is?", False, "Trivial: 'what is' wrapping"),
        ("Apply sorting.", "Apply", "How would you apply sorting?", False, "Trivial: 'apply' to 'how would you apply' without context"),
        ("Analyze the algorithm.", "Analyze", "How would you analyze the algorithm?", False, "Trivial: 'analyze' to 'how would you analyze' without analysis"),
        ("Design a system.", "Create", "How would you design a system?", False, "Trivial: 'design' to 'how would you design' without elaboration"),
    ]
    
    results = {
        "true_positives": 0,  # Valid transformations accepted
        "true_negatives": 0,  # Trivial transformations rejected
        "false_positives": 0,  # Valid transformations incorrectly rejected
        "false_negatives": 0,  # Trivial transformations incorrectly accepted
    }
    
    details = []
    
    for original, target, rewrite, should_accept, description in test_cases:
        print(f"Test: {description}")
        print(f"  Original: {original}")
        print(f"  Target: {target}")
        print(f"  Rewrite: {rewrite}")
        print(f"  Expected: {'Accept' if should_accept else 'Reject'}")
        
        # Run all validation layers
        format_valid, format_reason = _validate_output_format(rewrite)
        semantic_valid, semantic_reason = _semantic_cognitive_check_deterministic(rewrite, target)
        task_valid, task_reason = _analyze_cognitive_task_structure(rewrite, target)
        is_trivial, trivial_reason = _is_trivial_transformation(original, rewrite, target)
        
        # Determine overall acceptance
        accepted = format_valid and semantic_valid and task_valid and not is_trivial
        
        print(f"  Format: {format_valid} ({format_reason})")
        print(f"  Semantic: {semantic_valid} ({semantic_reason})")
        print(f"  Task: {task_valid} ({task_reason})")
        print(f"  Trivial: {is_trivial} ({trivial_reason})")
        print(f"  Actual: {'Accept' if accepted else 'Reject'}")
        
        # Track results
        if should_accept and accepted:
            results["true_positives"] += 1
            print(f"  [PASS] - Valid transformation accepted")
        elif not should_accept and not accepted:
            results["true_negatives"] += 1
            print(f"  [PASS] - Trivial transformation rejected")
        elif should_accept and not accepted:
            results["false_positives"] += 1
            print(f"  [FAIL] - False positive: Valid transformation rejected")
            details.append(f"FP: {description} - {format_reason} {semantic_reason} {task_reason} {trivial_reason}")
        elif not should_accept and accepted:
            results["false_negatives"] += 1
            print(f"  [FAIL] - False negative: Trivial transformation accepted")
            details.append(f"FN: {description}")
        
        print()
    
    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"True Positives (valid accepted): {results['true_positives']}")
    print(f"True Negatives (trivial rejected): {results['true_negatives']}")
    print(f"False Positives (valid rejected): {results['false_positives']}")
    print(f"False Negatives (trivial accepted): {results['false_negatives']}")
    print()
    
    if details:
        print("FAILURE DETAILS:")
        for detail in details:
            print(f"  {detail}")
    else:
        print("ALL TESTS PASSED")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = test_all_levels_and_domains()
