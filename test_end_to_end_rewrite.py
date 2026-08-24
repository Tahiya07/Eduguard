"""
End-to-end test for target-level Bloom rewrite generation.

Tests the actual generator with real model calls to verify that rewrites
genuinely demonstrate the requested Bloom operations.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bloom_prompt import (
    rewrite_to_target_level,
    _validate_output_format,
    _semantic_cognitive_check_deterministic,
    _analyze_cognitive_task_structure,
    _canonical_bloom_label,
)


def test_end_to_end_generation():
    """Test actual generation with the provided test matrix."""
    print("=" * 70)
    print("END-TO-END BLOOM REWRITE GENERATION TEST")
    print("=" * 70)
    print()
    
    test_matrix = [
        ("Define virtual memory.", "Understand"),
        ("Explain how a compiler works.", "Apply"),
        ("Calculate the average of a dataset.", "Analyze"),
        ("Analyze the causes of a network failure.", "Evaluate"),
        ("Evaluate an existing database design.", "Create"),
        ("What is an operating system?", "Analyze"),
        ("Explain TCP/IP.", "Create"),
    ]
    
    results = []
    
    for original, target_level in test_matrix:
        print(f"\n{'=' * 70}")
        print(f"Original: {original}")
        print(f"Target: {target_level}")
        print(f"{'=' * 70}")
        
        try:
            # Generate rewrite with actual model
            rewrite, needs_review, error_message = rewrite_to_target_level(original, target_level)
            
            if not rewrite:
                print(f"\n[FAIL] Generation failed: {error_message}")
                results.append({
                    "original": original,
                    "target_level": target_level,
                    "generated_rewrite": "",
                    "predicted_level": "",
                    "confidence": 0.0,
                    "format_valid": False,
                    "semantic_valid": False,
                    "task_valid": False,
                    "needs_review": False,
                    "review_reason": error_message,
                    "acceptance_criterion": "Generation failed"
                })
                continue
            
            print(f"\nGenerated Rewrite: {rewrite}")
            print(f"Needs Review: {needs_review}")
            if error_message:
                print(f"Review Reason: {error_message}")
            
            # Perform detailed validation analysis
            format_valid, format_reason = _validate_output_format(rewrite)
            semantic_valid, semantic_reason = _semantic_cognitive_check_deterministic(rewrite, target_level)
            task_valid, task_reason = _analyze_cognitive_task_structure(rewrite, target_level)
            
            # Get classifier prediction
            predicted_level = ""
            confidence = 0.0
            try:
                from predict_bloom import QwenBloomPredictor
                predictor = QwenBloomPredictor()
                validation = predictor.predict(rewrite)
                predicted_level = _canonical_bloom_label(validation["prediction"])
                confidence = validation.get("confidence", 0.0)
                if confidence is None or not isinstance(confidence, (int, float)):
                    confidence = 0.0
            except Exception as e:
                predicted_level = "Unknown"
                confidence = 0.0
            
            # Determine the most important acceptance criterion
            if not format_valid:
                acceptance_criterion = "Failed format validation"
            elif not semantic_valid:
                acceptance_criterion = "Failed semantic validation (target-level cognitive operation)"
            elif not task_valid:
                acceptance_criterion = "Failed task structure validation"
            elif predicted_level == _canonical_bloom_label(target_level) and confidence >= 0.60:
                acceptance_criterion = "Full classifier validation"
            elif task_valid and confidence >= 0.40:
                acceptance_criterion = "Task structure valid with moderate confidence"
            elif task_valid:
                acceptance_criterion = "Task structure valid despite classifier disagreement"
            else:
                acceptance_criterion = "Fallback candidate with limited validation"
            
            print(f"\nValidation Results:")
            print(f"  Format Valid: {format_valid} ({format_reason})")
            print(f"  Semantic Valid: {semantic_valid} ({semantic_reason})")
            print(f"  Task Valid: {task_valid} ({task_reason})")
            print(f"  Predicted Level: {predicted_level}")
            print(f"  Confidence: {confidence:.0%}")
            print(f"  Most Important Acceptance Criterion: {acceptance_criterion}")
            
            results.append({
                "original": original,
                "target_level": target_level,
                "generated_rewrite": rewrite,
                "predicted_level": predicted_level,
                "confidence": confidence,
                "format_valid": format_valid,
                "semantic_valid": semantic_valid,
                "task_valid": task_valid,
                "needs_review": needs_review,
                "review_reason": error_message,
                "acceptance_criterion": acceptance_criterion
            })
            
        except Exception as e:
            print(f"\n[ERROR] Exception during generation: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "original": original,
                "target_level": target_level,
                "generated_rewrite": "",
                "predicted_level": "",
                "confidence": 0.0,
                "format_valid": False,
                "semantic_valid": False,
                "task_valid": False,
                "needs_review": False,
                "review_reason": str(e),
                "acceptance_criterion": "Exception during generation"
            })
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()
    
    for i, result in enumerate(results, 1):
        print(f"Test {i}:")
        print(f"  Original: {result['original']}")
        print(f"  Target: {result['target_level']}")
        print(f"  Generated: {result['generated_rewrite']}")
        print(f"  Predicted: {result['predicted_level']}")
        print(f"  Confidence: {result['confidence']:.0%}")
        print(f"  Format Valid: {result['format_valid']}")
        print(f"  Semantic Valid: {result['semantic_valid']}")
        print(f"  Task Valid: {result['task_valid']}")
        print(f"  Needs Review: {result['needs_review']}")
        print(f"  Review Reason: {result['review_reason']}")
        print(f"  Acceptance Criterion: {result['acceptance_criterion']}")
        print()
    
    # Count successes
    successful_rewrites = sum(1 for r in results if r['generated_rewrite'])
    fully_validated = sum(1 for r in results if r['generated_rewrite'] and r['format_valid'] and r['semantic_valid'] and r['task_valid'])
    
    print("=" * 70)
    print(f"Total Tests: {len(results)}")
    print(f"Successful Rewrites: {successful_rewrites}")
    print(f"Fully Validated (format+semantic+task): {fully_validated}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = test_end_to_end_generation()
