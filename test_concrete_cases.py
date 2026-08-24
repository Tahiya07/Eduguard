"""
Test the new compact prompt with the concrete cases specified.
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


def test_concrete_cases():
    """Test the concrete cases specified."""
    print("=" * 70)
    print("CONCRETE CASES TEST - NEW COMPACT PROMPT")
    print("=" * 70)
    print()
    
    test_cases = [
        ("Explain what virtual memory is.", "Understand"),
        ("Explain what virtual memory is.", "Apply"),
        ("Explain what virtual memory is.", "Analyze"),
    ]
    
    results = []
    
    for original, target_level in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Original: {original}")
        print(f"Target: {target_level}")
        print(f"{'=' * 70}")
        
        try:
            rewrite, needs_review, error_message = rewrite_to_target_level(original, target_level)
            
            if not rewrite:
                print(f"\n[FAIL] Generation failed: {error_message}")
                results.append({
                    "original": original,
                    "target_level": target_level,
                    "generated_rewrite": "",
                    "is_question": False,
                    "topic_preserved": False,
                    "cognitively_valid": False,
                    "not_verb_substitution": False,
                    "not_explanation": False,
                    "not_meta_language": False,
                    "error": error_message
                })
                continue
            
            print(f"\nGenerated Rewrite: {rewrite}")
            print(f"Needs Review: {needs_review}")
            if error_message:
                print(f"Review Reason: {error_message}")
            
            # Perform detailed validation
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
            
            print(f"\nValidation Results:")
            print(f"  Format Valid: {format_valid} ({format_reason})")
            print(f"  Semantic Valid: {semantic_valid} ({semantic_reason})")
            print(f"  Task Valid: {task_valid} ({task_reason})")
            print(f"  Predicted Level: {predicted_level}")
            print(f"  Confidence: {confidence:.0%}")
            
            # Check if it's a question
            is_question = format_valid
            
            # Check if topic is preserved
            original_words = set(original.lower().split())
            rewrite_words = set(rewrite.lower().split())
            topic_overlap = len(original_words & rewrite_words)
            topic_preserved = topic_overlap >= 2
            
            # Check if cognitively valid
            cognitively_valid = semantic_valid and task_valid
            
            # Check if not merely verb substitution
            # This is harder to test programmatically, but we can check if the cognitive structure is valid
            not_verb_substitution = task_valid  # If task structure is valid, it's not just verb substitution
            
            # Check if not an explanation
            not_explanation = format_valid and not rewrite.lower().startswith(("this is", "it is", "the answer is", "the purpose is"))
            
            # Check if not meta-language
            not_meta_language = format_valid  # format_valid already checks this
            
            print(f"\nQuality Checks:")
            print(f"  Is Question: {is_question}")
            print(f"  Topic Preserved: {topic_preserved} (overlap: {topic_overlap} words)")
            print(f"  Cognitively Valid: {cognitively_valid}")
            print(f"  Not Verb Substitution: {not_verb_substitution}")
            print(f"  Not Explanation: {not_explanation}")
            print(f"  Not Meta-Language: {not_meta_language}")
            
            results.append({
                "original": original,
                "target_level": target_level,
                "generated_rewrite": rewrite,
                "is_question": is_question,
                "topic_preserved": topic_preserved,
                "cognitively_valid": cognitively_valid,
                "not_verb_substitution": not_verb_substitution,
                "not_explanation": not_explanation,
                "not_meta_language": not_meta_language,
                "predicted_level": predicted_level,
                "confidence": confidence,
                "needs_review": needs_review,
                "error": ""
            })
            
        except Exception as e:
            print(f"\n[ERROR] Exception during generation: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "original": original,
                "target_level": target_level,
                "generated_rewrite": "",
                "is_question": False,
                "topic_preserved": False,
                "cognitively_valid": False,
                "not_verb_substitution": False,
                "not_explanation": False,
                "not_meta_language": False,
                "error": str(e)
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
        print(f"  Is Question: {result['is_question']}")
        print(f"  Topic Preserved: {result['topic_preserved']}")
        print(f"  Cognitively Valid: {result['cognitively_valid']}")
        print(f"  Not Verb Substitution: {result['not_verb_substitution']}")
        print(f"  Not Explanation: {result['not_explanation']}")
        print(f"  Not Meta-Language: {result['not_meta_language']}")
        if 'predicted_level' in result:
            print(f"  Predicted: {result['predicted_level']}")
            print(f"  Confidence: {result['confidence']:.0%}")
        if 'needs_review' in result:
            print(f"  Needs Review: {result['needs_review']}")
        if 'error' in result and result['error']:
            print(f"  Error: {result['error']}")
        print()
    
    # Count successes
    successful_rewrites = sum(1 for r in results if r['generated_rewrite'])
    fully_valid = sum(1 for r in results if r['generated_rewrite'] and r['is_question'] and r['topic_preserved'] and r['cognitively_valid'])
    
    print("=" * 70)
    print(f"Total Tests: {len(results)}")
    print(f"Successful Rewrites: {successful_rewrites}")
    print(f"Fully Valid (question + topic + cognitive): {fully_valid}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = test_concrete_cases()
