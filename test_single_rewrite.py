"""
Test a single rewrite to debug the end-to-end generation.
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


def test_single():
    """Test a single case."""
    print("Testing single rewrite...")
    
    original = "Define virtual memory."
    target_level = "Understand"
    
    print(f"Original: {original}")
    print(f"Target: {target_level}")
    
    try:
        rewrite, needs_review, error_message = rewrite_to_target_level(original, target_level)
        
        print(f"\nGenerated Rewrite: {rewrite}")
        print(f"Needs Review: {needs_review}")
        print(f"Error Message: {error_message}")
        
        if rewrite:
            format_valid, format_reason = _validate_output_format(rewrite)
            semantic_valid, semantic_reason = _semantic_cognitive_check_deterministic(rewrite, target_level)
            task_valid, task_reason = _analyze_cognitive_task_structure(rewrite, target_level)
            
            print(f"\nFormat Valid: {format_valid} - {format_reason}")
            print(f"Semantic Valid: {semantic_valid} - {semantic_reason}")
            print(f"Task Valid: {task_valid} - {task_reason}")
            
            try:
                from predict_bloom import QwenBloomPredictor
                predictor = QwenBloomPredictor()
                validation = predictor.predict(rewrite)
                predicted_level = _canonical_bloom_label(validation["prediction"])
                confidence = validation.get("confidence", 0.0)
                print(f"Predicted Level: {predicted_level}")
                print(f"Confidence: {confidence:.0%}")
            except Exception as e:
                print(f"Classifier error: {e}")
        else:
            print("No rewrite generated")
            
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_single()
