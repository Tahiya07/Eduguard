"""
Test target-level rewrite for the question: "Explain return on investment"
Test all six Bloom levels.
"""

import sys
sys.path.insert(0, ".")

from bloom_prompt import (
    rewrite_to_target_level,
    _canonical_bloom_label
)
from predict_bloom import QwenBloomPredictor

# Test question
test_question = "Explain return on investment"

# Test all six Bloom levels
levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

print("=" * 80)
print(f"Testing: {test_question}")
print("=" * 80)

for level in levels:
    print(f"\n--- Testing {level} ---")
    print(f"Target level: {level}")

    # Run the rewrite function
    rewrite, needs_review, error = rewrite_to_target_level(test_question, level)

    print(f"Rewrite: {rewrite}")
    print(f"Needs review: {needs_review}")
    print(f"Error: {error}")

    # Get classifier prediction if rewrite was generated
    if rewrite:
        try:
            predictor = QwenBloomPredictor()
            validation = predictor.predict(rewrite)
            predicted_level = _canonical_bloom_label(validation["prediction"])
            confidence = validation.get("confidence", 0.0)

            if confidence is None or not isinstance(confidence, (int, float)):
                confidence = 0.0

            print(f"Classifier prediction: {predicted_level}")
            print(f"Classifier confidence: {confidence:.2%}")
            print(f"Target level match: {predicted_level == _canonical_bloom_label(level)}")
        except Exception as e:
            print(f"Classifier prediction failed: {e}")

print("\n" + "=" * 80)
print("Test complete")
print("=" * 80)
