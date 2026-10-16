"""Regression tests for production target-level rewrite validation and retry."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import bloom_prompt  # noqa: E402


class _Predictor:
    def predict(self, text):
        return {"prediction": "Analyze", "confidence": 0.91}


class TargetRewriteRuntimeTests(unittest.TestCase):
    def test_target_cognitive_screen_checks_task_not_only_classifier(self):
        check = bloom_prompt._target_cognitive_operation_check

        self.assertTrue(check(
            "Analyze how database index choices affect query performance.", "Analyze"
        )[0])
        self.assertFalse(check(
            "Analyze database indexing.", "Analyze"
        )[0])
        self.assertTrue(check(
            "Evaluate two indexing methods against query speed and storage-cost criteria.", "Evaluate"
        )[0])
        self.assertFalse(check(
            "Evaluate two indexing methods.", "Evaluate"
        )[0])
        self.assertTrue(check(
            "Design a database indexing system that satisfies the stated latency constraints.", "Create"
        )[0])
        self.assertFalse(check("Create a database indexing strategy.", "Create")[0])

    def test_retry_prompt_adds_failure_correction_inside_user_turn(self):
        prompt = bloom_prompt.build_targeted_rewrite_prompt(
            "Describe database indexes.",
            target_level="Analyze",
            previous_failure="Classifier predicted Understand; target is Analyze.",
        )
        self.assertIn("Correction after validation failure:", prompt)
        self.assertIn("Classifier predicted Understand", prompt)
        self.assertIn("Produce a corrected Analyze task.", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("Source Bloom level:", prompt)

    def test_rewrite_retries_once_and_returns_validation_details(self):
        validations = [
            {
                "cleaned_output": "Explain database indexes.",
                "format_valid": True,
                "semantic_similarity": 1.0,
                "final_validation": False,
                "classifier_prediction": "Understand",
                "classifier_confidence": 0.9,
                "failure_reason": "Classifier predicted Understand; target is Analyze.",
            },
            {
                "cleaned_output": "Analyze how database index choices affect query performance.",
                "format_valid": True,
                "semantic_similarity": 0.8,
                "final_validation": True,
                "classifier_prediction": "Analyze",
                "classifier_confidence": 0.91,
                "failure_reason": "",
            },
        ]
        prompts = []

        def fake_generate(model, prompt, max_tokens=None):
            prompts.append(prompt)
            return (
                "Explain database indexes."
                if len(prompts) == 1
                else "Analyze how database index choices affect query performance."
            )

        with (
            patch.object(bloom_prompt, "generate_v3_rewrite", side_effect=fake_generate),
            patch.object(
                bloom_prompt,
                "_validate_generated_candidate",
                side_effect=validations,
            ),
        ):
            result = bloom_prompt.rewrite_to_target_level(
                "Describe database indexes.",
                "Analyze",
                llm=object(),
                predictor=_Predictor(),
                max_tokens=64,
            )

        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.validation_match)
        self.assertFalse(result.needs_review)
        self.assertEqual(result.predicted_level, "Analyze")
        self.assertEqual(result.validation_details["classifier_prediction"], "Analyze")
        self.assertIn("Correction after validation failure:", prompts[1])
        self.assertIn("target is Analyze", prompts[1])

    def test_retry_is_bounded_and_failed_result_requires_review(self):
        failed = {
            "cleaned_output": "Explain database indexes.",
            "format_valid": True,
            "semantic_similarity": 1.0,
            "final_validation": False,
            "classifier_prediction": "Understand",
            "classifier_confidence": 0.9,
            "failure_reason": "Classifier predicted Understand; target is Analyze.",
        }
        with (
            patch.object(
                bloom_prompt, "generate_v3_rewrite", return_value="Explain database indexes."
            ) as generate,
            patch.object(
                bloom_prompt, "_validate_generated_candidate", return_value=failed
            ),
        ):
            result = bloom_prompt.rewrite_to_target_level(
                "Describe database indexes.",
                "Analyze",
                llm=object(),
                predictor=_Predictor(),
                max_attempts=10,
            )
        self.assertEqual(result.attempts, 2)
        self.assertEqual(generate.call_count, 2)
        self.assertTrue(result.needs_review)
        self.assertFalse(result.validation_match)
        self.assertIn("Validation failed after 2 attempt(s)", result.note)

    def test_missing_predictor_never_marks_rewrite_as_verified(self):
        with patch.object(
            bloom_prompt, "generate_v3_rewrite", return_value="Analyze database index trade-offs."
        ):
            result = bloom_prompt.rewrite_to_target_level(
                "Describe database indexes.",
                "Analyze",
                llm=object(),
                predictor=None,
            )
        self.assertTrue(result.needs_review)
        self.assertTrue(result.validation_error)
        self.assertFalse(result.validation_match)
        self.assertEqual(result.attempts, 1)


if __name__ == "__main__":
    unittest.main()
