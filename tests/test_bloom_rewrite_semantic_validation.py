"""Regression tests for Bloom rewrite semantic/context validation."""
from __future__ import annotations

import re
import unittest

from bloom_prompt import rewrite_to_target_level, unsupported_context_check


class UnsupportedContextTests(unittest.TestCase):
    def test_rejects_conditions_that_are_not_in_the_source(self):
        original = "Explain peer to peer network."
        rewrite = "Predict the outcome of peer-to-peer network operations under the stated conditions."
        valid, reason = unsupported_context_check(original, rewrite)
        self.assertFalse(valid)
        self.assertIn("unsupported context", reason.lower())

    def test_rejects_unfounded_prediction_even_without_the_conditions_phrase(self):
        original = "Explain peer to peer network."
        rewrite = "Predict the outcome of peer-to-peer network operations."
        valid, reason = unsupported_context_check(original, rewrite)
        self.assertFalse(valid)
        self.assertIn("unsupported context", reason.lower())

    def test_allows_conditions_that_are_present_in_the_source(self):
        original = "Given the stated conditions, explain how a peer-to-peer network operates."
        rewrite = "Predict the outcome of peer-to-peer network operations under the stated conditions."
        valid, reason = unsupported_context_check(original, rewrite)
        self.assertTrue(valid, reason)

    def test_target_rewrite_uses_one_generation_attempt_on_verb_mismatch(self):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def reset(self):
                pass

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                return {"choices": [{"text": "Judge the client-server architecture."}]}

        class PredictorMustNotRun:
            def predict(self, question):
                raise AssertionError("Classifier must not run after prevalidation failure")

        llm = FakeLLM()
        result = rewrite_to_target_level(
            "Describe the client-server architecture.",
            "Evaluate",
            llm=llm,
            predictor=PredictorMustNotRun(),
        )
        self.assertEqual(llm.calls, 1)
        self.assertTrue(result.needs_review)
        self.assertTrue(result.validation_error)
        self.assertIn("did not follow the selected action verb", result.note.lower())

    def test_generation_does_not_mark_context_drift_as_valid(self):
        class FakeLLM:
            def reset(self):
                pass

            def __call__(self, prompt, **kwargs):
                match = re.search(
                    r"required action verb: ([A-Za-z]+)", prompt, flags=re.IGNORECASE
                )
                self.asserted_verb = match.group(1) if match else "Predict"
                return {
                    "choices": [{
                        "text": (
                            f"{self.asserted_verb} the outcome of peer-to-peer network "
                            "operations under the stated conditions."
                        )
                    }]
                }

        class PredictorMustNotRun:
            def predict(self, question):
                raise AssertionError("Classifier must not validate a semantically invalid rewrite")

        result = rewrite_to_target_level(
            "Explain peer to peer network.",
            "Apply",
            llm=FakeLLM(),
            predictor=PredictorMustNotRun(),
        )
        self.assertTrue(result.needs_review)
        self.assertTrue(result.validation_error)
        self.assertIn("semantic validation", result.note.lower())
        self.assertIn("unsupported context", result.note.lower())


if __name__ == "__main__":
    unittest.main()
