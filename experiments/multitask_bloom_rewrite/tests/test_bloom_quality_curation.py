"""Regression tests for conservative Bloom rewrite data curation."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "curate_bloom_rewrite_splits.py"
SPEC = importlib.util.spec_from_file_location("curate_bloom_rewrite_splits", SCRIPT)
assert SPEC and SPEC.loader
CURATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURATOR)


class BloomRewriteCurationTests(unittest.TestCase):
    def test_rejects_grafted_network_evaluate_example(self) -> None:
        row = {
            "source_question": (
                "Construct a family of networks to demonstrate that the number of "
                "different mincuts in a network can be exponential in the size of the network"
            ),
            "source_bloom_level": "Apply",
            "target_bloom_level": "Evaluate",
            "target_rewrite": (
                "Determine whether a family of networks to demonstrate that the number "
                "of different mincuts in a network can be exponential in the size of "
                "the network satisfies its requirements using relevant criteria and evidence."
            ),
        }
        reasons = CURATOR.find_rejection_reasons(row)
        self.assertIn("source_instruction_grafted_into_noun_phrase", reasons)
        self.assertIn("unsupported_generic_evaluation_rubric", reasons)

    def test_rejects_malformed_apply_template(self) -> None:
        row = {
            "source_question": "Explain why the story has the title that it does.",
            "source_bloom_level": "Understand",
            "target_bloom_level": "Apply",
            "target_rewrite": (
                "Use the principle underlying the story has the title that it does "
                "in a similar practical situation."
            ),
        }
        reasons = CURATOR.find_rejection_reasons(row)
        self.assertIn("principle_underlying_graft", reasons)
        self.assertIn("stock_apply_template", reasons)

    def test_keeps_topic_specific_evaluation_when_source_has_requirements(self) -> None:
        row = {
            "source_question": (
                "Evaluate whether the proposed sorting algorithm meets its stated "
                "time-complexity requirements."
            ),
            "source_bloom_level": "Evaluate",
            "target_bloom_level": "Evaluate",
            "target_rewrite": (
                "Assess whether the sorting algorithm meets its stated time-complexity "
                "requirements and justify the assessment."
            ),
        }
        self.assertEqual(CURATOR.find_rejection_reasons(row), [])

    def test_keeps_clear_topic_specific_apply_example(self) -> None:
        row = {
            "source_question": "Explain the Euclidean algorithm.",
            "source_bloom_level": "Understand",
            "target_bloom_level": "Apply",
            "target_rewrite": (
                "Apply the Euclidean algorithm to determine the greatest common divisor "
                "of two given integers."
            ),
        }
        self.assertEqual(CURATOR.find_rejection_reasons(row), [])


if __name__ == "__main__":
    unittest.main()
