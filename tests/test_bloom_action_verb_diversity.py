"""Regression tests for inference-time Bloom action-verb selection."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "multitask_bloom_rewrite"
    / "prompts.py"
)
SPEC = importlib.util.spec_from_file_location("eduguard_multitask_prompts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROMPTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROMPTS)


class ActionVerbSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_for_same_input(self):
        first = PROMPTS.select_action_verbs(
            "Explain how an algorithm processes input and produces output.", "Analyze"
        )
        second = PROMPTS.select_action_verbs(
            "Explain how an algorithm processes input and produces output.", "Analyze"
        )
        self.assertEqual(first, second)

    def test_selected_verbs_belong_to_requested_level(self):
        for level in ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"):
            with self.subTest(level=level):
                verbs = PROMPTS.select_action_verbs("Explain the system and its operation.", level)
                self.assertTrue(verbs)
                self.assertEqual(len(verbs), len(set(verbs)))
                self.assertTrue(all(verb in PROMPTS._ACTION_VERB_BANKS[level] for verb in verbs))

    def test_apply_uses_calculation_verbs_for_numeric_questions(self):
        verbs = PROMPTS.select_action_verbs(
            "Calculate the average response time from these measurements.", "Apply"
        )
        self.assertIn(verbs[0], {"Calculate", "Compute", "Determine", "Estimate", "Solve", "Apply"})

    def test_broad_apply_question_does_not_default_to_prediction_without_conditions(self):
        verbs = PROMPTS.select_action_verbs(
            "Explain peer to peer network.", "Apply"
        )
        self.assertNotIn("Predict", verbs)
        self.assertNotIn("Calculate", verbs)
        self.assertTrue(set(verbs) & {"Demonstrate", "Use", "Apply", "Implement", "Execute"})

    def test_production_target_verbs_are_fixed_to_avoid_retry_drift(self):
        expected = {"Apply": "Demonstrate", "Analyze": "Analyze", "Evaluate": "Assess"}
        for level, verb in expected.items():
            with self.subTest(level=level):
                self.assertEqual(
                    PROMPTS.select_action_verbs("Describe any academic concept.", level),
                    [verb],
                )

    def test_selected_verb_is_an_explicit_prompt_requirement(self):
        messages = PROMPTS.bloom_messages(
            "Explain how virtual memory handles a page fault.",
            "Apply",
            selected_action_verb="Determine",
        )
        user_prompt = messages[-1]["content"]
        self.assertIn("required action verb: Determine", user_prompt)
        self.assertIn("exact opening verb", user_prompt.lower())


if __name__ == "__main__":
    unittest.main()
