"""Regression tests for Bloom prompts and conservative validation heuristics."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "multitask_bloom_rewrite"
sys.path.insert(0, str(EXPERIMENT))

from bloom_validation import _cognitive_valid, semantic_similarity  # noqa: E402
from prompts import BLOOM_LEVEL_GUIDANCE, build_generation_prompt  # noqa: E402
from scripts.prepare_multitask_dataset_final import validate_annotation_provenance  # noqa: E402


class BloomPromptTests(unittest.TestCase):
    def test_all_six_levels_have_operational_guidance(self):
        self.assertEqual(
            set(BLOOM_LEVEL_GUIDANCE),
            {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"},
        )
        for guidance in BLOOM_LEVEL_GUIDANCE.values():
            self.assertGreater(len(guidance), 50)

    def test_shared_system_prompt_requires_action_led_target_task(self):
        from prompts import BLOOM_SYSTEM

        self.assertIn("requested target level is authoritative", BLOOM_SYSTEM)
        self.assertIn("action-verb-led", BLOOM_SYSTEM)
        self.assertIn("Do not merely swap the opening verb", BLOOM_SYSTEM)

    def test_generation_prompt_contains_target_operation_not_source_level(self):
        row = {
            "source_question": "Describe two database indexing methods.",
            "source_bloom_level": "Understand",
            "target_bloom_level": "Evaluate",
        }
        prompt = build_generation_prompt("bloom_rewrite", row)
        self.assertIn("Required cognitive operation:", prompt)
        self.assertIn("criteria", prompt.lower())
        self.assertNotIn("Source Bloom level:", prompt)
        self.assertNotIn("Original Bloom level:", prompt)

    def test_invalid_target_level_fails_loudly(self):
        with self.assertRaises(ValueError):
            build_generation_prompt(
                "bloom_rewrite",
                {"source_question": "Explain indexing.", "target_bloom_level": "Expert"},
            )


class BloomHeuristicTests(unittest.TestCase):
    def test_long_but_wrong_cognitive_task_does_not_pass_evaluate(self):
        self.assertFalse(
            _cognitive_valid(
                "What are the main components of a database indexing system and how are they named?",
                "Evaluate",
            )
        )

    def test_evaluate_requires_judgment_basis(self):
        self.assertTrue(
            _cognitive_valid(
                "Evaluate the two indexing methods against query speed and storage cost criteria, and justify your recommendation.",
                "Evaluate",
            )
        )
        self.assertFalse(
            _cognitive_valid("Evaluate the two indexing methods.", "Evaluate")
        )

    def test_create_requires_a_product_or_constraint(self):
        self.assertTrue(
            _cognitive_valid(
                "Design a database indexing strategy that satisfies the stated storage and query-latency constraints.",
                "Create",
            )
        )
        self.assertFalse(_cognitive_valid("Create a database indexing strategy.", "Create"))

    def test_lexical_score_is_jaccard_not_claimed_semantics(self):
        score = semantic_similarity("Compare database indexing methods", "Compare database indexing methods")
        self.assertEqual(score, 1.0)
        self.assertEqual(
            semantic_similarity("database indexing methods", "photosynthesis"),
            0.0,
        )

    def test_identical_reviewer_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = (
                "example_id,bloom_aligned,reviewer_bloom_level,content_preserved,"
                "context_preserved,meaningful_transformation,pedagogically_valid,"
                "clear_and_grammatical,final_decision\\n"
            )
            (root / "reviewer_1.csv").write_text(header, encoding="utf-8")
            (root / "reviewer_2.csv").write_text(header, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                validate_annotation_provenance(root, [], [])

    def test_unmarked_rows_are_not_treated_as_human_validated(self):
        from scripts.prepare_multitask_dataset_final import normalize_bloom

        unreviewed = [{
            "source_id": "s1",
            "group_id": "g1",
            "source_question": "Explain database indexing.",
            "target_bloom_level": "Analyze",
            "target_rewrite": "Analyze the relationship between indexing choices and query performance.",
            "example_id": "e1",
        }]
        with self.assertRaisesRegex(ValueError, "No human-validated"):
            normalize_bloom(unreviewed, "train")


if __name__ == "__main__":
    unittest.main()
